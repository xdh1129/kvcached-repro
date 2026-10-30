"""Wake storm: controlled concurrent wakes on several GPUs, to measure data movement.

The instances in <run_dir>/instances.tsv are named g<gpu>_p<i> (pool models, which
take turns on their GPU) and g<gpu>_bg (an optional background model that stays
awake and decodes all the time). launch_servers.sh with SLEEP_AFTER_START=1 leaves
them all asleep. This driver then calls /sleep and /wake_up itself (no swap proxy),
so the timing of every wake is set by the experiment, not by a policy:

1. prime    per GPU, each pool model in turn: wake, send enough long requests
            (--prompt-len tokens each, ignore_eos, decoding up to max_model_len) that
            their KV is about --kv-gib, wait for every first token, then
            /sleep?level=1&mode=keep. The requests stay in flight, frozen, so every
            later wake of that model brings back its weights and that KV.
2. rounds   per round: every GPU puts its awake pool model to sleep (all at once),
            waits --settle-s, then wakes its next pool model. The first GPU starts
            its wake at T0, GPU i at T0 + i*delta. delta comes from --deltas, each
            value --reps times, in a shuffled order (so KV growth over the run is
            spread over all deltas). "seq": GPU i starts once GPU i-1's wake is done.
            Then --dwell-s with the woken models decoding.
All GPUs wake the same pool index in a round (same model, same bytes) unless
--stagger is set.

Writes to <run_dir>:
  swaps.jsonl         one line per sleep/wake: t_unix, gpu, instance, action, phase
                      (prime|round), round, delta, duration_s, gpu_used_mib_before/after,
                      inflight (requests frozen / resumed)
  requests.jsonl      pool requests, written at the end: instance, gpu, prompt_len,
                      max_tokens, sent_t, first_token_t, tokens, done_t (None if still
                      running at the end), gaps: [[start, end], ...] stream stalls longer
                      than --gap-s (unix times; a frozen request's stall spans its sleep)
  bg_requests.jsonl   background requests: instance, gpu, sent_t, first_token_t, done_t,
                      tokens, tpot_s, max_gap_s
  primed.tsv          per pool model: KV bytes per token, requests sent, expected KV GiB
The tiered_restore plugin's per-wake details (bytes and time per path) are in
tiered_<instance>.jsonl.

usage: python wake_storm.py <run_dir> [--kv-gib 5] [--prompt-len 7168] [--max-model-len 8192]
                            [--deltas 0,0.1,0.3,1,seq] [--reps 4] [--dwell-s 3] [--settle-s 1]
                            [--bg-concurrency 16] [--stagger] [--seed 0]
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import time

import aiohttp
import numpy as np
import pynvml
from huggingface_hub import try_to_load_from_cache

MIB = 1024 * 1024
GIB = 1 << 30


def kv_bytes_per_token(repo):
    """K and V bytes per token over all layers, in 16-bit, from the model's config.json."""
    path = try_to_load_from_cache(repo, "config.json")
    if not isinstance(path, str):
        raise RuntimeError(f"config.json of {repo} not in the HF cache")
    c = json.load(open(path))
    c = c.get("text_config", c)
    heads = c["num_attention_heads"]
    kv_heads = c.get("num_key_value_heads") or heads
    head_dim = c.get("head_dim") or c["hidden_size"] // heads
    return 2 * c["num_hidden_layers"] * kv_heads * head_dim * 2


class Inst:
    def __init__(self, name, repo, gpu, port):
        self.name, self.repo, self.gpu, self.port = name, repo, gpu, port
        self.url = f"http://localhost:{port}"
        self.inflight = 0


class Storm:
    def __init__(self, args):
        self.args = args
        self.run_dir = args.run_dir
        self.pool, self.bg = {}, {}
        for line in open(os.path.join(self.run_dir, "instances.tsv")).read().splitlines():
            name, repo, gpu, port = line.split("\t")
            inst = Inst(name, repo, int(gpu), int(port))
            m = re.fullmatch(r"g(\d+)_(p(\d+)|bg)", name)
            if not m:
                raise ValueError(f"unexpected instance name {name}")
            if m.group(2) == "bg":
                self.bg[inst.gpu] = inst
            else:
                self.pool.setdefault(inst.gpu, []).append((int(m.group(3)), inst))
        self.gpus = sorted(self.pool)
        self.pool = {g: [i for _, i in sorted(v, key=lambda x: x[0])] for g, v in self.pool.items()}
        pynvml.nvmlInit()
        self.handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in self.gpus}
        self.swaps = open(os.path.join(self.run_dir, "swaps.jsonl"), "a")
        self.bglog = open(os.path.join(self.run_dir, "bg_requests.jsonl"), "a")
        self.records = []  # pool requests, written at the end
        self.tasks = []
        self.stop = asyncio.Event()
        self.rng = np.random.default_rng(args.seed)
        self.session = None

    def used_mib(self, gpu):
        return pynvml.nvmlDeviceGetMemoryInfo(self.handles[gpu]).used // MIB

    async def swap(self, inst, action, **extra):
        before = self.used_mib(inst.gpu)
        url = inst.url + ("/wake_up" if action == "wake" else "/sleep?level=1&mode=keep")
        t = time.time()
        async with self.session.post(url) as r:
            if r.status != 200:
                raise RuntimeError(f"{action} {inst.name}: HTTP {r.status} {await r.text()}")
        dur = time.time() - t
        self.swaps.write(json.dumps({
            "t_unix": round(t, 4), "gpu": inst.gpu, "instance": inst.name, "action": action,
            "duration_s": round(dur, 4), "gpu_used_mib_before": before,
            "gpu_used_mib_after": self.used_mib(inst.gpu), "inflight": inst.inflight, **extra}) + "\n")
        self.swaps.flush()
        return t, dur

    def body(self, inst, prompt_len, max_tokens):
        return {"model": inst.repo, "prompt": self.rng.integers(1000, 30000, prompt_len).tolist(),
                "max_tokens": max_tokens, "min_tokens": max_tokens, "ignore_eos": True,
                "temperature": 0.0, "stream": True}

    async def stream(self, inst, body, rec, first_token=None):
        """Stream one completion, recording token count, first token and stalls > --gap-s."""
        inst.inflight += 1
        try:
            async with self.session.post(inst.url + "/v1/completions", json=body) as resp:
                if resp.status != 200:
                    rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                    return
                last = None
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    now = time.time()
                    if rec["first_token_t"] is None:
                        rec["first_token_t"] = now
                        if first_token is not None:
                            first_token.set()
                    elif now - last > self.args.gap_s:
                        rec["gaps"].append([round(last, 4), round(now, 4)])
                    last = now
                    rec["tokens"] += 1
                rec["done_t"] = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # recorded; the run goes on
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            inst.inflight -= 1
            if first_token is not None:
                first_token.set()  # do not block priming on a failed request

    # -- background load --

    async def bg_worker(self, inst):
        a = self.args
        while not self.stop.is_set():
            rec = {"instance": inst.name, "gpu": inst.gpu, "sent_t": time.time(), "first_token_t": None,
                   "done_t": None, "tokens": 0, "gaps": [], "error": None}
            await self.stream(inst, self.body(inst, a.bg_prompt_len, a.bg_output_len), rec)
            if rec["done_t"] and rec["tokens"] > 1:
                rec["tpot_s"] = (rec["done_t"] - rec["first_token_t"]) / (rec["tokens"] - 1)
            rec["max_gap_s"] = max((e - s for s, e in rec.pop("gaps")), default=0.0)
            self.bglog.write(json.dumps(rec) + "\n")
            self.bglog.flush()
            if rec["error"]:
                await asyncio.sleep(1)

    # -- phases --

    async def prime(self, gpu, primed):
        a = self.args
        max_tokens = a.max_model_len - a.prompt_len - 16
        for inst in self.pool[gpu]:
            per_tok = kv_bytes_per_token(inst.repo)
            n = max(1, math.ceil(a.kv_gib * GIB / (a.prompt_len * per_tok)))
            primed.append((inst.name, per_tok, n, n * a.prompt_len * per_tok / GIB))
            await self.swap(inst, "wake", phase="prime")
            events = []
            for _ in range(n):
                rec = {"instance": inst.name, "gpu": gpu, "prompt_len": a.prompt_len, "max_tokens": max_tokens,
                       "sent_t": time.time(), "first_token_t": None, "tokens": 0, "done_t": None, "gaps": [],
                       "error": None}
                self.records.append(rec)
                ev = asyncio.Event()
                events.append(ev)
                self.tasks.append(asyncio.create_task(self.stream(inst, self.body(inst, a.prompt_len, max_tokens),
                                                                  rec, ev)))
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in events)), timeout=600)
            await self.swap(inst, "sleep", phase="prime")
            print(f"primed {inst.name}: {n} requests x {a.prompt_len} tokens "
                  f"= {n * a.prompt_len * per_tok / GIB:.2f} GiB KV", flush=True)

    async def rounds(self):
        a = self.args
        deltas = [d for d in a.deltas.split(",") for _ in range(a.reps)]
        random.Random(a.seed).shuffle(deltas)
        cur = {g: None for g in self.gpus}
        k = min(len(p) for p in self.pool.values())
        for r, delta in enumerate(deltas):
            sleeping = [self.swap(cur[g], "sleep", phase="round", round=r, delta=delta)
                        for g in self.gpus if cur[g] is not None]
            await asyncio.gather(*sleeping)
            await asyncio.sleep(a.settle_s)
            nxt = {g: self.pool[g][(r + (gi if a.stagger else 0)) % k] for gi, g in enumerate(self.gpus)}
            t0 = time.time() + 0.05
            if delta == "seq":
                for g in self.gpus:
                    await self.swap(nxt[g], "wake", phase="round", round=r, delta=delta)
            else:
                d = float(delta)

                async def wake_at(gi, g):
                    await asyncio.sleep(max(0.0, t0 + gi * d - time.time()))
                    await self.swap(nxt[g], "wake", phase="round", round=r, delta=delta)

                await asyncio.gather(*(wake_at(gi, g) for gi, g in enumerate(self.gpus)))
            cur = nxt
            print(f"round {r + 1}/{len(deltas)} delta={delta}: woke "
                  + ", ".join(i.name for i in nxt.values()), flush=True)
            await asyncio.sleep(a.dwell_s)

    async def run(self):
        a = self.args
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=3600),
                                             connector=aiohttp.TCPConnector(limit=0, force_close=True))
        try:
            for inst in self.bg.values():
                await self.swap(inst, "wake", phase="bg")
                for _ in range(a.bg_concurrency):
                    self.tasks.append(asyncio.create_task(self.bg_worker(inst)))
            primed = []
            await asyncio.gather(*(self.prime(g, primed) for g in self.gpus))
            with open(os.path.join(self.run_dir, "primed.tsv"), "w") as f:
                f.write("instance\tkv_bytes_per_token\trequests\tkv_gib_expected\n")
                for name, per_tok, n, gib in primed:
                    f.write(f"{name}\t{per_tok}\t{n}\t{gib:.3f}\n")
            with open(os.path.join(self.run_dir, "config.txt"), "a") as cfg:
                cfg.write(f"rounds_start={time.time():.3f}\n")
            await self.rounds()
            with open(os.path.join(self.run_dir, "config.txt"), "a") as cfg:
                cfg.write(f"rounds_end={time.time():.3f}\n")
        finally:
            self.stop.set()
            for t in self.tasks:
                t.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            with open(os.path.join(self.run_dir, "requests.jsonl"), "w") as f:
                for rec in self.records:
                    f.write(json.dumps(rec) + "\n")
            await self.session.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--kv-gib", type=float, default=5.0, help="KV each pool model holds while asleep")
    ap.add_argument("--prompt-len", type=int, default=7168)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--deltas", default="0,0.1,0.3,1,seq",
                    help="wake start offsets between GPUs, seconds; 'seq' = after the previous GPU's wake")
    ap.add_argument("--reps", type=int, default=4, help="rounds per delta")
    ap.add_argument("--dwell-s", type=float, default=3.0, help="time the woken models run before the next round")
    ap.add_argument("--settle-s", type=float, default=1.0, help="gap between the sleeps and the wakes of a round")
    ap.add_argument("--stagger", action="store_true", help="GPU i wakes pool index round+i instead of round")
    ap.add_argument("--bg-concurrency", type=int, default=16)
    ap.add_argument("--bg-prompt-len", type=int, default=256)
    ap.add_argument("--bg-output-len", type=int, default=256)
    ap.add_argument("--gap-s", type=float, default=0.15, help="stream stalls longer than this are recorded")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(Storm(args).run())


if __name__ == "__main__":
    main()
