"""Interleaved multi-turn agents: one conversation per model, requests round-robin across models.

The instances in <run_dir>/instances.tsv are named g<gpu>_a<i>: agent i on that GPU.
Each agent is one conversation with no initial history. Turn j of agent i (request i_j)
sends the whole conversation so far plus a new input (--first-len tokens on turn 1,
--turn-len after), and decodes exactly --out-len tokens. The generated token ids
(return_token_ids) are appended to the history, so with prefix caching the next turn
re-uses the cached history and only prefills the new input.

Per GPU the requests go one at a time in the order 1_1, 2_1, ..., A_1, 1_2, 2_2, ...
through the swap proxy (scripts/swap_proxy.py --keep), which wakes the agent's model
and evicts the least recently used one (weights + cached history KV) when the GPU is
full. With more agents than fit, every request needs a swap. With --sync (default), the
GPUs advance together: step k starts on all GPUs once every GPU finished step k-1, so
their swaps coincide.

Writes <run_dir>/requests.jsonl, one line per request:
  id ("<gpu>:<agent>:<turn>", also sent as X-Request-Id), gpu, agent, turn, step,
  prompt_len, history_len (tokens a full prefix hit would cover), sent_t, first_token_t,
  done_t (unix), output_tokens, hit_tokens (vllm:prefix_cache_hits_total delta of the
  agent's instance), ok, error

usage: python agent_rr.py <run_dir> --proxy URL [--turns 24] [--first-len 2048]
                          [--turn-len 1024] [--out-len 300] [--no-sync] [--seed 0]
"""

import argparse
import asyncio
import json
import os
import re
import time

import aiohttp
import numpy as np


class Agent:
    def __init__(self, name, repo, gpu, port, index):
        self.name, self.repo, self.gpu, self.port, self.index = name, repo, gpu, port, index
        self.history = []  # token ids of the conversation so far


async def prefix_hits(session, agent):
    """vllm:prefix_cache_hits_total of the agent's instance (read directly, not via the proxy)."""
    async with session.get(f"http://localhost:{agent.port}/metrics") as r:
        text = await r.text()
    return sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines()
               if l.startswith("vllm:prefix_cache_hits_total"))


async def turn(session, args, agent, j, step, rng, out):
    new = rng.integers(1000, 30000, args.first_len if j == 1 else args.turn_len).tolist()
    prompt = agent.history + new
    rid = f"{agent.gpu}:{agent.index}:{j}"
    rec = {"id": rid, "gpu": agent.gpu, "agent": agent.index, "turn": j, "step": step,
           "prompt_len": len(prompt), "history_len": len(agent.history), "sent_t": None,
           "first_token_t": None, "done_t": None, "output_tokens": 0, "hit_tokens": None,
           "ok": False, "error": None}
    body = {"model": agent.repo, "prompt": prompt, "max_tokens": args.out_len, "min_tokens": args.out_len,
            "ignore_eos": True, "temperature": 0.0, "stream": True, "return_token_ids": True}
    out_ids = []
    try:
        hits0 = await prefix_hits(session, agent)
        rec["sent_t"] = time.time()
        async with session.post(f"{args.proxy}/{agent.name}/v1/completions", json=body,
                                headers={"X-Request-Id": rid}) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
            else:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    for c in chunk.get("choices") or []:
                        ids = c.get("token_ids") or []
                        if ids and rec["first_token_t"] is None:
                            rec["first_token_t"] = time.time()
                        out_ids += ids
                rec["done_t"] = time.time()
                rec["output_tokens"] = len(out_ids)
                rec["ok"] = len(out_ids) == args.out_len
                if not rec["ok"]:
                    rec["error"] = f"got {len(out_ids)} output tokens"
        rec["hit_tokens"] = int(await prefix_hits(session, agent) - hits0)
    except Exception as e:  # recorded; the conversation goes on without this turn's output
        rec["error"] = f"{type(e).__name__}: {e}"
    if rec["ok"]:
        agent.history = prompt + out_ids
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--proxy", required=True, help="swap proxy base URL, e.g. http://localhost:30999")
    ap.add_argument("--turns", type=int, default=24, help="requests per agent (J)")
    ap.add_argument("--first-len", type=int, default=2048, help="new input tokens on turn 1")
    ap.add_argument("--turn-len", type=int, default=1024, help="new input tokens on later turns")
    ap.add_argument("--out-len", type=int, default=300)
    ap.add_argument("--no-sync", action="store_true", help="let each GPU run its sequence independently")
    ap.add_argument("--request-timeout", type=float, default=600,
                    help="seconds before a request is abandoned and recorded as failed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    agents = {}
    for line in open(os.path.join(args.run_dir, "instances.tsv")).read().splitlines():
        name, repo, gpu, port = line.split("\t")
        m = re.fullmatch(r"g(\d+)_a(\d+)", name)
        if not m:
            raise ValueError(f"unexpected instance name {name}")
        agents.setdefault(int(gpu), []).append(Agent(name, repo, int(gpu), int(port), int(m.group(2))))
    for g in agents:
        agents[g].sort(key=lambda a: a.index)
    gpus = sorted(agents)
    n = min(len(v) for v in agents.values())
    steps = n * args.turns
    rngs = {g: np.random.default_rng([args.seed, g]) for g in gpus}
    print(f"{len(gpus)} GPUs x {n} agents x {args.turns} turns = {steps} requests per GPU"
          f" ({'synchronized' if not args.no_sync else 'independent'})", flush=True)

    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                                    connector=aiohttp.TCPConnector(limit=0, force_close=True))
    with open(os.path.join(args.run_dir, "requests.jsonl"), "w") as out, \
            open(os.path.join(args.run_dir, "config.txt"), "a") as cfg:
        cfg.write(f"workload_start={time.time():.3f}\n")
        cfg.flush()

        def request(g, k):
            return turn(session, args, agents[g][k % n], k // n + 1, k, rngs[g], out)

        if args.no_sync:
            async def run_gpu(g):
                for k in range(steps):
                    await request(g, k)

            await asyncio.gather(*(run_gpu(g) for g in gpus))
        else:
            for k in range(steps):
                recs = await asyncio.gather(*(request(g, k) for g in gpus))
                if (k + 1) % n == 0:
                    print(f"step {k + 1}/{steps}: turn {k // n + 1} done; history "
                          + ", ".join(f"g{r['gpu']}_a{r['agent']}={r['prompt_len'] + r['output_tokens']}" for r in recs)
                          + ("; errors: " + "; ".join(r["error"] for r in recs if r["error"]) if any(r["error"] for r in recs) else ""),
                          flush=True)
        cfg.write(f"workload_end={time.time():.3f}\n")
    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
