"""Replay Prism's 8-model e2e workload against the instances of a run directory.

Same request set as analysis/memory_demand.py (real_trace.pkl, ranks -> model_1..8,
arrival times x time_scale). Each request sends exactly prompt_len prompt tokens
(random token ids) and decodes exactly output_len tokens (ignore_eos), streamed
from /v1/completions on the instance named after its model.

Writes one JSON line per request to <run_dir>/requests.jsonl:
  id, model, port, prompt_len, output_len, scheduled_s, sent_s, first_token_s, done_s
  (seconds since replay start), ttft_s, tpot_s, e2e_s, output_tokens, ok, error

With --proxy URL, requests go to URL/<model>/v1/completions (scripts/swap_proxy.py)
with an X-Request-Id header, so proxy_requests.jsonl joins on id.

usage: python replay_prism_trace.py <run_dir> [--time-scale 0.2] [--replication 1] [--proxy URL]
"""

import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from memory_demand import PRISM, load_trace  # noqa: E402


async def send(session, req, t0, out):
    await asyncio.sleep(max(0.0, t0 + req["scheduled_s"] - time.perf_counter()))
    body = {
        "model": req["repo"],
        "prompt": req["prompt_ids"],
        "max_tokens": req["output_len"],
        "min_tokens": req["output_len"],
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec = {k: req[k] for k in ("id", "model", "port", "prompt_len", "output_len", "scheduled_s")}
    rec.update(sent_s=time.perf_counter() - t0, first_token_s=None, done_s=None,
               output_tokens=0, ok=False, error=None)
    try:
        async with session.post(req["url"], json=body, headers={"X-Request-Id": str(req["id"])}) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
            else:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    if chunk.get("choices") and chunk["choices"][0].get("text") is not None \
                            and rec["first_token_s"] is None:
                        rec["first_token_s"] = time.perf_counter() - t0
                    if chunk.get("usage"):
                        rec["output_tokens"] = chunk["usage"]["completion_tokens"]
                rec["done_s"] = time.perf_counter() - t0
                rec["ok"] = rec["first_token_s"] is not None
    except Exception as e:  # recorded per request; the replay continues
        rec["error"] = f"{type(e).__name__}: {e}"

    if rec["ok"]:
        rec["ttft_s"] = rec["first_token_s"] - rec["sent_s"]
        rec["e2e_s"] = rec["done_s"] - rec["sent_s"]
        n = rec["output_tokens"]
        rec["tpot_s"] = (rec["done_s"] - rec["first_token_s"]) / (n - 1) if n > 1 else None
    out.write(json.dumps(rec) + "\n")
    out.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--trace", default=os.path.join(PRISM, "benchmark/multi-model/real_trace.pkl"))
    ap.add_argument("--time-scale", type=float, default=0.2)
    ap.add_argument("--replication", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy", default=None, help="swap proxy base URL, e.g. http://localhost:30999")
    args = ap.parse_args()

    instances = {}
    with open(os.path.join(args.run_dir, "instances.tsv")) as f:
        for line in f.read().splitlines():
            name, repo, _, port = line.split("\t")
            instances[name] = (repo, int(port))

    rng = np.random.default_rng(args.seed)
    reqs = []
    for model, rows in load_trace(args.trace, args.time_scale).items():
        if model not in instances:
            continue
        repo, port = instances[model]
        url = f"{args.proxy}/{model}/v1/completions" if args.proxy else f"http://localhost:{port}/v1/completions"
        for arrival, prompt_len, output_len in rows:
            for _ in range(args.replication):
                reqs.append({"model": model, "repo": repo, "port": port, "url": url, "scheduled_s": float(arrival),
                             "prompt_len": int(prompt_len), "output_len": max(1, int(output_len)),
                             # token ids valid in every vocabulary used (all >= 32k)
                             "prompt_ids": rng.integers(1000, 30000, int(prompt_len)).tolist()})
    reqs.sort(key=lambda r: r["scheduled_s"])
    for i, r in enumerate(reqs):
        r["id"] = i
    print(f"replaying {len(reqs)} requests over {reqs[-1]['scheduled_s']:.0f}s "
          f"to {len(instances)} instances", flush=True)

    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=0)
    with open(os.path.join(args.run_dir, "requests.jsonl"), "w") as out:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            t0 = time.perf_counter()
            with open(os.path.join(args.run_dir, "config.txt"), "a") as cfg:
                cfg.write(f"workload_start={time.time() - (time.perf_counter() - t0):.3f}\n")
            await asyncio.gather(*(send(session, r, t0, out) for r in reqs))
    print("replay done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
