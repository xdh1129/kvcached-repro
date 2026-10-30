"""Does an agent's cached history survive sleep(mode=keep) / wake_up as prefix-cache hits?

Against one running instance (prefix caching on, sleep mode on), all greedy:
  cold     history + message, first time: full prefill
  warm     the same prompt again while awake: prefix-cache hit (reference TTFT)
  sleep    /sleep?level=1&mode=keep, then /wake_up
  restored the same prompt again: with the history's KV restored it should hit the
           cache like "warm" and return exactly the "cold" output
Per step: TTFT, prefix-cache hit tokens (vllm:prefix_cache_hits_total delta) and the
output text. Prints a JSON summary and exits 1 if "restored" differs from "cold".

usage: python check_prefix_restore.py --port 31901 --model meta-llama/Llama-3.2-1B
                                      [--history 6000] [--message 64] [--max-tokens 32]
"""

import argparse
import json
import sys
import time
import urllib.request

import numpy as np


def post(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def metric(base, name):
    text = urllib.request.urlopen(f"{base}/metrics", timeout=10).read().decode()
    return sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines() if l.startswith(name + "{") or l.startswith(name + " "))


def complete(base, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "ignore_eos": True, "logprobs": 1}
    req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0, ttft, text, logprobs = time.time(), None, [], []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("choices"):
                if ttft is None:
                    ttft = time.time() - t0
                c = chunk["choices"][0]
                text.append(c.get("text") or "")
                logprobs += (c.get("logprobs") or {}).get("token_logprobs") or []
    return ttft, "".join(text), logprobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--history", type=int, default=6000)
    ap.add_argument("--message", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base = f"http://localhost:{args.port}"
    rng = np.random.default_rng(args.seed)
    prompt = rng.integers(1000, 30000, args.history + args.message).tolist()

    out = {}

    def step(name):
        hits0 = metric(base, "vllm:prefix_cache_hits_total")
        ttft, text, logprobs = complete(base, args.model, prompt, args.max_tokens)
        out[name] = {"ttft_s": round(ttft, 4), "hit_tokens": int(metric(base, "vllm:prefix_cache_hits_total") - hits0),
                     "text": text, "logprobs": logprobs}

    step("cold")
    step("warm")
    t = time.time()
    post(f"{base}/sleep?level=1&mode=keep")
    out["sleep_s"] = round(time.time() - t, 4)
    t = time.time()
    post(f"{base}/wake_up")
    out["wake_s"] = round(time.time() - t, 4)
    step("restored")
    # Random-token prompts make text outputs repetitive, so a wrong KV could still give the
    # same text; per-token logprobs expose it. warm vs cold shows the numerical noise floor.
    lp = {k: np.array(out[k].pop("logprobs"), dtype=float) for k in ("cold", "warm", "restored")}
    out["warm_vs_cold_max_logprob_diff"] = float(np.max(np.abs(lp["warm"] - lp["cold"])))
    out["restored_vs_cold_max_logprob_diff"] = float(np.max(np.abs(lp["restored"] - lp["cold"])))
    out["restored_vs_warm_max_logprob_diff"] = float(np.max(np.abs(lp["restored"] - lp["warm"])))
    out["restored_matches_cold"] = out["restored"]["text"] == out["cold"]["text"]
    out["warm_matches_cold"] = out["warm"]["text"] == out["cold"]["text"]
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["restored_matches_cold"] else 1)


if __name__ == "__main__":
    main()
