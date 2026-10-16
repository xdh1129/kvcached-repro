"""Correctness check for the tiered sleep backend: KV survives sleep/wake mid-generation.

Against one running vLLM instance (sleep mode + tiered_restore plugin):
  reference  N greedy requests run to completion with no sleep.
  noise      the same again with no sleep: how far greedy outputs drift between
             two identical runs (batching and chunked-prefill timing in bf16).
  restore    the same N requests; while they are generating, POST
             /sleep?level=1&mode=keep (weights and in-use KV pages offloaded),
             wait, POST /wake_up, let them finish. Outputs must match the reference.
  control    (--control) same as restore but the backend skips the KV load
             (DEBUG_SKIP_KV_LOAD file in TIERED_DIR) and zeroes the KV pages.
             Outputs should break, which shows the check can fail.
The byte-exact check is the backend's own (TIERED_VERIFY=1): per-region checksums
of weights and KV before release and after load, logged in TIERED_LOG.

usage: python check_tiered_restore.py --port P --gpu G --tiered-dir D --out result.json [--control]
"""

import argparse
import json
import os
import subprocess
import threading
import time

import requests

N_REQ = 8
MAX_TOKENS = 300
TOPICS = ["the history of the Roman empire", "how photosynthesis works", "the rules of chess",
          "the design of TCP congestion control", "the life cycle of stars", "baking sourdough bread",
          "the causes of the French revolution", "how vaccines train the immune system"]


# CHECK_PARAGRAPHS: 110 gives ~1.3k prompt tokens (Llama-3 tokenizer); use fewer with a short
# max-model-len (60 fits 2048 with every tokenizer of the 18-model set).
PARAGRAPHS = int(os.environ.get("CHECK_PARAGRAPHS", "110"))


def prompt(i):
    # distinct per request, so the KV spans several kvcached pages
    body = " ".join(f"Paragraph {k}: notes about {TOPICS[i]}, item {k * 7 + i}." for k in range(PARAGRAPHS))
    return f"{body}\nNow write a detailed essay about {TOPICS[i]}.\n"


def gpu_used_mib(gpu):
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    return int(out.strip())


def run_batch(base, model, sleep_at=None, sleep_for=3.0, events=None):
    results = [None] * N_REQ
    t0 = time.time()

    def one(i):
        r = requests.post(f"{base}/v1/completions", json={
            "model": model, "prompt": prompt(i), "max_tokens": MAX_TOKENS, "temperature": 0,
            "ignore_eos": True}, timeout=600)
        r.raise_for_status()
        results[i] = {"text": r.json()["choices"][0]["text"], "done_s": time.time() - t0}

    threads = [threading.Thread(target=one, args=(i,)) for i in range(N_REQ)]
    for th in threads:
        th.start()
    if sleep_at is not None:
        time.sleep(sleep_at)
        events["sleep_start_s"] = time.time() - t0
        requests.post(f"{base}/sleep", params={"level": 1, "mode": "keep"}, timeout=600).raise_for_status()
        events["sleep_end_s"] = time.time() - t0
        events["asleep_gpu_mib"] = gpu_used_mib(events["gpu"])
        time.sleep(sleep_for)
        events["wake_start_s"] = time.time() - t0
        requests.post(f"{base}/wake_up", timeout=600).raise_for_status()
        events["wake_end_s"] = time.time() - t0
    for th in threads:
        th.join()
    return results


def compare(ref, got):
    rows = []
    for r, g in zip(ref, got):
        a, b = r["text"], g["text"]
        common = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        rows.append({"match": a == b, "common_prefix_chars": common, "ref_chars": len(a)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--tiered-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep-at", type=float, default=2.0)
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()
    base = f"http://localhost:{args.port}"
    model = requests.get(f"{base}/v1/models").json()["data"][0]["id"]

    out = {"model": model, "n_req": N_REQ, "max_tokens": MAX_TOKENS,
           "awake_gpu_mib": gpu_used_mib(args.gpu)}
    out["reference"] = run_batch(base, model)
    rows = compare(out["reference"], run_batch(base, model))
    out["noise"] = {"compare": rows, "all_match": all(r["match"] for r in rows)}
    print(f"noise: match {sum(r['match'] for r in rows)}/{N_REQ} between two runs without sleep", flush=True)
    phases = [("restore", False)] + ([("control", True)] if args.control else [])
    for name, skip in phases:
        flag = os.path.join(args.tiered_dir, "DEBUG_SKIP_KV_LOAD")
        if skip:
            open(flag, "w").close()
        ev = {"gpu": args.gpu}
        try:
            got = run_batch(base, model, sleep_at=args.sleep_at, events=ev)
        finally:
            if skip:
                os.remove(flag)
        rows = compare(out["reference"], got)
        ev["requests_in_flight_at_sleep"] = sum(g["done_s"] > ev["sleep_start_s"] for g in got)
        out[name] = {"events": ev, "outputs": got, "compare": rows,
                     "all_match": all(r["match"] for r in rows)}
        print(f"{name}: in flight at sleep {ev['requests_in_flight_at_sleep']}/{N_REQ}, "
              f"match {sum(r['match'] for r in rows)}/{N_REQ}, "
              f"GPU awake {out['awake_gpu_mib']} MiB -> asleep {ev['asleep_gpu_mib']} MiB", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
