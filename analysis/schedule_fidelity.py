"""How faithfully each replay run followed the reference sleep/wake schedule.

Per config (results/<prefix><config>/), against the reference schedule:
  swaps        sleeps and wakes executed vs. scheduled (by instance and action, on every GPU)
  lag          how late each replayed swap ran (lag_s; > 0 when a slower path falls behind)
  frozen       requests in flight at the busy sleeps, and total KV offloaded

usage: python analysis/schedule_fidelity.py [--prefix prism_8m2g_rep2_sched_]
                                             [--ref results/schedules/ref_dram+ssd]
Writes results/comparison/<prefix>fidelity.md
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIGS = ["dram+recompute", "dram+dram", "ssd+ssd", "gds+gds", "dram+ssd", "dram+gds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="prism_8m2g_rep2_sched_")
    ap.add_argument("--ref", default=os.path.join(ROOT, "results", "schedules", "ref_dram+ssd"))
    args = ap.parse_args()
    ref = [json.loads(l) for l in open(os.path.join(args.ref, "swaps.jsonl"))]
    ref_count = collections.Counter((s["instance"], s["action"]) for s in ref)
    lines = [f"# Schedule replay fidelity ({args.prefix}*)", "",
             f"Reference: `{os.path.relpath(args.ref, ROOT)}`, {len(ref)} swaps "
             f"({sum(s['action'] == 'sleep' for s in ref)} sleeps, {sum(s['action'] == 'wake' for s in ref)} wakes).",
             "",
             "| config | swaps run / scheduled | extra (policy fallback) | missing | lag s p50 / max "
             "| busy sleeps | frozen requests | KV offloaded GB |",
             "|---|---|---|---|---|---|---|---|"]
    for c in CONFIGS:
        d = os.path.join(ROOT, "results", args.prefix + c)
        if not os.path.exists(os.path.join(d, "swaps.jsonl")):
            continue
        sw = [json.loads(l) for l in open(os.path.join(d, "swaps.jsonl"))]
        got = collections.Counter((s["instance"], s["action"]) for s in sw if s["reason"] == "schedule")
        extra = sum(1 for s in sw if s["reason"] != "schedule")
        missing = sum((ref_count - got).values())
        lag = [s["lag_s"] for s in sw if "lag_s" in s]
        busy = [s for s in sw if s["action"] == "sleep" and s.get("inflight", 0) > 0]
        kv = 0.0
        for f in glob.glob(os.path.join(d, "tiered_*.jsonl")):
            t0 = min(s["t_unix"] for s in sw)
            kv += sum(r["kv_bytes"] for r in map(json.loads, open(f))
                      if r["event"] == "suspend" and r["t_unix"] >= t0 - 1)
        lines.append(f"| {c} | {sum(got.values())} / {len(ref)} | {extra} | {missing} "
                     f"| {np.median(lag):.2f} / {max(lag):.2f} | {len(busy)} "
                     f"| {sum(s['inflight'] for s in busy)} | {kv / 1e9:.1f} |")
    text = "\n".join(lines) + "\n"
    out = os.path.join(ROOT, "results", "comparison", f"{args.prefix}fidelity.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
