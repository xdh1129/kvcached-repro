"""Swap and restore summary of a Prism trace run with tiered swapping (PATHS=...).

Reads <run_dir>/swaps.jsonl (proxy) and <run_dir>/tiered_<instance>.jsonl
(backend). Only events after the trace starts count: the sleeps right after
startup and their one-time weight saves are excluded.

Reports, per GPU and in total:
  wakes, sleeps, and keep-sleeps with requests in flight ("busy": agent restore needed)
  per restore: weights/state and KV bytes, time of each path, restore time
  traffic through host DRAM: GPU->host on sleep, host->GPU on wake

usage: python analysis/tiered_run_summary.py <run_dir>    (writes <run_dir>/tiered_summary.md)
"""

import glob
import json
import os
import sys

import numpy as np

GB = 1e9


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def main():
    run_dir = sys.argv[1]
    swaps = [json.loads(l) for l in open(os.path.join(run_dir, "swaps.jsonl"))]
    t_start = min((s["t_unix"] for s in swaps), default=0)  # first swap is the first request's wake
    tiered = {}
    for f in glob.glob(os.path.join(run_dir, "tiered_*.jsonl")):
        name = os.path.basename(f)[len("tiered_"):-len(".jsonl")]
        tiered[name] = [r for r in map(json.loads, open(f)) if r["t_unix"] >= t_start - 1]

    lines = [f"# Tiered swap summary: {os.path.basename(os.path.normpath(run_dir))}", ""]
    cfg = dict(l.strip().split("=", 1) for l in open(os.path.join(run_dir, "config.txt")) if "=" in l)
    lines.append(f"paths `{cfg.get('swap', '?')}` · time scale {cfg.get('time_scale')} · "
                 f"replication {cfg.get('replication')} · min resident {cfg.get('min_resident_s', '-')} s")
    lines += ["", "## Swaps", "",
              "| GPU | wakes | sleeps | busy sleeps (frozen requests) | frozen requests total |",
              "|---|---|---|---|---|"]
    for gpu in sorted({s["gpu"] for s in swaps}) + ["all"]:
        ss = [s for s in swaps if gpu == "all" or s["gpu"] == gpu]
        sleeps = [s for s in ss if s["action"] == "sleep"]
        busy = [s for s in sleeps if s.get("inflight", 0) > 0]
        lines.append(f"| {gpu} | {sum(s['action'] == 'wake' for s in ss)} | {len(sleeps)} | {len(busy)} "
                     f"| {sum(s['inflight'] for s in busy)} |")

    resumes = [r for rs in tiered.values() for r in rs if r["event"] == "resume"]
    suspends = [r for rs in tiered.values() for r in rs if r["event"] == "suspend"]
    with_kv = [r for r in resumes if r.get("kv_bytes", 0) > 0.2 * GB]  # more than the always-mapped null page
    lines += ["", "## Restores", "",
              f"{len(resumes)} restores, {len(with_kv)} of them with > 0.2 GB of KV.", "",
              "| | n | weights+state GB p50 | KV GB p50 / max | weights path s p50 | KV path s p50 / max "
              "| restore s p50 / p95 / max |",
              "|---|---|---|---|---|---|---|"]
    for label, rs in (("all restores", resumes), ("restores with KV", with_kv)):
        if not rs:
            continue
        w = [r["weights_bytes"] / GB for r in rs]
        kv = [r.get("kv_bytes", 0) / GB for r in rs]
        wt = [r["weights_end_s"] - r["weights_start_s"] for r in rs]
        kt = [r["kv_end_s"] - r["kv_start_s"] for r in rs if "kv_end_s" in r]
        tot = [max(r["weights_end_s"], r.get("kv_end_s", 0)) for r in rs]
        lines.append(f"| {label} | {len(rs)} | {pct(w, 50):.2f} | {pct(kv, 50):.2f} / {max(kv):.2f} "
                     f"| {pct(wt, 50):.3f} | {pct(kt, 50):.3f} / {max(kt, default=float('nan')):.3f} "
                     f"| {pct(tot, 50):.3f} / {pct(tot, 95):.3f} / {max(tot):.3f} |")

    to_host = sum(r["state_save_host_bytes"] + r["kv_save_host_bytes"] for r in suspends)
    from_host = sum(r["weights_host_bytes"] + r.get("kv_host_bytes", 0) for r in resumes)
    kv_out = sum(r["kv_bytes"] for r in suspends)
    lines += ["", "## Data moved during the trace", "",
              "| | GB |", "|---|---|",
              f"| KV offloaded on sleep (any path) | {kv_out / GB:.2f} |",
              f"| through host DRAM, GPU -> host (sleep) | {to_host / GB:.2f} |",
              f"| through host DRAM, host -> GPU (wake) | {from_host / GB:.2f} |"]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(run_dir, "tiered_summary.md"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
