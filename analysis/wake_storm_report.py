"""Report of a wake-storm run (scripts/run_wake_storm.sh).

Joins each wake in swaps.jsonl with the tiered_restore resume record of the same
instance (tiered_<instance>.jsonl; absent for PATHS=vllm) and with pcie.csv, then
summarises per delta (wake start offset between GPUs):
  wake       HTTP /wake_up time; weights / KV bytes and GiB/s per path (plugin timing)
  overlap    share of a wake's time during which another GPU was also waking
  round      first wake start -> last wake end over all GPUs, and the bytes all GPUs
             moved in it per second (aggregate bandwidth)
  resume     wake start -> next token of the woken model's frozen requests (includes
             re-prefill under PATHS=vllm)
  pcie       mean PCIe RX per GPU while its wake runs (NVML)
  sleep      /sleep time and save bandwidth per path
  bg         background TPOT with and without a wake on its GPU
Writes <run_dir>/summary.md and summary.json.

usage: python analysis/wake_storm_report.py <run_dir>
"""

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

GIB = 1 << 30


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def p50(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def delta_key(d):
    return (1, 0.0) if d == "seq" else (0, float(d))


def main(run_dir):
    cfg = dict(l.strip().split("=", 1) for l in open(os.path.join(run_dir, "config.txt")) if "=" in l)
    swaps = load_jsonl(os.path.join(run_dir, "swaps.jsonl"))
    tiered = {}
    for f in glob.glob(os.path.join(run_dir, "tiered_*.jsonl")):
        tiered[os.path.basename(f)[len("tiered_"):-len(".jsonl")]] = load_jsonl(f)
    reqs = load_jsonl(os.path.join(run_dir, "requests.jsonl"))
    bg = load_jsonl(os.path.join(run_dir, "bg_requests.jsonl"))
    pcie = []
    if os.path.exists(os.path.join(run_dir, "pcie.csv")):
        with open(os.path.join(run_dir, "pcie.csv")) as f:
            pcie = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

    def resume_rec(sw, event):
        """The plugin record of this swap: same instance, started inside the HTTP call."""
        t0, t1 = sw["t_unix"], sw["t_unix"] + sw["duration_s"]
        for r in tiered.get(sw["instance"], []):
            if r["event"] == event and t0 - 0.05 <= r["t_unix"] <= t1:
                return r
        return None

    wakes = [s for s in swaps if s["action"] == "wake" and s.get("phase") == "round"]
    sleeps = [s for s in swaps if s["action"] == "sleep" and s.get("phase") == "round"]
    for w in wakes:
        w["t1"] = w["t_unix"] + w["duration_s"]
        r = resume_rec(w, "resume")
        w["weights_gib"] = w["kv_gib"] = w["weights_gibs"] = w["kv_gibs"] = None
        if r:
            w["weights_gib"] = r["weights_bytes"] / GIB
            w["kv_gib"] = r["kv_bytes"] / GIB
            wd = r["weights_end_s"] - r["weights_start_s"]
            kd = r["kv_end_s"] - r["kv_start_s"]
            w["weights_gibs"] = w["weights_gib"] / wd if wd > 0 else None
            w["kv_gibs"] = w["kv_gib"] / kd if kd > 0 and r["kv_bytes"] else None
        others = [o for o in wakes if o["round"] == w["round"] and o["gpu"] != w["gpu"]]
        o = sum(overlap(w["t_unix"], w["t1"], x["t_unix"], x["t_unix"] + x["duration_s"]) for x in others)
        w["overlap_frac"] = min(1.0, o / w["duration_s"]) if w["duration_s"] > 0 else 0.0
        # frozen requests of this instance: stall spanning the wake start, ending after it
        ends = [g[1] for q in reqs if q["instance"] == w["instance"] for g in q["gaps"]
                if g[0] <= w["t_unix"] < g[1]]
        w["resume_s"] = p50([e - w["t_unix"] for e in ends]) if ends else None
        col = f"gpu{w['gpu']}_rx_mbs"
        rx = [p[col] for p in pcie if w["t_unix"] <= p["t"] <= w["t1"] and col in p]
        w["rx_gbs"] = float(np.mean(rx)) / 1000 if rx else None

    gpus = sorted({w["gpu"] for w in wakes})
    deltas = sorted({w["delta"] for w in wakes}, key=delta_key)
    out = {"run": os.path.basename(run_dir), "config": cfg, "per_delta": {}}
    lines = [f"# {os.path.basename(run_dir)}", "",
             f"paths `{cfg.get('paths')}` · GPUs {cfg.get('gpus')} · KV {cfg.get('kv_gib')} GiB per model · "
             f"pool {cfg.get('pool')} · background {cfg.get('bg_model') or 'none'} · "
             f"dwell {cfg.get('dwell_s')} s · {cfg.get('reps')} rounds per delta", ""]

    lines += ["## Wakes per delta and GPU (medians)", "",
              "| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(x, nd=2):
        return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

    for d in deltas:
        out["per_delta"][d] = {"gpus": {}}
        for g in gpus:
            ws = [w for w in wakes if w["delta"] == d and w["gpu"] == g]
            col = lambda k: [w[k] for w in ws if w[k] is not None]  # noqa: E731
            row = {"n": len(ws), "wake_s": p50(col("duration_s")), "overlap": p50(col("overlap_frac")),
                   "weights_gib": p50(col("weights_gib")), "weights_gibs": p50(col("weights_gibs")),
                   "kv_gib": p50(col("kv_gib")), "kv_gibs": p50(col("kv_gibs")),
                   "resume_s": p50(col("resume_s")), "rx_gbs": p50(col("rx_gbs"))}
            out["per_delta"][d]["gpus"][g] = row
            lines.append(f"| {d} | {g} | {row['n']} | {fmt(row['wake_s'], 3)} | {row['overlap']:.0%} | "
                         f"{fmt(row['weights_gib'])} | {fmt(row['weights_gibs'], 1)} | {fmt(row['kv_gib'])} | "
                         f"{fmt(row['kv_gibs'], 1)} | {fmt(row['resume_s'], 3)} | {fmt(row['rx_gbs'], 1)} |")

    lines += ["", "## Rounds: all GPUs together (medians)", "",
              "| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |",
              "|---|---|---|---|---|---|"]
    by_round = defaultdict(list)
    for w in wakes:
        by_round[w["round"]].append(w)
    for d in deltas:
        spans, gibs, aggs, peaks = [], [], [], []
        for ws in by_round.values():
            if ws[0]["delta"] != d:
                continue
            t0, t1 = min(w["t_unix"] for w in ws), max(w["t1"] for w in ws)
            moved = sum((w["weights_gib"] or 0) + (w["kv_gib"] or 0) for w in ws)
            spans.append(t1 - t0)
            gibs.append(moved)
            aggs.append(moved / (t1 - t0))
            rx = [sum(p.get(f"gpu{g}_rx_mbs", 0) for g in gpus) for p in pcie if t0 <= p["t"] <= t1]
            if rx:
                peaks.append(max(rx) / 1000)
        out["per_delta"][d].update(span_s=p50(spans), moved_gib=p50(gibs), aggregate_gibs=p50(aggs),
                                   peak_rx_gbs=p50(peaks))
        lines.append(f"| {d} | {len(spans)} | {fmt(p50(spans), 3)} | {fmt(p50(gibs))} | {fmt(p50(aggs), 1)} | "
                     f"{fmt(p50(peaks), 1)} |")

    lines += ["", "## Sleeps (all rounds, medians)", "",
              "| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |", "|---|---|---|---|---|"]
    for g in gpus:
        ss = [s for s in sleeps if s["gpu"] == g]
        wr, kr = [], []
        for s in ss:
            r = resume_rec(s, "suspend")
            if r and r.get("weights_save_s"):
                wr.append(r["weights_bytes"] / GIB / r["weights_save_s"])
            if r and r.get("kv_save_s") and r.get("kv_bytes"):
                kr.append(r["kv_bytes"] / GIB / r["kv_save_s"])
        lines.append(f"| {g} | {len(ss)} | {fmt(p50([s['duration_s'] for s in ss]), 3)} | {fmt(p50(wr), 1)} | "
                     f"{fmt(p50(kr), 1)} |")

    if bg:
        windows = defaultdict(list)
        for w in wakes:
            windows[w["gpu"]].append((w["t_unix"], w["t1"]))
        lines += ["", "## Background decode (TPOT ms, p50)", "", "| GPU | no wake on GPU | during a wake |", "|---|---|---|"]
        for g in gpus:
            quiet, busy = [], []
            for b in bg:
                if b["gpu"] != g or not b.get("tpot_s"):
                    continue
                hit = any(overlap(b["first_token_t"], b["done_t"], a, z) > 0 for a, z in windows[g])
                (busy if hit else quiet).append(b["tpot_s"] * 1000)
            lines.append(f"| {g} | {fmt(p50(quiet), 1)} (n={len(quiet)}) | {fmt(p50(busy), 1)} (n={len(busy)}) |")

    lines += ["", "overlap: share of the wake during which another GPU was also waking. "
              "GiB/s per path from the plugin's per-path start/end. resume: wake start to the first "
              "token of the frozen requests. PCIe RX from NVML (~20 ms windows)."]
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({**out, "wakes": wakes}, f, indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
