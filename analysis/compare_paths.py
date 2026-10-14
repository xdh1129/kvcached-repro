"""Compare the offload-path configs of the Prism trace (one run directory per config).

Default runs: results/prism_8m2g_rep2_kvp_<config> for the six configs
  dram+recompute  weights -> DRAM, KV dropped and recomputed (Prism's RECOMPUTE)
  dram+dram       weights + KV -> DRAM
  ssd+ssd         weights + KV -> SSD (POSIX O_DIRECT through a pinned host buffer)
  gds+gds         weights + KV -> SSD via GDS (cuFile, no host buffer)
  dram+ssd        weights -> DRAM, KV -> SSD, restored in parallel
  dram+gds        weights -> DRAM, KV -> SSD via GDS, restored in parallel

Per config: request latency (all models, and the pressured GPU), vLLM preemptions,
swaps, restore times, and bytes moved through host DRAM and through the SSD
during the trace (startup saves excluded).

usage: python analysis/compare_paths.py [--prefix prism_8m2g_rep2_kvp_] [--out results/comparison/paths_rep2]
Writes <out>.md and <out>.json.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIGS = ["dram+recompute", "dram+dram", "ssd+ssd", "gds+gds", "dram+ssd", "dram+gds"]
GB = 1e9
STORAGE = {"ssd", "gds"}


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def load_run(run_dir):
    inst_gpu = {}
    with open(os.path.join(run_dir, "instances.tsv")) as f:
        for name, _, gpu, _ in csv.reader(f, delimiter="\t"):
            inst_gpu[name] = int(gpu)
    reqs = [json.loads(l) for l in open(os.path.join(run_dir, "requests.jsonl"))]
    swaps = [json.loads(l) for l in open(os.path.join(run_dir, "swaps.jsonl"))]
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    t_start = min((s["t_unix"] for s in swaps), default=0)
    tiered = []
    for f in glob.glob(os.path.join(run_dir, "tiered_*.jsonl")):
        tiered += [r for r in map(json.loads, open(f)) if r["t_unix"] >= t_start - 1]
    return inst_gpu, reqs, swaps, summary, tiered


def latency(reqs):
    ok = [r for r in reqs if r["ok"]]
    return {"n": len(reqs), "failed": len(reqs) - len(ok),
            "ttft_p50": pct([r["ttft_s"] for r in ok], 50), "ttft_p95": pct([r["ttft_s"] for r in ok], 95),
            "ttft_p99": pct([r["ttft_s"] for r in ok], 99),
            "tpot_p50_ms": 1e3 * pct([r["tpot_s"] for r in ok if r.get("tpot_s")], 50),
            "e2e_p50": pct([r["e2e_s"] for r in ok], 50), "e2e_p99": pct([r["e2e_s"] for r in ok], 99)}


def analyze(run_dir, hot_gpu):
    inst_gpu, reqs, swaps, summary, tiered = load_run(run_dir)
    hot = {n for n, g in inst_gpu.items() if g == hot_gpu}
    res = [r for r in tiered if r["event"] == "resume"]
    sus = [r for r in tiered if r["event"] == "suspend"]
    restore_s = [max(r["weights_end_s"], r.get("kv_end_s", 0)) for r in res]
    with_kv = [r for r in res if r.get("kv_bytes", 0) > 0.2 * GB]
    storage = (sum(r["weights_bytes"] for r in res if r["weights_path"] in STORAGE)
               + sum(r.get("kv_bytes", 0) for r in res if r["kv_path"] in STORAGE)
               + sum(r["state_bytes"] for r in sus if r["weights_path"] in STORAGE)
               + sum(r["kv_bytes"] for r in sus if r["kv_path"] in STORAGE))
    hot_swaps = [s for s in swaps if s["gpu"] == hot_gpu]
    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "all": latency(reqs),
        "hot_gpu": latency([r for r in reqs if r["model"] in hot]),
        "preemptions_hot": sum(summary["instances"][n].get("preemptions") or 0 for n in hot),
        "peak_used_gib_hot": summary["gpus"][str(hot_gpu)]["peak_used_gib"],
        "wakes": sum(s["action"] == "wake" for s in swaps),
        "sleeps": sum(s["action"] == "sleep" for s in swaps),
        "busy_sleeps": sum(s["action"] == "sleep" and s.get("inflight", 0) > 0 for s in swaps),
        "frozen_requests": sum(s.get("inflight", 0) for s in swaps if s["action"] == "sleep"),
        "swap_time_hot_s": sum(s["duration_s"] for s in hot_swaps),
        "sleep_s_p50": pct([s["duration_s"] for s in hot_swaps if s["action"] == "sleep"], 50),
        "restores": len(res), "restores_with_kv": len(with_kv),
        "kv_gb_p50_with_kv": pct([r["kv_bytes"] / GB for r in with_kv], 50),
        "restore_s_p50": pct(restore_s, 50), "restore_s_p95": pct(restore_s, 95),
        "restore_s_max": max(restore_s, default=float("nan")),
        "restore_with_kv_s_p50": pct([max(r["weights_end_s"], r.get("kv_end_s", 0)) for r in with_kv], 50),
        "weights_path_s_p50": pct([r["weights_end_s"] - r["weights_start_s"] for r in res], 50),
        "kv_path_s_p50_with_kv": pct([r["kv_end_s"] - r["kv_start_s"] for r in with_kv], 50),
        "kv_offloaded_gb": sum(r["kv_bytes"] for r in sus) / GB,
        "dram_gpu_to_host_gb": sum(r["state_save_host_bytes"] + r["kv_save_host_bytes"] for r in sus) / GB,
        "dram_host_to_gpu_gb": sum(r["weights_host_bytes"] + r.get("kv_host_bytes", 0) for r in res) / GB,
        "storage_io_gb": storage / GB,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="prism_8m2g_rep2_kvp_")
    ap.add_argument("--hot-gpu", type=int, default=1, help="our GPU id of Prism's GPU0 (the pressured one)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "comparison", "paths_rep2"))
    args = ap.parse_args()

    rows = {}
    for c in CONFIGS:
        d = os.path.join(ROOT, "results", args.prefix + c)
        if os.path.exists(os.path.join(d, "summary.json")):
            rows[c] = analyze(d, args.hot_gpu)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(rows, f, indent=1)

    f2 = lambda x: f"{x:.2f}"
    f3 = lambda x: f"{x:.3f}"
    lines = [f"# Offload paths on the Prism trace ({args.prefix}*)", "",
             "## Requests", "",
             "| config | ok / total | TTFT p50 / p95 / p99 s (GPU %d) | E2E p50 / p99 s (GPU %d) | TPOT p50 ms (GPU %d) "
             "| vLLM preemptions (GPU %d) | peak GiB (GPU %d) |" % ((args.hot_gpu,) * 5),
             "|---|---|---|---|---|---|---|"]
    for c, r in rows.items():
        a, h = r["all"], r["hot_gpu"]
        lines.append(f"| {c} | {a['n'] - a['failed']} / {a['n']} | {f2(h['ttft_p50'])} / {f2(h['ttft_p95'])} / "
                     f"{f2(h['ttft_p99'])} | {f2(h['e2e_p50'])} / {f2(h['e2e_p99'])} | {h['tpot_p50_ms']:.1f} "
                     f"| {r['preemptions_hot']} | {r['peak_used_gib_hot']} |")
    lines += ["", "## Swaps and restores", "",
              "| config | wakes | sleeps (busy) | frozen requests | restores with KV | KV GB p50 | restore s p50 / p95 / max "
              "| restore with KV s p50 | weights path s p50 | KV path s p50 | swap time s (GPU %d) |" % args.hot_gpu,
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in rows.items():
        lines.append(f"| {c} | {r['wakes']} | {r['sleeps']} ({r['busy_sleeps']}) | {r['frozen_requests']} "
                     f"| {r['restores_with_kv']} / {r['restores']} | {f2(r['kv_gb_p50_with_kv'])} "
                     f"| {f3(r['restore_s_p50'])} / {f3(r['restore_s_p95'])} / {f3(r['restore_s_max'])} "
                     f"| {f3(r['restore_with_kv_s_p50'])} | {f3(r['weights_path_s_p50'])} "
                     f"| {f3(r['kv_path_s_p50_with_kv'])} | {r['swap_time_hot_s']:.1f} |")
    lines += ["", "## Data moved during the trace (GB)", "",
              "| config | KV offloaded | DRAM: GPU -> host | DRAM: host -> GPU | SSD I/O |",
              "|---|---|---|---|---|"]
    for c, r in rows.items():
        lines.append(f"| {c} | {f2(r['kv_offloaded_gb'])} | {f2(r['dram_gpu_to_host_gb'])} "
                     f"| {f2(r['dram_host_to_gpu_gb'])} | {f2(r['storage_io_gb'])} |")
    text = "\n".join(lines) + "\n"
    with open(args.out + ".md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
