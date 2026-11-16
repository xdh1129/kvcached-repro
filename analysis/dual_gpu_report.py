"""Compare offload paths of the two-GPU closed-loop runs (scripts/run_dual_gpu_suite.sh).

Both GPUs (6, 7) run the same three models; step k starts on both once both finished k-1.
Per run, from tiered_g<gpu>_a<i>.jsonl, requests.jsonl, proxy_requests.jsonl, pcie.csv,
pcie_nvml.csv:
  - whole run per GPU: sum of sleep / wake, pin time, workload time
  - per model: median wake and sleep (excl. pin) by turn range, both GPUs pooled
  - wake overlap: for each pair of same-step wakes on the two GPUs, how much of the
    shorter wake ran while the other GPU was waking too
  - SSD read rate of each wake (SSD job bytes / its data-movement time): GDS staged or Phoenix
  - DRAM: tiered's logical bytes held in DRAM (weights, state, KV), summed over all
    instances, from the "dram" events; peak and time-weighted mean while the workload ran.
    The step function is written to <run_dir>/dram_logical.csv.
  - monitor: peak 10 ms NVMe read (sum of members) and GPU RX (NVML ~24 ms samples)
Sleeps: suspend n >= 1 (n = 0 is the sleep at startup). The k-th resume is the wake for turn k.

usage: python dual_gpu_report.py <out.md> <run_dir> [<run_dir> ...]
"""

import csv
import json
import os
import statistics as st
import sys

GIB = 1 << 30
RANGES = [(2, 5), (6, 10), (11, 15)]
MODELS = {1: "Qwen3-32B", 2: "Qwen3-14B", 3: "Phi-3.5"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def pin_s(r):
    return sum(v for k, v in r.items() if k.endswith("_save_alloc_s"))


def med(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")


def ssd_job(r):
    """name of the job that read/wrote the SSD in a record, or None"""
    if r.get("split_frac"):
        return "ssd"
    return "kv" if r.get("kv_path") in ("gds", "phoenix", "ssd") else None


def wake_ssd_rate(r):
    j = ssd_job(r)
    if not j or f"{j}_bytes" not in r:
        return None
    a = r.get(f"{j}_data_start_s", r.get(f"{j}_start_s"))
    b = r.get(f"{j}_data_end_s", r.get(f"{j}_end_s"))
    n = r[f"{j}_bytes"]
    return (n / GIB / (b - a), n / GIB) if b and b > a and n else None


def analyze(d):
    cfg = dict(l.split("=", 1) for l in open(f"{d}/config.txt").read().splitlines() if "=" in l)
    gpus = [int(g) for g in cfg["gpus"].split()]
    reqs = load(f"{d}/requests.jsonl")
    t0, t1 = float(cfg["workload_start"]), float(cfg["workload_end"])
    out = {"name": os.path.basename(d.rstrip("/")), "paths": cfg["paths"], "gpus": gpus,
           "workload_s": t1 - t0, "ok": sum(r["ok"] for r in reqs), "n": len(reqs), "inst": {}}
    dram_ev = []
    for g in gpus:
        for a in MODELS:
            recs = load(f"{d}/tiered_g{g}_a{a}.jsonl")
            out["inst"][(g, a)] = {"sus": [r for r in recs if r["event"] == "suspend" and r["n"] >= 1],
                                   "res": [r for r in recs if r["event"] == "resume"]}
            dram_ev += [((g, a), r) for r in recs if r["event"] == "dram"]
    # per GPU sums
    out["per_gpu"] = {}
    for g in gpus:
        sus = [r for (gg, _), v in out["inst"].items() if gg == g for r in v["sus"]]
        res = [r for (gg, _), v in out["inst"].items() if gg == g for r in v["res"]]
        out["per_gpu"][g] = {"sleep": sum(r["total_s"] for r in sus), "pin": sum(map(pin_s, sus)),
                             "wake": sum(r["total_s"] for r in res)}
    # wake overlap between the two GPUs, pairing the k-th wake of the same agent
    ov = []
    if len(gpus) == 2:
        ga, gb = gpus
        for a in MODELS:
            for x, y in zip(out["inst"][(ga, a)]["res"], out["inst"][(gb, a)]["res"]):
                xa, xb = x["t_unix"], x["t_unix"] + x["total_s"]
                ya, yb = y["t_unix"], y["t_unix"] + y["total_s"]
                inter = max(0.0, min(xb, yb) - max(xa, ya))
                ov.append((inter / min(xb - xa, yb - ya), abs(xa - ya)))
    out["overlap"] = ov
    # DRAM logical bytes: step function over all instances
    state, rows = {}, []
    for inst, r in sorted(dram_ev, key=lambda e: e[1]["t_unix"]):
        state[inst] = {k: r.get(f"{k}_bytes", 0) for k in ("weights", "state", "kv")}
        rows.append((r["t_unix"], *(sum(s[k] for s in state.values()) for k in ("weights", "state", "kv"))))
    with open(f"{d}/dram_logical.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "weights_gib", "state_gib", "kv_gib", "total_gib"])
        for t, *v in rows:
            w.writerow([round(t, 4)] + [round(x / GIB, 3) for x in v] + [round(sum(v) / GIB, 3)])
    # the value in force at t0 (held since an earlier event), then every change up to t1
    before = [(t0, sum(v), v) for t, *v in rows if t <= t0][-1:]
    inside = before + [(t, sum(v), v) for t, *v in rows if t0 < t <= t1]
    out["dram_peak"] = max(inside, key=lambda x: x[1]) if inside else (0, 0, [0, 0, 0])
    tw = sum(a[1] * (b[0] - a[0]) for a, b in zip(inside, inside[1:] + [(t1, 0, None)]))
    out["dram_mean"] = tw / (t1 - t0) if inside else 0.0
    # monitor
    out["mon"] = {}
    if os.path.exists(f"{d}/pcie.csv"):
        P = list(csv.DictReader(open(f"{d}/pcie.csv")))
        dt = [float(b["t"]) - float(a["t"]) for a, b in zip(P, P[1:])]
        cols = [c for c in P[0] if c.endswith("n1_read_mbs")] if P else []
        out["mon"]["dt_ms"] = med(dt) * 1e3 if dt else float("nan")
        out["mon"]["nvme_read_peak"] = max((sum(float(r[c]) for c in cols) for r in P), default=0) / 1e3 * 1e9 / GIB
    if os.path.exists(f"{d}/pcie_nvml.csv"):
        N = list(csv.DictReader(open(f"{d}/pcie_nvml.csv")))
        for g in gpus:
            rx = [float(r["mbs"]) for r in N if int(r["gpu"]) == g and r["dir"] == "rx"]
            out["mon"][f"gpu{g}_rx_peak"] = max(rx, default=0) * 1e6 / GIB
    return out


def main(out_path, dirs):
    runs = [analyze(d) for d in dirs]
    L = ["## Whole run", "",
         "| path | requests ok | workload s | per GPU: sleep excl. pin / pin / wake (s) | sleep+wake excl. pin, both GPUs |",
         "|---|---|---|---|---|"]
    for r in runs:
        cells = "; ".join(f"g{g} {v['sleep'] - v['pin']:.1f} / {v['pin']:.1f} / {v['wake']:.1f}" for g, v in r["per_gpu"].items())
        tot = sum(v["sleep"] - v["pin"] + v["wake"] for v in r["per_gpu"].values())
        L.append(f"| {r['paths']} | {r['ok']}/{r['n']} | {r['workload_s']:.0f} | {cells} | {tot:.1f} |")
    L += ["", "## Wake overlap between the GPUs", "",
          "Pairs = same agent's k-th wake on both GPUs. overlap = shared time / the shorter wake.", "",
          "| path | pairs | median overlap | pairs with overlap >= 50% | median start offset ms |", "|---|---|---|---|---|"]
    for r in runs:
        ov = r["overlap"]
        L.append(f"| {r['paths']} | {len(ov)} | {med([o for o, _ in ov]):.0%} | "
                 f"{sum(o >= 0.5 for o, _ in ov)}/{len(ov)} | {med([s for _, s in ov]) * 1e3:.0f} |")
    for a, label in MODELS.items():
        L += ["", f"## {label}: median sleep excl. pin / wake (s) by turn range, both GPUs", "",
              "| path | " + " | ".join(f"turns {x}-{y}" for x, y in RANGES) + " | KV GiB at last sleep |",
              "|---|---|---|---|---|"]
        for r in runs:
            cells = []
            for x, y in RANGES:
                s = med([q["total_s"] - pin_s(q) for g in r["gpus"] for q in r["inst"][(g, a)]["sus"] if x <= q["n"] <= y])
                w = med([q["total_s"] for g in r["gpus"] for i, q in enumerate(r["inst"][(g, a)]["res"], 1) if x <= i <= y])
                cells.append(f"{s:.2f} / {w:.2f}")
            last = r["inst"][(r["gpus"][0], a)]["sus"]
            L.append(f"| {r['paths']} | " + " | ".join(cells) + f" | {last[-1]['kv_bytes'] / GIB:.1f} |" if last else "")
    L += ["", "## SSD read rate during wakes (GiB/s, median over wakes with an SSD part)", "",
          "| path | " + " | ".join(MODELS.values()) + " | all |", "|---|---|---|---|---|"]
    for r in runs:
        per = {a: [x for g in r["gpus"] for q in r["inst"][(g, a)]["res"] if (x := wake_ssd_rate(q))] for a in MODELS}
        if not any(per.values()):
            continue
        cells = [f"{med([v for v, _ in per[a]]):.1f} ({med([n for _, n in per[a]]):.1f} GiB)" if per[a] else "-" for a in MODELS]
        L.append(f"| {r['paths']} | " + " | ".join(cells) + f" | {med([v for a in MODELS for v, _ in per[a]]):.1f} |")
    L += ["", "## DRAM held by tiered (logical bytes, all 6 instances, during the workload)", "",
          "| path | peak total GiB | at peak: weights / state / KV GiB | time-weighted mean GiB |", "|---|---|---|---|"]
    for r in runs:
        _, tot, v = r["dram_peak"]
        L.append(f"| {r['paths']} | {tot / GIB:.1f} | {v[0] / GIB:.1f} / {v[1] / GIB:.1f} / {v[2] / GIB:.1f} | {r['dram_mean'] / GIB:.1f} |")
    L += ["", "## Monitor", "",
          "| path | pcie.csv median interval ms | peak NVMe read, both members (GiB/s, 10 ms) | peak GPU RX (GiB/s, NVML) |",
          "|---|---|---|---|"]
    for r in runs:
        m = r["mon"]
        rx = ", ".join(f"g{g} {m.get(f'gpu{g}_rx_peak', float('nan')):.1f}" for g in r["gpus"])
        L.append(f"| {r['paths']} | {m.get('dt_ms', float('nan')):.1f} | {m.get('nvme_read_peak', float('nan')):.1f} | {rx} |")
    text = "\n".join(L) + "\n"
    print(text)
    open(out_path, "w").write(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
