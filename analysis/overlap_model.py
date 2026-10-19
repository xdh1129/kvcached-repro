"""Fluid model of wakes on two GPUs behind one PCIe switch (GPUs 6+7), dram+ssd vs dram+gds.

Each wake loads its weights from DRAM and its KV from storage, on two threads. Transfers
share bandwidth max-min fairly within these limits (GiB/s, measured on GPUs 6+7):
  U     host -> switch uplink: DRAM weights of both GPUs + POSIX KV (host bounce)   53.7
  G_g   GPU g's own link: everything into GPU g                                       53.7
  R     NVMe RAID reads: POSIX and GDS KV                                             13.1
  per-flow caps: POSIX KV 13.1, GDS KV 12.4 (one process's I/O pipeline)
POSIX KV copies to GPU g wait while GPU g's own weights load (they queue behind them on
the GPU's host-to-device copy path; measured: B's 6.5 GB weights keep 53.6 GiB/s next to
its own POSIX KV), after a head start of the bounce buffers (8 x 64 MiB). Each KV
transfer also has a fixed start-up cost (POSIX 15 ms, GDS 20 ms, fitted).

  validate    the model against results/bench_overlap_wake and results/bench_cross_gpu
  replay      the wakes of a finished run (start times and sizes from its tiered logs)
  sweep       random wakes (Poisson per GPU, sizes drawn from a run) while scaling the
              wake rate, the RAID bandwidth and the model size

usage: python analysis/overlap_model.py [--run results/prism_18m_2g67_balanced_ts0.2_rep3_ref_dram+ssd]
         [--out results/overlap_model]
"""

import argparse
import glob
import json
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = 1 << 30
U, G, R = 53.7, 53.7, 13.1
CAP = {"ssd": 13.1, "gds": 12.4}
LAT = {"ssd": 0.015, "gds": 0.020}
PREFILL = 8 * 64 / 1024  # GiB of POSIX KV read ahead into bounce buffers


def rates(flows, raid):
    """max-min fair rates (GiB/s) of the active flows; a flow: dict(gpu, kind, ...)"""
    links = {"U": U, "R": raid}
    for f in flows:
        links[f"G{f['gpu']}"] = G
    uses = {}
    for i, f in enumerate(flows):
        u = [f"G{f['gpu']}"]
        if f["kind"] == "w" or f["kind"] == "ssd":
            u.append("U")
        if f["kind"] in ("ssd", "gds"):
            u.append("R")
        uses[i] = u
    cap = {i: (CAP[f["kind"]] * raid / R if f["kind"] in CAP else float("inf")) for i, f in enumerate(flows)}
    # progressive filling: raise all unfrozen flows together until a link or a flow cap saturates
    rate, left, active = {i: 0.0 for i in uses}, dict(links), set(uses)
    while active:
        step = min(min(left[l] / sum(1 for i in active if l in uses[i]) for l in links
                       if any(l in uses[i] for i in active)),
                   min(cap[i] - rate[i] for i in active))
        for i in active:
            rate[i] += step
            for l in uses[i]:
                left[l] -= step
        active = {i for i in active if cap[i] - rate[i] > 1e-9 and all(left[l] > 1e-9 for l in uses[i])}
    return rate


def simulate(wakes, kv_path, raid=R):
    """wakes: [(gpu, t_request, weights_gib, kv_gib)]; a GPU runs one wake at a time.
    Returns [(gpu, t_request, t_start, t_end)] in input order."""
    queue = sorted(range(len(wakes)), key=lambda i: wakes[i][1])
    busy, waiting = {}, {}  # gpu -> wake index running / list queued
    flows, out, t = [], {}, 0.0
    pending = list(queue)

    def start(i, now):
        g, _, w, k = wakes[i]
        busy[g] = i
        out[i] = [now, None]
        if w > 0:
            flows.append({"wake": i, "gpu": g, "kind": "w", "left": w, "delay": 0.0})
        if k > 0:
            left = k
            if kv_path == "ssd":
                left = max(0.0, k - PREFILL)
            flows.append({"wake": i, "gpu": g, "kind": kv_path, "left": left, "delay": LAT[kv_path]})

    while pending or flows or any(waiting.values()):
        # admit arrivals at t
        while pending and wakes[pending[0]][1] <= t + 1e-12:
            i = pending.pop(0)
            g = wakes[i][0]
            if g in busy:
                waiting.setdefault(g, []).append(i)
            else:
                start(i, t)
        for g in list(waiting):
            if waiting[g] and g not in busy:
                start(waiting[g].pop(0), t)
        moving = []
        for f in flows:
            if f["delay"] > 0:
                continue
            if f["kind"] == "ssd" and any(o["gpu"] == f["gpu"] and o["kind"] == "w" for o in flows):
                continue
            moving.append(f)
        r = rates(moving, raid) if moving else {}
        # next event: a flow ends, a delay ends, or an arrival
        dt = float("inf")
        for i, f in enumerate(moving):
            dt = min(dt, f["left"] / r[i])
        for f in flows:
            if f["delay"] > 0:
                dt = min(dt, f["delay"])
        if pending:
            dt = min(dt, wakes[pending[0]][1] - t)
        if dt == float("inf"):
            break
        for i, f in enumerate(moving):
            f["left"] -= r[i] * dt
        for f in flows:
            if f["delay"] > 0:
                f["delay"] = max(0.0, f["delay"] - dt)
        t += dt
        for f in [f for f in flows if f["delay"] == 0 and f["left"] <= 1e-9]:
            flows.remove(f)
            i = f["wake"]
            if not any(o["wake"] == i for o in flows):
                out[i][1] = t
                del busy[wakes[i][0]]
        # a wake with no flows at all
        for g, i in list(busy.items()):
            if not any(o["wake"] == i for o in flows) and out[i][1] is None:
                out[i][1] = t
                del busy[g]
    return [(wakes[i][0], wakes[i][1], out[i][0], out[i][1]) for i in range(len(wakes))]


def validate():
    rows = []
    bench = json.load(open(os.path.join(ROOT, "results", "bench_overlap_wake", "g67.json")))["results"]
    for b in bench:
        aw, bw, bk = (b[k] * 1e9 / GIB for k in ("a_weights_gb", "b_weights_gb", "b_kv_gb"))
        for kv in ("ssd", "gds"):
            alone = simulate([(7, 0.0, bw, 0.0)], kv)[0][3], simulate([(7, 0.0, 0.0, bk)], kv)[0][3]
            b_alone = simulate([(7, 0.0, bw, bk)], kv)[0][3]
            both = simulate([(6, 0.0, aw, 0.0), (7, 0.0, bw, bk)], kv)
            rows.append((b["event"], kv, "A weights with B", b[f"a_with_b_{kv}"]["weights"], both[0][3]))
            rows.append((b["event"], kv, "B wake alone", max(b[f"b_alone_{kv}"].values()), b_alone))
            rows.append((b["event"], kv, "B wake with A", max(b[f"b_with_a_{kv}"].values()), both[1][3]))
        rows.append((b["event"], "-", "A weights alone", b["a_alone"]["weights"], simulate([(6, 0.0, aw, 0.0)], "gds")[0][3]))
    return rows


def run_wakes(run_dir):
    gpu = {}
    for line in open(os.path.join(run_dir, "instances.tsv")):
        name, _, g, _ = line.rstrip("\n").split("\t")
        gpu[name] = int(g)
    wakes = []
    for f in glob.glob(os.path.join(run_dir, "tiered_*.jsonl")):
        m = os.path.basename(f)[len("tiered_"):-len(".jsonl")]
        for line in open(f):
            r = json.loads(line)
            if r["event"] == "resume":
                wakes.append((gpu[m], r["t_unix"], r["weights_bytes"] / GIB, r.get("kv_bytes", 0) / GIB))
    t0 = min(w[1] for w in wakes)
    return sorted([(g, t - t0, w, k) for g, t, w, k in wakes], key=lambda w: w[1])


def summarize(wakes, res):
    dur = [e - s for _, _, s, e in res]
    kv = [d for d, w in zip(dur, wakes) if w[3] > 1]
    return {"wakes": len(wakes), "wake_s_mean": float(np.mean(dur)), "wake_s_sum": float(np.sum(dur)),
            "kv_wakes": len(kv), "kv_wake_s_mean": float(np.mean(kv)) if kv else float("nan")}


def overlapped(res, wakes):
    n = 0
    for i, (g, _, s, e) in enumerate(res):
        if wakes[i][3] <= 1:
            continue
        if any(g2 != g and s2 < e and s < e2 and wakes[j][2] > 0 for j, (g2, _, s2, e2) in enumerate(res)):
            n += 1
    return n


def sweep(base, horizon, seeds):
    sizes = {g: [(w, k) for gg, _, w, k in base if gg == g and w + k > 0] for g in (6, 7)}
    rate = {g: len(sizes[g]) / horizon for g in (6, 7)}
    out = []
    for rate_x in (1, 2, 4, 8):
        for raid_x in (1, 2, 4):
            for size_x in (1, 2):
                acc = {"ssd": [], "gds": [], "ov": []}
                for seed in range(seeds):
                    rng = random.Random(seed)
                    wakes = []
                    for g in (6, 7):
                        t = 0.0
                        while True:
                            t += rng.expovariate(rate[g] * rate_x)
                            if t > horizon:
                                break
                            w, k = rng.choice(sizes[g])
                            wakes.append((g, t, w * size_x, k * size_x))
                    wakes.sort(key=lambda w: w[1])
                    for kv in ("ssd", "gds"):
                        res = simulate(wakes, kv, raid=R * raid_x)
                        acc[kv].append(summarize(wakes, res))
                        if kv == "ssd":
                            acc["ov"].append(overlapped(res, wakes) / max(1, sum(1 for w in wakes if w[3] > 1)))
                row = {"rate_x": rate_x, "raid_x": raid_x, "size_x": size_x,
                       "kv_wakes_overlapped_frac": float(np.mean(acc["ov"]))}
                for kv in ("ssd", "gds"):
                    row[f"{kv}_wake_s_mean"] = float(np.mean([a["wake_s_mean"] for a in acc[kv]]))
                    row[f"{kv}_kv_wake_s_mean"] = float(np.nanmean([a["kv_wake_s_mean"] for a in acc[kv]]))
                out.append(row)
    return out, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "results", "prism_18m_2g67_balanced_ts0.2_rep3_ref_dram+ssd"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "overlap_model"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = ["# Overlap model: dram+ssd vs dram+gds on GPUs 6+7", "", "## Validation against the microbenchmarks (s)", "",
             "| event | KV path | quantity | measured | model |", "|---|---|---|---|---|"]
    for ev, kv, q, meas, mod in validate():
        lines.append(f"| {ev} | {kv} | {q} | {meas:.3f} | {mod:.3f} |")

    base = run_wakes(args.run)
    horizon = max(w[1] for w in base)
    rep = {kv: summarize(base, simulate(base, kv)) for kv in ("ssd", "gds")}
    ov = overlapped(simulate(base, "ssd"), base)
    lines += ["", f"## Replay of the wakes of `{os.path.relpath(args.run, ROOT)}`", "",
              f"{len(base)} wakes over {horizon:.0f} s, {rep['ssd']['kv_wakes']} with KV > 1 GiB; "
              f"{ov} of those overlap a wake on the other GPU (model, dram+ssd).", "",
              "| KV path | wake s mean | wake s sum | KV wake s mean |", "|---|---|---|---|"]
    for kv in ("ssd", "gds"):
        r = rep[kv]
        lines.append(f"| {kv} | {r['wake_s_mean']:.3f} | {r['wake_s_sum']:.2f} | {r['kv_wake_s_mean']:.3f} |")

    rows, rate = sweep(base, horizon, args.seeds)
    lines += ["", "## What-if sweep (Poisson wakes, sizes drawn from the run)", "",
              f"Base wake rates: GPU 6 {rate[6]:.3f}/s, GPU 7 {rate[7]:.3f}/s. rate x: more frequent wakes; "
              f"RAID x: storage bandwidth (13.1 GiB/s x n, e.g. more NVMe drives); size x: weights and KV scaled. "
              f"Mean over {args.seeds} seeds; wake time includes queueing behind the GPU's previous wake.", "",
              "| rate x | RAID x | size x | KV wakes overlapping the other GPU | wake s mean ssd / gds | gain | "
              "KV wake s mean ssd / gds | gain |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g1 = 1 - r["gds_wake_s_mean"] / r["ssd_wake_s_mean"]
        g2 = 1 - r["gds_kv_wake_s_mean"] / r["ssd_kv_wake_s_mean"]
        lines.append(f"| {r['rate_x']} | {r['raid_x']} | {r['size_x']} | {100 * r['kv_wakes_overlapped_frac']:.0f}% "
                     f"| {r['ssd_wake_s_mean']:.3f} / {r['gds_wake_s_mean']:.3f} | {100 * g1:.0f}% "
                     f"| {r['ssd_kv_wake_s_mean']:.3f} / {r['gds_kv_wake_s_mean']:.3f} | {100 * g2:.0f}% |")
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "sweep.json"), "w") as f:
        json.dump({"replay": rep, "sweep": rows, "rates": rate}, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
