"""PCIe traffic during one model's wake on both GPUs at once (16:9 slide, English).

For a model (--agent: 1 Qwen3-32B, 2 Qwen3-14B, 3 Phi-3.5) and a turn (default 15), takes
that instance's wake for the turn on GPUs 6 and 7 (the k-th resume in
tiered_g<gpu>_a<agent>.jsonl) in each run, and plots around it:
  GPU 6 RX, GPU 7 RX   pcie_nvml.csv: NVML PCIe RX, one sample ~ every 24 ms; a sample
                       stamped t (when the call returned) is drawn over (previous t, t]
  both GPUs            their sum on a 2 ms grid
  NVMe read            pcie.csv: nvme1n1 + nvme2n1 read (the RAID0 members), 10 ms
Reference lines: the switch uplink to the host (~58 GiB/s for both GPUs together, the
plateau dram+dram reaches) and the two disks' read ceiling (~13 GiB/s).
t = 0 is when the earlier of the two wakes started. The "data" time of a wake is when it
really moved data: from the first job's start (after the GPU memory is mapped again) to the
last byte in (DRAM jobs: their end, after the stream sync; SSD jobs: data_end_s, which
leaves out the Phoenix / GDS cleanup). Markers show when each GPU's data was all in.

Each panel title says how many GiB that wake read from DRAM and from SSD (per GPU).

Output: results/comparison/slides_dual3m_g67/wake_bw_<model>_turn<k><suffix>.png
usage: python analysis/plot_dual_wake_bw.py [--agent 1] [--turn 15] [--suffix _x] [run_dir ...]
       (default: dual3m_g67 dram+dram, dram+phoenix, split+phoenix f=0.20)
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "comparison", "slides_dual3m_g67")
DEFAULT = ["dual3m_g67_dram+dram", "dual3m_g67_dram+phoenix", "dual3m_g67_split+phoenix_f0.20"]
GIB = 1 << 30
UPLINK, DISKS = 58, 13.2  # GiB/s, measured: dram+dram plateau; Phoenix/GDS ceiling of both members
PAD = 0.3
GRID_S = 0.002
NAME = {"dram+ssd": "dram+ssd (POSIX)", "split": "split f=0.20", "split+phoenix": "split+phoenix f=0.20",
        "split+phoenix+w": "split+phoenix+w f=0.20"}
MODELS = {1: ("Qwen3-32B", "32b"), 2: ("Qwen3-14B", "14b"), 3: ("Phi-3.5", "phi")}

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_G6, C_G7, C_SUM, C_NVME = "#2a78d6", "#6250d6", "#0b0b0b", "#eb6834"

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 20, "axes.titlesize": 18, "axes.labelsize": 20, "xtick.labelsize": 17,
    "ytick.labelsize": 17, "legend.fontsize": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "lines.linewidth": 2.2,
})


def sources(rec):
    """(GiB read from DRAM, GiB read from SSD) by one wake: each job with <job>_host_bytes"""
    jobs = [k[:-len("_host_bytes")] for k in rec if k.endswith("_host_bytes")]
    # a POSIX job (path "ssd") reads the SSD into a host bounce buffer: its bytes are SSD bytes
    host = sum(rec[f"{j}_host_bytes"] for j in jobs if rec.get(f"{j}_path") != "ssd")
    return host / GIB, (sum(rec[f"{j}_bytes"] for j in jobs) - host) / GIB


def data_span(rec):
    """(start, end) in s since the wake started, of the wake's data movement"""
    jobs = [k[:-len("_host_bytes")] for k in rec if k.endswith("_host_bytes")]
    jobs = [j for j in jobs if f"{j}_start_s" in rec]
    start = min(rec.get(f"{j}_data_start_s", rec[f"{j}_start_s"]) if rec[f"{j}_host_bytes"] == 0
                else rec[f"{j}_start_s"] for j in jobs)
    end = max(rec.get(f"{j}_data_end_s", rec[f"{j}_end_s"]) for j in jobs)
    return start, end


def load(d, turn, agent):
    cfg = dict(l.split("=", 1) for l in open(f"{d}/config.txt").read().splitlines() if "=" in l)
    gpus = [int(g) for g in cfg["gpus"].split()]
    wakes, rec, data = {}, None, {}
    for g in gpus:
        res = [json.loads(l) for l in open(f"{d}/tiered_g{g}_a{agent}.jsonl") if '"resume"' in l]
        r = res[turn - 1]
        wakes[g] = (r["t_unix"], r["t_unix"] + r["total_s"])
        data[g] = data_span(r)
        rec = rec if g != gpus[0] else r
    t0 = min(a for a, _ in wakes.values())
    lo, hi = t0 - PAD, max(b for _, b in wakes.values()) + PAD
    # NVML RX per GPU: (t, GiB/s) samples, each covering (previous t, t]
    nv = {g: [] for g in gpus}
    for r in csv.DictReader(open(f"{d}/pcie_nvml.csv")):
        t = float(r["t"])
        if r["dir"] == "rx" and lo - 0.1 <= t <= hi + 0.1 and int(r["gpu"]) in nv:
            nv[int(r["gpu"])].append((t, float(r["mbs"]) * 1e6 / GIB))
    grid = np.arange(lo, hi, GRID_S)
    rx = {}
    for g, s in nv.items():
        ts, vs = np.array([x for x, _ in s]), np.array([v for _, v in s])
        idx = np.clip(np.searchsorted(ts, grid, "left"), 0, len(ts) - 1)  # first sample ending at/after
        rx[g] = vs[idx]
    nvme_t, nvme_v = [], []
    for r in csv.DictReader(open(f"{d}/pcie.csv")):
        t = float(r["t"])
        if lo - 0.05 <= t <= hi + 0.05:
            nvme_t.append(t)
            nvme_v.append((float(r["nvme1n1_read_mbs"]) + float(r["nvme2n1_read_mbs"])) * 1e6 / GIB)
    return {"path": cfg["paths"], "gpus": gpus, "t0": t0, "grid": grid - t0, "rx": rx,
            "nvme": (np.array(nvme_t) - t0, np.array(nvme_v)),
            "ends": {g: b - t0 for g, (_, b) in wakes.items()},
            "starts": {g: a - t0 for g, (a, _) in wakes.items()}, "rec": rec,
            "data": {g: (wakes[g][0] - t0 + a, wakes[g][0] - t0 + b) for g, (a, b) in data.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=int, default=1, choices=sorted(MODELS))
    ap.add_argument("--turn", type=int, default=15)
    ap.add_argument("--suffix", default="", help="appended to the output file name")
    ap.add_argument("runs", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    runs = [load(d if os.path.isdir(d) else os.path.join(ROOT, "results", d), args.turn, args.agent) for d in args.runs]
    fig, axes = plt.subplots(len(runs), 1, sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.42, "left": 0.08, "right": 0.86, "top": 0.78, "bottom": 0.09})
    axes = np.atleast_1d(axes)
    colors = dict(zip(runs[0]["gpus"], (C_G6, C_G7)))
    for ax, r in zip(axes, runs):
        x = r["grid"]
        for g in r["gpus"]:
            ax.plot(x, r["rx"][g], color=colors[g], linewidth=1.8, drawstyle="steps-pre")
        both = sum(r["rx"].values())
        ax.plot(x, both, color=C_SUM, linewidth=2.4, drawstyle="steps-pre")
        nt, nvv = r["nvme"]
        ax.plot(nt, nvv, color=C_NVME, linewidth=2.2, drawstyle="steps-pre")
        for y, txt in ((UPLINK, f"uplink {UPLINK} GiB/s"), (DISKS, f"2 disks {DISKS:.0f} GiB/s")):
            ax.axhline(y, color=INK2, linestyle=":", linewidth=1.6)
            ax.text(1.005, y, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=15, color=INK2)
        for g in r["gpus"]:
            ax.plot([r["data"][g][1]], [0], marker="v", markersize=11, color=colors[g], clip_on=False, zorder=5)
        inside = (x >= min(a for a, _ in r["data"].values())) & (x <= max(b for _, b in r["data"].values()))
        p90 = np.percentile(both[inside], 90)
        dram, ssd = sources(r["rec"])
        src = f"{dram:.1f} GiB DRAM" + (f" + {ssd:.1f} GiB SSD" if ssd else "")
        ax.set_title(f"{NAME.get(r['path'], r['path'])}: {src}", loc="left", pad=6)
        moving = " / ".join(f"{r['data'][g][1] - r['data'][g][0]:.2f}" for g in r["gpus"])
        ax.text(1.0, 1.03, f"moving data {moving} s  ·  p90 both {p90:.0f} GiB/s", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=14, color=INK2)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])
    axes[-1].set_xlim(0, max(max(r["ends"].values()) for r in runs) + PAD)
    axes[-1].set_xlabel("time since the wake started (s)")
    fig.text(0.015, 0.465, "throughput (GiB/s)", rotation=90, va="center", fontsize=20)
    rec0 = runs[0]["rec"]
    fig.text(0.04, 0.965, f"{MODELS[args.agent][0]} waking on GPUs 6 and 7 at once, turn {args.turn}", fontsize=26,
             weight="bold", va="top")
    where = "" if any(r["path"].endswith("+w") or r["path"].startswith("meet") for r in runs) else " (always from DRAM)"
    fig.text(0.04, 0.91, f"Per GPU: {rec0['weights_bytes'] / GIB:.1f} GiB weights{where} + "
             f"{rec0['kv_bytes'] / GIB:.1f} GiB KV. GPU RX: NVML, ~24 ms samples. NVMe: 10 ms", fontsize=17, color=INK2, va="top")
    handles = [Line2D([], [], color=C_G6, linewidth=2.5, label="GPU 6 RX"),
               Line2D([], [], color=C_G7, linewidth=2.5, label="GPU 7 RX"),
               Line2D([], [], color=C_SUM, linewidth=2.5, label="both GPUs RX"),
               Line2D([], [], color=C_NVME, linewidth=2.5, label="NVMe read (2 members)"),
               Line2D([], [], color=INK2, marker="v", linestyle="", markersize=11, label="data all in")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.885), ncol=5, frameon=False, fontsize=16)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"wake_bw_{MODELS[args.agent][1]}_turn{args.turn}{args.suffix}.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
