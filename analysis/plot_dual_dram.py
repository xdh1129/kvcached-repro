"""DRAM held by tiered over the two-GPU runs (16:9 slide, English).

Reads <run_dir>/dram_logical.csv (written by analysis/dual_gpu_report.py): logical bytes
of weights and KV that all 6 instances (3 models x GPUs 6, 7) hold in DRAM, as a step
function of time. One panel per path, shared axes, weights stacked under KV. The x axis is
seconds since the workload started; the dashed line is the time-weighted mean over the
workload, the marker the peak (same numbers as results/comparison/dual3m_g67.md).

Output: results/comparison/slides_dual3m_g67/dram_logical.png
usage: python analysis/plot_dual_dram.py [run_dir ...]
       (default: dual3m_g67 dram+dram, dram+gds, split f=0.20)
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "comparison", "slides_dual3m_g67")
DEFAULT = ["dual3m_g67_dram+dram", "dual3m_g67_dram+gds", "dual3m_g67_split_f0.20"]
LABEL = {"dram+dram": "dram+dram: weights and KV in DRAM",
         "dram+gds": "dram+gds: weights in DRAM, KV on SSD",
         "dram+phoenix": "dram+phoenix: weights in DRAM, KV on SSD",
         "split": "split f=0.20: weights in DRAM, 80% of KV in DRAM",
         "split+phoenix": "split+phoenix f=0.20"}

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
W_COLOR, KV_COLOR = "#52514e", "#2a78d6"

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20, "xtick.labelsize": 18,
    "ytick.labelsize": 18, "legend.fontsize": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
})


def load(d):
    cfg = dict(l.split("=", 1) for l in open(f"{d}/config.txt").read().splitlines() if "=" in l)
    t0, t1 = float(cfg["workload_start"]), float(cfg["workload_end"])
    rows = [(float(r["t"]), float(r["weights_gib"]), float(r["kv_gib"]))
            for r in csv.DictReader(open(f"{d}/dram_logical.csv"))]
    # value in force at t0, then every change inside the workload, held until t1
    before = [r for r in rows if r[0] <= t0]
    w0, k0 = (before[-1][1], before[-1][2]) if before else (0.0, 0.0)
    pts = [(0.0, w0, k0)] + [(t - t0, w, k) for t, w, k in rows if t0 < t <= t1]
    pts.append((t1 - t0, pts[-1][1], pts[-1][2]))
    t, w, k = map(np.array, zip(*pts))
    tot = w + k
    mean = float(np.sum(tot[:-1] * np.diff(t)) / (t[-1] - t[0]))
    i = int(np.argmax(tot))
    return cfg["paths"], t, w, k, mean, i


def main(dirs):
    runs = [load(os.path.join(ROOT, "results", d) if not os.path.isdir(d) else d) for d in dirs]
    ymax = max(r[2].max() + r[3].max() for r in runs)
    fig, axes = plt.subplots(len(runs), 1, sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45, "left": 0.08, "right": 0.97, "top": 0.83, "bottom": 0.15})
    axes = np.atleast_1d(axes)
    for ax, (path, t, w, k, mean, i) in zip(axes, runs):
        ax.fill_between(t, 0, w, step="post", color=W_COLOR, linewidth=0)
        ax.fill_between(t, w, w + k, step="post", color=KV_COLOR, linewidth=0)
        ax.axhline(mean, color=INK, linestyle="--", linewidth=2)
        ax.plot([t[i]], [w[i] + k[i]], marker="v", markersize=11, color=INK, zorder=5, clip_on=False)
        ax.set_title(LABEL.get(path, path), loc="left", pad=8)
        ax.text(1.0, 1.04, f"peak {w[i] + k[i]:.0f} GiB ({w[i]:.0f} weights + {k[i]:.0f} KV)   ·   mean {mean:.0f} GiB",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=17, color=INK2)
        ax.set_ylim(0, ymax * 1.08)
        ax.set_yticks(np.arange(0, ymax * 1.08, 100))
        ax.set_xlim(0, max(r[1][-1] for r in runs))
    axes[-1].set_xlabel("time since workload start (s)", labelpad=4)
    fig.text(0.015, 0.49, "DRAM held (GiB)", rotation=90, va="center", fontsize=20)
    fig.text(0.04, 0.965, "DRAM held by tiered offload, 2 GPUs x 3 models", fontsize=26, weight="bold", va="top")
    fig.text(0.04, 0.905, "Logical bytes of the 6 sleeping instances, Qwen3-32B / Qwen3-14B / Phi-3.5 on GPUs 6 and 7, 15 turns",
             fontsize=18, color=INK2, va="top")
    handles = [Patch(color=W_COLOR, label="weights"), Patch(color=KV_COLOR, label="KV cache"),
               Line2D([], [], color=INK, linestyle="--", linewidth=2, label="time-weighted mean"),
               Line2D([], [], color=INK, marker="v", linestyle="", markersize=11, label="peak")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "dram_logical.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
