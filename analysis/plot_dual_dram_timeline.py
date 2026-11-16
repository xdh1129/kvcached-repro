"""DRAM held by tiered over the two-GPU runs, weights and KV stacked (16:9 slide, English).

Reads <run_dir>/dram_logical.csv (written by analysis/dual_gpu_report.py): logical bytes
of weights and KV that all 6 instances (3 models x GPUs 6, 7) hold in DRAM, as a step
function of time, from the first request of turn 1 to the end of the workload. One panel
per path, shared axes; weights light, KV dark, hue = path (as in wake_data_by_turn_*.png).
The x axis is turns: turn k spans [k, k + 1), from its first request (requests.jsonl, both
GPUs) to the next turn's first one (the last turn: to the end of the workload), with time
linear inside each turn, so the panels line up turn by turn although the runs differ in
speed. The time-weighted mean is over real time.
Each panel title gives the peak and the time-weighted mean of weights + KV.

Output: results/comparison/slides_dual3m_g67/dram_timeline.png
usage: python analysis/plot_dual_dram_timeline.py [run_dir ...]
       (default: dual3m_g67 dram+dram, dram+phoenix, split+phoenix f=0.20, meet+phoenix)
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "comparison", "slides_dual3m_g67")
DEFAULT = ["dual3m_g67_dram+dram", "dual3m_g67_dram+phoenix", "dual3m_g67_split+phoenix_f0.20",
           "dual3m_g67_meet+phoenix"]
STYLE = {  # path: (label, color)
    "dram+dram": ("dram+dram", "#2a78d6"),
    "dram+phoenix": ("dram+phoenix", "#eb6834"),
    "split+phoenix": ("split+phoenix f=0.20", "#1baf7a"),
    "meet+phoenix": ("meet+phoenix", "#e87ba4"),
    "split": ("split f=0.20 (= split+phoenix)", "#1baf7a"),
    "dram+gds": ("dram+gds (= dram+phoenix)", "#eb6834"),
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 20, "axes.titlesize": 19, "axes.labelsize": 20, "xtick.labelsize": 16,
    "ytick.labelsize": 17, "legend.fontsize": 16, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
})


def tint(color, f=0.45):
    c, s = np.array(to_rgb(color)), np.array(to_rgb(SURFACE))
    return tuple(s + (c - s) * f)


def load(d):
    cfg = dict(l.split("=", 1) for l in open(f"{d}/config.txt").read().splitlines() if "=" in l)
    t_end = float(cfg["workload_end"])
    reqs = [json.loads(l) for l in open(f"{d}/requests.jsonl") if l.strip()]
    turns = sorted({r["turn"] for r in reqs})
    starts = [min(r["sent_t"] for r in reqs if r["turn"] == k) for k in turns]
    t0 = starts[0]
    # meet splits the weight pool into static weights + mutable state: both count as weights here
    rows = [(float(r["t"]), float(r["weights_gib"]) + float(r.get("state_gib") or 0), float(r["kv_gib"]))
            for r in csv.DictReader(open(f"{d}/dram_logical.csv"))]
    before = [r for r in rows if r[0] <= t0]
    pts = [(0.0, *before[-1][1:])] if before else [(0.0, 0.0, 0.0)]
    pts += [(t - t0, w, k) for t, w, k in rows if t0 < t <= t_end]
    pts.append((t_end - t0, pts[-1][1], pts[-1][2]))
    t, w, k = map(np.array, zip(*pts))
    tot = w + k
    mean = float(np.sum(tot[:-1] * np.diff(t)) / (t[-1] - t[0]))
    edges = np.array([s - t0 for s in starts] + [t_end - t0])
    i = np.clip(np.searchsorted(edges, t, "right") - 1, 0, len(turns) - 1)
    x = np.array(turns)[i] + (t - edges[i]) / (edges[i + 1] - edges[i])  # turn coordinate
    return cfg["paths"], x, w, k, mean, turns


def main(dirs):
    runs = [load(d if os.path.isdir(d) else os.path.join(ROOT, "results", d)) for d in dirs]
    fig, axes = plt.subplots(len(runs), 1, sharey=True,
                             gridspec_kw={"hspace": 0.6 if len(runs) > 3 else 0.55, "left": 0.08, "right": 0.97,
                                          "top": 0.79 if len(runs) > 3 else 0.80, "bottom": 0.09})
    axes = np.atleast_1d(axes)
    for ax, (path, t, w, k, mean, turns) in zip(axes, runs):
        label, color = STYLE.get(path, (path, INK2))
        ax.fill_between(t, 0, w, step="post", color=tint(color), linewidth=0)
        ax.fill_between(t, w, w + k, step="post", color=color, linewidth=0)
        ax.axhline(mean, color=INK, linestyle="--", linewidth=1.8)
        i = int(np.argmax(w + k))
        ax.set_title(label, loc="left", pad=6)
        ax.text(1.0, 1.04, f"peak {w[i] + k[i]:.0f} GiB ({w[i]:.0f} weights + {k[i]:.0f} KV)  ·  mean {mean:.0f} GiB",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=15, color=INK2)
        ax.set_xticks([x + 0.5 for x in turns], [str(x) for x in turns])
        ax.set_xticks([x for x in turns] + [turns[-1] + 1], minor=True)
        ax.tick_params(axis="x", which="major", length=0)
        ax.grid(axis="x", which="major", visible=False)
        ax.grid(axis="x", which="minor", visible=True, color=INK2, linewidth=0.8, alpha=0.5)
        ax.set_xlim(turns[0], turns[-1] + 1)
        ax.set_ylim(0, 260)
        ax.set_yticks([0, 100, 200])
    axes[-1].set_xlabel("turn (each turn drawn the same width; time is linear within a turn)")
    fig.text(0.015, 0.45, "DRAM held (GiB)", rotation=90, va="center", fontsize=20)
    fig.text(0.04, 0.965, "DRAM held by tiered offload over the run: weights and KV", fontsize=26,
             weight="bold", va="top")
    fig.text(0.04, 0.91, "Logical bytes of 6 instances = 3 models x 2 GPUs (one weights copy each: "
             "97 GiB per GPU, 194 GiB in all)", fontsize=16, color=INK2, va="top")
    handles = [Patch(color=tint(INK2), label="weights (light shade)"), Patch(color=INK2, label="KV (dark shade)"),
               Line2D([], [], color=INK, linestyle="--", linewidth=1.8, label="time-weighted mean")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.885), ncol=3, frameon=False, fontsize=16)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "dram_timeline.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
