"""Data-movement time of one model's wake in every turn of the two-GPU runs (16:9 slide).

For each run and turn k, the k-th resume of the model (--agent: 1 Qwen3-32B, 2 Qwen3-14B,
3 Phi-3.5) on GPUs 6 and 7: its data-movement time, from the first job's start to the last
byte in (data_span in plot_dual_wake_bw.py: DRAM jobs end after their stream sync, SSD jobs
at data_end_s, leaving out Phoenix / GDS cleanup). The two GPUs wake at the same time; the
line is their mean, the band their min-max. The y axis is zoomed to the data.

Output: results/comparison/slides_dual3m_g67/wake_data_by_turn_<model>.png
usage: python analysis/plot_dual_wake_by_turn.py [--agent 1] [run_dir ...]
       (default: dual3m_g67 dram+dram, dram+phoenix, split+phoenix, split+phoenix+w)
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_dual_wake_bw import MODELS, data_span

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "comparison", "slides_dual3m_g67")
DEFAULT = ["dual3m_g67_dram+dram", "dual3m_g67_dram+phoenix", "dual3m_g67_split+phoenix_f0.20",
           "dual3m_g67_split+phoenix+w_f0.20"]
STYLE = {  # path: (label, color, marker); categorical slots 1-4 in order
    "dram+dram": ("dram+dram", "#2a78d6", "o"),
    "dram+phoenix": ("dram+phoenix", "#eb6834", "s"),
    "split+phoenix": ("split+phoenix f=0.20", "#1baf7a", "D"),
    "split+phoenix+w": ("split+phoenix+w f=0.20", "#eda100", "^"),
}

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "lines.linewidth": 2,
})


def per_turn(d, agent):
    """(path, turns, [[data s of each GPU] per turn])"""
    cfg = dict(l.split("=", 1) for l in open(f"{d}/config.txt").read().splitlines() if "=" in l)
    per = []
    for g in cfg["gpus"].split():
        res = [json.loads(l) for l in open(f"{d}/tiered_g{g}_a{agent}.jsonl") if '"resume"' in l]
        per.append([b - a for a, b in map(data_span, res)])
    n = min(map(len, per))
    return cfg["paths"], np.arange(1, n + 1), np.array([p[:n] for p in per]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=int, default=1, choices=sorted(MODELS))
    ap.add_argument("runs", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    runs = [per_turn(d if os.path.isdir(d) else os.path.join(ROOT, "results", d), args.agent) for d in args.runs]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.11)
    for path, turns, v in runs:
        label, color, marker = STYLE.get(path, (path, INK2, "o"))
        mean = v.mean(axis=1)
        ax.fill_between(turns, v.min(axis=1), v.max(axis=1), color=color, alpha=0.18, linewidth=0)
        ax.plot(turns, mean, color=color, marker=marker, markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    turns = runs[0][1]
    ax.set_xticks(turns)
    ax.set_xlim(turns[0] - 0.4, turns[-1] + 0.4)
    lo = min(r[2].min() for r in runs)
    hi = max(r[2].max() for r in runs)
    ax.set_ylim(np.floor(lo * 10 - 1) / 10, np.ceil(hi * 10 + 1) / 10)  # zoomed: y does not start at 0
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("turn")
    ax.set_ylabel("data-movement time of the wake (s)")
    name = MODELS[args.agent][0]
    fig.text(0.04, 0.965, f"{name} wake on GPUs 6 and 7 at once: time moving data, per turn", fontsize=26,
             weight="bold", va="top")
    fig.text(0.04, 0.905, "First job start to last byte in. Line: mean of GPUs 6, 7; band: min-max. "
             "y axis does not start at 0", fontsize=18, color=INK2, va="top")
    fig.legend(loc="upper left", bbox_to_anchor=(0.035, 0.87), ncol=4, frameon=False)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"wake_data_by_turn_{MODELS[args.agent][1]}.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
