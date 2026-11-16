"""Peak DRAM held by tiered in each turn of the two-GPU runs (16:9 slide, English).

Reads <run_dir>/dram_logical.csv (written by analysis/dual_gpu_report.py): logical bytes
of weights + KV that all 6 instances (3 models x GPUs 6, 7) hold in DRAM, as a step
function of time. Turn k runs from the first request of turn k (requests.jsonl sent_t,
over both GPUs) to the first request of turn k + 1 (the last turn: to workload_end); its
peak is the largest total in force during that window, including the value at its start.

The phoenix paths store exactly what dram+gds / split store (only the SSD read differs),
so their per-turn peaks are identical and they are not drawn.

Output: results/comparison/slides_dual3m_g67/dram_peak_by_turn.png
usage: python analysis/plot_dual_dram.py [run_dir ...]
       (default: dual3m_g67 dram+dram, dram+gds, split f=0.20)
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "comparison", "slides_dual3m_g67")
DEFAULT = ["dual3m_g67_dram+dram", "dual3m_g67_dram+gds", "dual3m_g67_split_f0.20"]
STYLE = {  # path: (label, color, marker)
    "dram+dram": ("dram+dram", "#2a78d6", "o"),
    "dram+gds": ("dram+gds (= dram+phoenix)", "#eb6834", "s"),
    "split": ("split f=0.20 (= split+phoenix)", "#1baf7a", "D"),
}

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 20, "xtick.labelsize": 18,
    "ytick.labelsize": 18, "legend.fontsize": 18, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "lines.linewidth": 2.5,
})


def turn_peaks(d):
    """(path, [turn], [peak GiB of weights + KV])"""
    cfg = dict(l.split("=", 1) for l in open(f"{d}/config.txt").read().splitlines() if "=" in l)
    t_end = float(cfg["workload_end"])
    reqs = [json.loads(l) for l in open(f"{d}/requests.jsonl") if l.strip()]
    turns = sorted({r["turn"] for r in reqs})
    start = {k: min(r["sent_t"] for r in reqs if r["turn"] == k) for k in turns}
    rows = [(float(r["t"]), float(r["weights_gib"]) + float(r["kv_gib"]))
            for r in csv.DictReader(open(f"{d}/dram_logical.csv"))]
    peaks = []
    for i, k in enumerate(turns):
        a = start[k]
        b = start[turns[i + 1]] if i + 1 < len(turns) else t_end
        held = [v for t, v in rows if t <= a][-1:]  # value in force when the turn starts
        peaks.append(max(held + [v for t, v in rows if a < t <= b]))
    return cfg["paths"], turns, peaks


def main(dirs):
    runs = [turn_peaks(d if os.path.isdir(d) else os.path.join(ROOT, "results", d)) for d in dirs]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.11)
    for path, turns, peaks in runs:
        label, color, marker = STYLE.get(path, (path, INK2, "o"))
        ax.plot(turns, peaks, color=color, marker=marker, markersize=9, label=label)
        ax.annotate(f"{peaks[-1]:.0f}", (turns[-1], peaks[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=18, color=INK)
    ax.set_xticks(runs[0][1])
    ax.set_xlim(0.5, runs[0][1][-1] + 0.9)
    ax.set_ylim(180, 250)
    ax.set_xlabel("turn")
    ax.set_ylabel("peak DRAM held, weights + KV (GiB)")
    fig.text(0.04, 0.965, "Peak DRAM held by tiered offload per turn", fontsize=26, weight="bold", va="top")
    fig.text(0.04, 0.905, "Logical bytes of all 6 instances (Qwen3-32B / Qwen3-14B / Phi-3.5 on GPUs 6 and 7); "
             "weights alone are 194 GiB", fontsize=18, color=INK2, va="top")
    ax.legend(loc="upper left", frameon=False)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "dram_peak_by_turn.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
