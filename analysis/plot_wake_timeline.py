"""Bandwidth over time inside a wake, from scripts/bench_wake_timeline.py (16:9 slides, English).

Each traced piece (start, end, bytes) is spread evenly over its interval and summed into
5 ms bins. Colors follow the slide encoding: weights from DRAM dark gray, KV through
POSIX SSD orange, KV through GDS aqua; GPU B's weights use a lighter gray, dashed.

Outputs: results/comparison/slides_18m_g67/wake_timeline_{dram_gds,dram_ssd,two_gpus}.png
usage: python analysis/plot_wake_timeline.py [results/bench_wake_timeline/g67.json]
"""

import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "comparison", "slides_18m_g67")
GIB = 1 << 30
BIN = 0.005

ORANGE, AQUA = "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
W_A, W_B = "#52514e", "#a3a29c"
KV_COLOR = {"ssd": ORANGE, "gds": AQUA}
KV_NAME = {"ssd": "SSD (POSIX)", "gds": "GDS"}

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 20, "xtick.labelsize": 18,
    "ytick.labelsize": 18, "legend.fontsize": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "lines.linewidth": 2.5,
})


def series(pieces, t_end):
    """(bin starts in ms, GiB/s per bin)"""
    edges = np.arange(0, t_end + BIN, BIN)
    acc = np.zeros(len(edges) - 1)
    for s, e, n in pieces:
        e = max(e, s + 1e-6)
        lo, hi = np.searchsorted(edges, s, "right") - 1, np.searchsorted(edges, e, "right") - 1
        for i in range(max(lo, 0), min(hi, len(acc) - 1) + 1):
            overlap = min(e, edges[i + 1]) - max(s, edges[i])
            if overlap > 0:
                acc[i] += n * overlap / (e - s)
    return edges[:-1] * 1e3, acc / BIN / GIB


def end_of(pieces):
    return max(e for _, e, _ in pieces)


def headline(fig, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=28, fontweight="bold", va="top")
    fig.text(0.04, 0.885, textwrap.fill(subtitle, 110), fontsize=18, color=INK2, va="top", linespacing=1.3)


def draw(ax, lines, t_end, ymax):
    """lines: [(pieces, color, linestyle, label)]"""
    for pieces, color, ls, label in lines:
        x, y = series(pieces, t_end)
        ax.step(np.append(x, x[-1] + BIN * 1e3), np.append(y, 0), where="post", color=color, linestyle=ls,
                linewidth=3, label=label)
        done = end_of(pieces) * 1e3
        ax.plot([done], [0], marker="v", markersize=12, color=color, clip_on=False, zorder=5)
    ax.set_xlim(0, t_end * 1e3)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Time since the wake started (ms)")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def fig_single(data, kv, t_end):
    tr = data["single"][kv]["trace"]
    (gpu, t), = tr.items()
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.08, right=0.72, top=0.74, bottom=0.12)
    draw(ax, [(t["weights"], W_A, "-", "weights from DRAM (7.8 GB)"),
              (t["kv"], KV_COLOR[kv], "-", f"KV from {KV_NAME[kv]} (5 GB)")], t_end, 60)
    ax.set_ylabel("GiB/s into the GPU")
    wake = max(end_of(t["weights"]), end_of(t["kv"])) * 1e3
    ax.axvline(wake, color=INK, linestyle=":", linewidth=2)
    ax.text(wake + 5, 57, f"wake done\n{wake:.0f} ms", va="top", fontsize=17)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="▼ = that path is done",
              title_fontsize=16, alignment="left")
    if kv == "gds":
        title = "DRAM + GDS: both paths move data at the same time"
        sub = ("One wake of a Phi-3-mini on GPU 6, weights and KV loaded on two threads. GDS reads go NVMe -> GPU "
               "inside the PCIe switch, so the KV starts at once; the two paths share the GPU's own link "
               "(~54 GiB/s), so the weights drop to ~41 GiB/s while the KV takes ~13.")
    else:
        title = "DRAM + SSD: the KV waits for the weights"
        sub = ("Same wake, KV through POSIX SSD reads into a host bounce buffer, then a host-to-GPU copy. "
               "Those copies queue behind the weights' copies, so the KV barely moves until the weights finish.")
    headline(fig, title, sub + f" Median of {len(data['single'][kv]['wake_s'])} reps.")
    save(fig, f"wake_timeline_dram_{kv}.png")


def fig_two_gpus(data, t_end):
    a, b = (str(g) for g in data["gpus"])
    fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.08))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.585, bottom=0.12)
    for ax, kv in zip(axes, ("ssd", "gds")):
        tr = data["two_gpus"][kv]["trace"]
        draw(ax, [(tr[a]["weights"], W_A, "-", f"GPU {a}: weights from DRAM (14.7 GB)"),
                  (tr[b]["weights"], W_B, "--", f"GPU {b}: weights from DRAM (6.5 GB)"),
                  (tr[b]["kv"], KV_COLOR[kv], "-", f"GPU {b}: KV (3.5 GB)")], t_end, 60)
        both = max(end_of(p) for g in (a, b) for p in tr[g].values()) * 1e3
        ax.axvline(both, color=INK, linestyle=":", linewidth=2)
        ax.text(both - 5, 57, f"both done\n{both:.0f} ms", va="top", ha="right", fontsize=17)
        ax.set_title(f"GPU {b}'s KV through {KV_NAME[kv]}", loc="left", fontweight="bold", pad=10, fontsize=20)
    axes[0].set_ylabel("GiB/s into each GPU")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=W_A, linewidth=3), Line2D([], [], color=W_B, linewidth=3, linestyle="--"),
               Line2D([], [], color=ORANGE, linewidth=3), Line2D([], [], color=AQUA, linewidth=3)]
    labels = [f"GPU {a}: weights from DRAM (14.7 GB)", f"GPU {b}: weights from DRAM (6.5 GB)",
              f"GPU {b}: KV from SSD (3.5 GB)", f"GPU {b}: KV from GDS (3.5 GB)"]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.06, 0.655), ncol=2, frameon=False, fontsize=16)
    headline(fig, f"Two GPUs wake together: GDS keeps GPU {b}'s KV off the uplink",
             f"GPU {a} and GPU {b} share one PCIe switch and its link to the CPU (~54 GiB/s for both). "
             f"With SSD, GPU {b}'s KV copies wait for its weights, then take uplink bandwidth from GPU {a}. "
             f"With GDS, the KV arrives from the NVMe inside the switch from the start.")
    save(fig, "wake_timeline_two_gpus.png")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "bench_wake_timeline", "g67.json")
    data = json.load(open(path))
    t_single = max(end_of(p) for kv in ("ssd", "gds") for p in next(iter(data["single"][kv]["trace"].values())).values())
    t_two = max(end_of(p) for kv in ("ssd", "gds") for g in data["two_gpus"][kv]["trace"].values() for p in g.values())
    for kv in ("gds", "ssd"):
        fig_single(data, kv, t_single * 1.12)
    fig_two_gpus(data, t_two * 1.08)


if __name__ == "__main__":
    main()
