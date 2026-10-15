"""Slide figure (16:9, English): the six offload-path configs as a table.

Each row: the config name with the line style used in the result plots, where the
weights and the KV go on sleep, and the path each takes back to the GPU on wake.

usage: python analysis/plot_configs.py [out_dir]   (default results/comparison/slides_sched)
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "comparison", "slides_sched")

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAY, BLUE, ORANGE, AQUA = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a"
# destination -> (chip label, edge color, fill)
DEST = {"dram": ("DRAM", BLUE, "#dce9f9"), "ssd": ("SSD", ORANGE, "#fbe3d8"),
        "gds": ("SSD (GDS)", AQUA, "#d6f1e6"), "drop": ("dropped", GRAY, SURFACE)}
PATH = {"dram": "DRAM → GPU", "ssd": "SSD → host buffer → GPU", "gds": "SSD → GPU (direct DMA)"}

# (group, label as in the result plots, line color, dashed, weights, KV)
ROWS = [
    ("Baseline", "DRAM / recompute", GRAY, False, "dram", "drop"),
    ("Single tier", "DRAM / DRAM", BLUE, False, "dram", "dram"),
    ("Single tier", "SSD / SSD", ORANGE, True, "ssd", "ssd"),
    ("Single tier", "GDS / GDS", AQUA, True, "gds", "gds"),
    ("Tiered", "DRAM / SSD", ORANGE, False, "dram", "ssd"),
    ("Tiered", "DRAM / GDS", AQUA, False, "dram", "gds"),
]
ROW_Y = [6.0, 4.85, 3.95, 3.05, 1.9, 1.0]
GROUP_LABEL = {"Baseline": "Baseline", "Single tier": "Single tier", "Tiered": "Tiered\n(ours)"}

plt.rcParams.update({"figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK})


def chip(ax, x, y, dest):
    label, ec, fc = DEST[dest]
    ax.add_patch(FancyBboxPatch((x - 0.95, y - 0.3), 1.9, 0.6, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=2, ls="--" if dest == "drop" else "-"))
    ax.text(x, y, label, ha="center", va="center", fontsize=16, fontweight="bold",
            color=INK2 if dest == "drop" else INK)


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
ax.text(0.64, 8.55, "Six offload configurations", fontsize=28, fontweight="bold", va="top")
ax.text(0.64, 7.97, "Weights are written to their store once at startup. On wake, weights and KV are restored "
        "in parallel.\nEvery config replays the same eviction schedule.", fontsize=18, color=INK2, va="top",
        linespacing=1.3)

# tiered rows highlighted
ax.add_patch(Rectangle((0.4, 0.45), 15.2, 2.0, fc="#f1f0ec", ec="none", zorder=0))
for x, h in [(3.05, "Config"), (7.25, "Weights →"), (9.55, "KV cache →"), (10.95, "Restore path on wake")]:
    ax.text(x, 6.75, h, fontsize=17, color=INK2, fontweight="bold", va="center",
            ha="center" if x in (7.25, 9.55) else "left")
for y in (6.45, 5.45, 2.45):
    ax.plot([0.4, 15.6], [y, y], color=GRID if y != 6.45 else INK2, lw=1.5)

for group in GROUP_LABEL:
    ys = [y for (g, *_), y in zip(ROWS, ROW_Y) if g == group]
    ax.text(0.64, sum(ys) / len(ys), GROUP_LABEL[group], fontsize=18, fontweight="bold", va="center",
            color=INK if group == "Tiered" else INK2)
for (group, label, color, dashed, w, kv), y in zip(ROWS, ROW_Y):
    ax.plot([2.3, 2.85], [y, y], color=color, lw=3.5, ls="--" if dashed else "-", solid_capstyle="butt")
    ax.text(3.05, y, label, fontsize=19, fontweight="bold", va="center", color=INK)
    chip(ax, 7.25, y, w)
    chip(ax, 9.55, y, kv)
    kv_path = "dropped; re-prefilled on wake" if kv == "drop" else PATH[kv]
    ax.text(10.95, y + 0.17, f"weights: {PATH[w]}", fontsize=15, va="center", color=INK)
    ax.text(10.95, y - 0.19, f"KV: {kv_path}", fontsize=15, va="center",
            color=DEST[kv][1] if kv != "drop" else INK2)

os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, "configs.png")
fig.savefig(path)
print("wrote", path)
