"""Slide diagram (16:9, English): restore paths of the two tiered configs, DRAM / SSD vs DRAM / GDS.

Both restore the weights from host DRAM. KV comes back from the SSD: through a pinned
host bounce buffer (POSIX O_DIRECT + cudaMemcpy) or by DMA straight into GPU memory (GDS).

usage: python analysis/plot_data_paths.py [out_dir]   (default results/comparison/slides_sched)
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "comparison", "slides_sched")

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
W_C, ORANGE, AQUA = "#52514e", "#eb6834", "#1baf7a"
BOX = "#f1f0ec"

plt.rcParams.update({"figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "font.size": 20})


def box(ax, x, y, w, h, title, sub=None, fc=BOX, ec=INK2, lw=2, size=22, title_y=None):
    """Rounded box; the title sits at title_y (default: centered)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw, zorder=1))
    ty = y + h / 2 if title_y is None else title_y
    ax.text(x + w / 2, ty + (0.25 if sub else 0), title, ha="center", va="center", fontsize=size,
            fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, ty - 0.4, sub, ha="center", va="center", fontsize=15, color=INK2, zorder=3)


def arrow(ax, p, q, color, label=None, lpos=None, ha="center", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=0.9,head_width=0.5", mutation_scale=18,
                                 color=color, lw=5, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=4))
    if label:
        ax.text(*lpos, label, ha=ha, va="center", fontsize=16, color=color, fontweight="bold", zorder=5)


def panel(ax, title, gds):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0, 10.0, title, fontsize=24, fontweight="bold", va="top")
    ax.text(0, 9.25, "weights and KV restored in parallel on wake", fontsize=15, color=INK2,
            style="italic", va="top")
    kv_c, kv_fc = (AQUA, "#d6f1e6") if gds else (ORANGE, "#fbe3d8")
    # GPU on top; host DRAM bottom left, NVMe bottom right, contents above their titles
    box(ax, 1.8, 6.0, 6.4, 2.0, "GPU", "weights + KV (kvcached pages)")
    box(ax, 0.2, 0.4, 4.2, 3.6, "Host DRAM", title_y=1.2)
    box(ax, 5.6, 0.4, 4.2, 3.6, "NVMe SSD", "RAID0", title_y=1.2)
    box(ax, 0.5, 2.6, 1.7, 1.0, "weights", fc="#e2e1dc", ec=W_C, lw=1.5, size=16)
    box(ax, 6.0, 2.6, 3.4, 1.0, "KV", fc=kv_fc, ec=kv_c, lw=1.5, size=16)
    arrow(ax, (1.35, 3.65), (3.0, 5.95), W_C, "weights", (1.55, 4.9), ha="right")
    if gds:
        ax.text(3.3, 3.1, "no KV\nin DRAM", ha="center", va="center", fontsize=15, color=AQUA, fontweight="bold")
        arrow(ax, (7.7, 3.65), (6.6, 5.95), kv_c, "DMA to GPU\n(cuFile)", (7.55, 4.8), ha="left")
    else:
        box(ax, 2.4, 2.6, 1.7, 1.0, "bounce\nbuffer", fc=kv_fc, ec=kv_c, lw=1.5, size=15)
        arrow(ax, (5.95, 3.1), (4.15, 3.1), kv_c, "① read", (4.95, 3.6))
        arrow(ax, (3.25, 3.65), (5.0, 5.95), kv_c, "② copy", (4.45, 4.8), ha="left")


fig, axes = plt.subplots(1, 2, gridspec_kw=dict(wspace=0.12))
fig.subplots_adjust(left=0.04, right=0.97, top=0.76, bottom=0.03)
panel(axes[0], "DRAM / SSD  (POSIX I/O)", gds=False)
panel(axes[1], "DRAM / GDS  (GPUDirect Storage)", gds=True)
fig.text(0.04, 0.95, "Tiered restore: weights from DRAM, KV from SSD", fontsize=28, fontweight="bold", va="top")
fig.text(0.04, 0.885, "Weights are written to DRAM once at startup and read back on every wake. KV goes to the SSD;\n"
         "with POSIX I/O it passes through a pinned host buffer, with GDS the SSD DMAs it straight into GPU memory.",
         fontsize=18, color=INK2, va="top", linespacing=1.3)
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, "data_paths.png")
fig.savefig(path)
print("wrote", path)
