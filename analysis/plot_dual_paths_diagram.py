"""Slide diagram (16:9, 繁中): wake data paths of five configs in the dual-GPU (6+7) experiment.

Every panel draws the same topology: host DRAM -> uplink (shared by both GPUs) -> PCIe switch ->
GPU and the NVMe RAID0 members. Arrows show where each part comes from on wake:
blue = from DRAM, orange = from SSD. The bar inside the GPU shows the share each source restores.
Only one GPU is drawn; GPUs 6 and 7 do the same thing at the same time.

Outputs (results/comparison/slides_dual3m_g67/):
  paths_diagram.png             all five panels + legend
  paths_diagram_<path>.png      one panel each

usage: python analysis/plot_dual_paths_diagram.py [out_dir]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "comparison", "slides_dual3m_g67")

INK, INK2, SURFACE, BOX = "#0b0b0b", "#52514e", "#fcfcfb", "#f1f0ec"
DRAM_C, SSD_C = "#2a78d6", "#eb6834"
UPLINK_FC = "#dcdad3"

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "font.size": 14,
                     "font.family": ["Noto Sans CJK TC", "DejaVu Sans"]})


def box(ax, x, y, w, h, title, sub=None, fc=BOX, ec=INK2, lw=1.8, size=15, ty=None, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=fc, ec=ec, lw=lw, zorder=1, alpha=alpha))
    ty = y + h / 2 if ty is None else ty
    ax.text(x + w / 2, ty + (0.22 if sub else 0), title, ha="center", va="center", fontsize=size,
            fontweight="bold", zorder=3, alpha=alpha)
    if sub:
        ax.text(x + w / 2, ty - 0.3, sub, ha="center", va="center", fontsize=11, color=INK2, zorder=3, alpha=alpha)


def route(ax, pts, color, dashed=False):
    """Polyline through pts with an arrowhead on the last segment."""
    ls = (0, (4, 2.5)) if dashed else "-"
    xs, ys = zip(*pts[:-1])
    if len(pts) > 2:
        ax.plot(xs, ys, color=color, lw=4, ls=ls, solid_capstyle="round", zorder=4)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=0.7,head_width=0.4",
                                 mutation_scale=16, color=color, lw=4, linestyle=ls, shrinkA=0, shrinkB=0,
                                 zorder=4))


def label(ax, x, y, s, color, ha="center", size=12):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))


def gpu_bar(ax, segs, meet=False):
    """segs: [(fraction, color, text)] drawn left to right inside the GPU box."""
    x0, y0, w, h = 0.6, 1.45, 3.6, 0.7
    x = x0
    for frac, c, t in segs:
        ax.add_patch(Rectangle((x, y0), w * frac, h, fc=c, ec=SURFACE, lw=1.5, zorder=3))
        ax.text(x + w * frac / 2, y0 + h / 2, t, ha="center", va="center", fontsize=10.5, color="white",
                fontweight="bold", zorder=4)
        x += w * frac
    if meet:  # meeting point of the two weight cursors
        mx = x0 + w * (segs[0][0] + segs[1][0])
        ax.plot([mx, mx], [y0 - 0.15, y0 + h + 0.15], color=INK, lw=2, zorder=5)
        ax.text(mx, y0 - 0.38, "→ 會合點 ←", ha="center", va="center", fontsize=10, color=INK, zorder=5)


# path -> drawing spec
PATHS = {
    "dram+dram": dict(
        title="dram+dram（baseline）", sleep="sleep：weights、KV 全部存 DRAM",
        dram="weights + KV", ssd=None,
        bar=[(0.6, DRAM_C, "weights"), (0.4, DRAM_C, "KV")],
        foot="uplink 下行：weights + KV 全部"),
    "dram+ssd": dict(
        title="dram+ssd（baseline）", sleep="sleep：weights 存 DRAM，KV 存 SSD（POSIX I/O）",
        dram="weights", ssd="bounce",
        bar=[(0.6, DRAM_C, "weights"), (0.4, SSD_C, "KV")],
        foot="KV 繞經 DRAM，和 weights 搶 uplink 下行"),
    "dram+phoenix": dict(
        title="dram+phoenix（baseline）", sleep="sleep：weights 存 DRAM，KV 經 GDS 存 SSD",
        dram="weights", ssd="p2p", ssd_label="KV\nPhoenix P2P",
        bar=[(0.6, DRAM_C, "weights"), (0.4, SSD_C, "KV")],
        foot="uplink 下行：只有 weights"),
    "split": dict(
        title="split（ours）", sleep="sleep：尾端 ≤20%（只限 KV）存 SSD，其餘存 DRAM",
        dram="weights + KV 前段", ssd="p2p", ssd_label="KV 尾段\nP2P (GDS/Phoenix)",
        bar=[(0.6, DRAM_C, "weights"), (0.2, DRAM_C, "KV"), (0.2, SSD_C, "KV")],
        foot="DRAM、SSD 同時讀，SSD 不佔 uplink"),
    "meet": dict(
        title="meet+phoenix（ours）", sleep="sleep：全部存 DRAM；weights 首次 sleep 另存一份到 SSD",
        dram="KV → weights（前→後）", ssd="p2p", ssd_label="weights\n（後→前）P2P",
        bar=[(0.24, DRAM_C, "KV"), (0.4, DRAM_C, "weights"), (0.36, SSD_C, "weights")], meet=True,
        foot="兩端一起讀 weights，同時結束"),
}


def panel(ax, spec):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    ax.text(0.1, 10.55, spec["title"], fontsize=17, fontweight="bold", va="top")
    ax.text(0.1, 9.85, spec["sleep"], fontsize=11.5, color=INK2, va="top")

    # topology
    box(ax, 2.3, 7.3, 5.4, 1.4, "" if spec["ssd"] == "bounce" else "Host DRAM")
    ax.add_patch(Rectangle((3.9, 5.8), 2.2, 1.5, fc=UPLINK_FC, ec="none", zorder=0.5))
    ax.text(6.25, 6.55, "uplink\n(2 GPU 共用)", ha="left", va="center", fontsize=10.5, color=INK2)
    box(ax, 0.6, 4.8, 7.6, 1.0, "")
    ax.text(0.85, 5.3, "PCIe switch", ha="left", va="center", fontsize=13, fontweight="bold", zorder=3)
    box(ax, 0.3, 0.75, 4.2, 2.65, "GPU", ty=2.85)
    used = spec["ssd"] is not None
    box(ax, 5.5, 0.75, 4.2, 2.65, "NVMe RAID0", "nvme1 + nvme2", ty=2.0, alpha=1.0 if used else 0.35)
    gpu_bar(ax, spec["bar"], spec.get("meet", False))

    # DRAM -> GPU
    route(ax, [(4.3, 7.3), (4.3, 5.3), (2.2, 3.4)], DRAM_C)
    label(ax, 3.75, 6.55, spec["dram"], DRAM_C, ha="right")

    if spec["ssd"] == "p2p":
        route(ax, [(7.6, 3.4), (6.2, 5.05), (3.4, 3.4)], SSD_C)
        label(ax, 7.75, 4.25, spec["ssd_label"], SSD_C, ha="left", size=11)
    elif spec["ssd"] == "bounce":
        ax.add_patch(FancyBboxPatch((5.25, 7.45), 2.25, 0.75, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="#fbe3d8", ec=SSD_C, lw=1.4, zorder=2))
        ax.text(6.37, 7.83, "bounce buffer", ha="center", va="center", fontsize=10.5, zorder=3)
        route(ax, [(7.6, 3.4), (5.7, 5.3), (5.7, 7.45)], SSD_C)            # ① SSD -> DRAM (up)
        route(ax, [(5.0, 7.45), (5.0, 5.3), (3.2, 3.4)], SSD_C, dashed=True)  # ② DRAM -> GPU (down)
        label(ax, 7.4, 4.3, "① KV 讀進 DRAM", SSD_C, ha="left", size=11)
        label(ax, 0.3, 4.25, "② 再拷到 GPU", SSD_C, ha="left", size=11)
        ax.text(2.4, 8.35, "Host DRAM", fontsize=13, fontweight="bold", va="center")

    ax.text(0.1, 0.25, spec["foot"], fontsize=12, color=INK, va="center", fontweight="bold")


def legend(ax):
    ax.axis("off")
    handles = [Line2D([], [], color=DRAM_C, lw=4, label="從 DRAM 讀"),
               Line2D([], [], color=SSD_C, lw=4, label="從 SSD 讀"),
               Line2D([], [], color=SSD_C, lw=4, ls=(0, (4, 2.5)), label="SSD 資料經 DRAM 轉送"),
               Rectangle((0, 0), 1, 1, fc=UPLINK_FC, ec="none", label="uplink：兩張 GPU 共用 (~58 GiB/s)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=14, handlelength=2.6,
              borderaxespad=0.2, labelspacing=0.9)
    ax.text(0.02, 0.38, "• GPU 6、7 拓樸相同、同時 wake，圖只畫一張\n"
            "• GPU 和 NVMe 在同一個 PCIe switch 下，\n   P2P 讀取不經過 uplink\n"
            "• GPU 內長條：wake 後各部分由誰載入\n   （比例只是示意）",
            transform=ax.transAxes, fontsize=12.5, color=INK2, va="top", linespacing=1.5)


os.makedirs(OUT, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 9), gridspec_kw=dict(wspace=0.06, hspace=0.1))
fig.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.01)
for ax, spec in zip(axes.flat, PATHS.values()):
    panel(ax, spec)
legend(axes.flat[5])
fig.text(0.02, 0.965, "Wake 資料路徑：五種 path", fontsize=24, fontweight="bold", va="top")
path = os.path.join(OUT, "paths_diagram.png")
fig.savefig(path)
print("wrote", path)
plt.close(fig)

for name, spec in PATHS.items():
    fig = plt.figure(figsize=(6.4, 6.6))
    ax = fig.add_axes([0.02, 0.01, 0.96, 0.97])
    panel(ax, spec)
    path = os.path.join(OUT, f"paths_diagram_{name.replace('+', '_')}.png")
    fig.savefig(path)
    print("wrote", path)
    plt.close(fig)
