"""Plot memory_demand.py output: per-GPU demand timelines for two load levels.

usage: python analysis/plot_memory_demand.py [--reps 1] [--dir results/memory_demand]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS, KV = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "memory_demand"))
    ap.add_argument("--reps", type=int, nargs="+", default=[1])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    capacity = json.load(open(os.path.join(args.dir, "params.json")))["capacity_gib"]

    fig, axes = plt.subplots(len(args.gpus), len(args.reps), figsize=(6 * len(args.reps), 3.6 * len(args.gpus)),
                             sharey=True, squeeze=False)
    for r, gpu in enumerate(args.gpus):
        for c, rep in enumerate(args.reps):
            ax = axes[r, c]
            d = pd.read_csv(os.path.join(args.dir, f"timeline_gpu{gpu}_rep{rep}.csv"))
            ax.stackplot(d["t"], d["weights_ondemand_gib"], d["kv_gib"], colors=[WEIGHTS, KV],
                         labels=["weights of active models", "KV cache"], linewidth=0)
            ax.plot(d["t"], d["weights_static_gib"] + d["kv_gib"], color=MUTED, linewidth=1.5,
                    linestyle="--", label="all models resident (static)")
            ax.axhline(capacity, color=INK, linewidth=1)
            ax.annotate(f"GPU capacity {capacity:.1f} GiB", (d["t"].iloc[-1], capacity),
                        xytext=(0, -4), textcoords="offset points", ha="right", va="top", fontsize=8, color=INK)
            ax.set_title(f"GPU{gpu} — load ×{rep}", loc="left", fontsize=11, color=INK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel("GPU memory demand (GiB)", color=MUTED, fontsize=9)
            if r == len(args.gpus) - 1:
                ax.set_xlabel("seconds (trace × 0.2)", color=MUTED, fontsize=9)
    axes[0, 0].set_ylim(0, None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(args.dir, "memory_demand.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
