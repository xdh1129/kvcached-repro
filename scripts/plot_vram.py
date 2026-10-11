"""Plot VRAM pressure timelines for one or more runs side by side.

Rows: GPU memory held per instance (+ device total), KV-cache usage %, queued
requests. Columns: runs. Rows share a y-scale so runs compare directly.

usage: python plot_vram.py results/<run_a> [results/<run_b> ...] [-o out.png]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "vram.csv"))
    cfg = {}
    with open(os.path.join(run_dir, "config.txt")) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            cfg[k] = v
    t0 = float(cfg.get("workload_start", df["t"].iloc[0]))
    df["s"] = df["t"] - t0
    with open(os.path.join(run_dir, "instances.tsv")) as f:
        insts = [line.split("\t")[0] for line in f.read().splitlines() if line]
    return df, insts, cfg


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def line(ax, x, y, color, label, direct):
    ax.plot(x, y, color=color, linewidth=2, label=label)
    if direct and y.notna().any():
        last = y.last_valid_index()
        ax.annotate(label, (x[last], y[last]), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    runs = [load(r) for r in args.runs]
    fig, axes = plt.subplots(3, len(runs), figsize=(6.5 * len(runs), 9), sharex="col",
                             sharey="row", squeeze=False)
    for col, ((df, insts, cfg), path) in enumerate(zip(runs, args.runs)):
        x = df["s"]
        ax_mem, ax_kv, ax_q = axes[:, col]
        gpu = next(c for c in df.columns if c.endswith("_used_mib") and c.startswith("gpu"))
        ax_mem.plot(x, df[gpu] / 1024, color=MUTED, linewidth=1.5, linestyle="--",
                    label="GPU total used")
        total = df[gpu.replace("_used_", "_total_")].iloc[0] / 1024
        ax_mem.axhline(total, color=MUTED, linewidth=0.8)
        for i, inst in enumerate(insts):
            c = SERIES[i]
            line(ax_mem, x, df[f"{inst}_proc_mib"] / 1024, c, inst, direct=True)
            if df[f"{inst}_kv_usage_pct"].notna().any():
                line(ax_kv, x, df[f"{inst}_kv_usage_pct"], c, inst, direct=True)
            if df[f"{inst}_waiting"].notna().any():
                line(ax_q, x, df[f"{inst}_waiting"], c, inst, direct=False)
        ax_mem.set_title(f"{os.path.basename(path.rstrip('/'))}  ({cfg.get('mode', '?')})",
                         color=INK, fontsize=11, loc="left")
        ax_mem.set_ylim(0, total * 1.05)
        ax_kv.set_ylim(0, 105)
        for ax in (ax_mem, ax_kv, ax_q):
            style(ax)
        ax_q.set_xlabel("seconds since workload start", color=MUTED, fontsize=9)
        if col == 0:
            ax_mem.set_ylabel("GPU memory (GiB)", color=MUTED, fontsize=9)
            ax_kv.set_ylabel("KV cache usage (%)", color=MUTED, fontsize=9)
            ax_q.set_ylabel("waiting requests", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = args.out or os.path.join(args.runs[0], "vram.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
