"""Memory of both GPUs side by side during the trace, Prism 8-model default trace, no swapping.

Each panel stacks the memory held by every model on that GPU, with the GPU
capacity and kvcached's safety headroom (1 - KVCACHED_GPU_UTILIZATION). Both
panels share the y scale.

usage: python analysis/plot_noswap_memory.py [--run results/prism_8m2g_noswap_default]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#b8b7b1"
MIB_TO_GIB = 1 / 1024
LEAD_S, TAIL_S = 10, 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "results", "prism_8m2g_noswap_default"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "comparison", "prism_8m2g_noswap_memory.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = dict(line.strip().split("=", 1) for line in open(os.path.join(args.run, "config.txt")) if "=" in line)
    inst = pd.read_csv(os.path.join(args.run, "instances.tsv"), sep="\t", header=None,
                       names=["name", "model", "gpu", "port"])
    inst["label"] = inst["name"] + " " + inst["model"].str.split("/").str[-1].str.replace("-Instruct", "") \
        .str.replace("-2407", "").str.replace("-v0.2", "").str.replace("Meta-", "")
    colors = {n: SERIES[i] for i, n in enumerate(inst["name"])}
    labels = dict(zip(inst["name"], inst["label"]))
    reqs = pd.read_json(os.path.join(args.run, "requests.jsonl"), lines=True)
    startup = pd.read_csv(os.path.join(args.run, "startup.tsv"), sep="\t")

    t_work = float(cfg["workload_start"])
    x_end = reqs.loc[reqs.ok, "done_s"].max() + TAIL_S
    vram = pd.read_csv(os.path.join(args.run, "vram.csv"))
    vram = vram[(vram.t >= t_work - LEAD_S) & (vram.t <= t_work + x_end)]
    x = vram.t - t_work
    util = float(cfg.get("kvcached_gpu_utilization", 0.95))
    gpus = sorted(inst.gpu.unique())

    fig, axes = plt.subplots(1, len(gpus), figsize=(7 * len(gpus), 5.6), sharey=True)
    for prism_idx, (ax, g) in enumerate(zip(axes, gpus)):
        names = list(inst.loc[inst.gpu == g, "name"])
        cap = vram[f"gpu{g}_total_mib"].iloc[0] * MIB_TO_GIB
        headroom = cap * (1 - util)
        ax.stackplot(x, *[vram[f"{n}_proc_mib"].fillna(0) * MIB_TO_GIB for n in names],
                     colors=[colors[n] for n in names], labels=[labels[n] for n in names], linewidth=0)
        ax.axhspan(cap - headroom, cap, color=NEUTRAL, alpha=0.5, linewidth=0, zorder=0)  # behind the stack
        ax.axhline(cap, color=INK, linewidth=1)
        ax.text(2, cap + 1, f"GPU capacity {cap:.1f} GiB", ha="left", va="bottom", fontsize=8, color=INK)
        ax.text(x_end, cap + 1, f"shaded: kvcached headroom {headroom:.1f} GiB", ha="right", va="bottom",
                fontsize=8, color=INK)

        peak = vram[f"gpu{g}_used_mib"].max() * MIB_TO_GIB
        weights = startup.loc[startup.name.isin(names), "instance_mib"].sum() * MIB_TO_GIB
        r = reqs[reqs.model.isin(names)]
        stuck = [n for n in names if (~r.loc[r.model == n, "ok"]).any()]
        ax.set_title(f"GPU {g} (Prism GPU{prism_idx})", loc="left", fontsize=12, color=INK, pad=34)
        lines = [f"4 models hold {weights:.1f} GiB when loaded; peak {peak:.1f} GiB, {cap - peak:.1f} GiB free",
                 f"requests completed: {int(r.ok.sum())} / {len(r)}"
                 + (f" — {', '.join(stuck)} never ran (no KV memory)" if stuck else " — all served")]
        ax.text(0, 1.02, "\n".join(lines), transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom",
                linespacing=1.4)

        ax.set_ylim(0, cap * 1.1)
        ax.set_xlim(-LEAD_S, x_end)
        ax.set_xlabel("seconds since trace start", color=MUTED, fontsize=9)
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=8, frameon=False, ncol=2,
                  handlelength=1.2)
    axes[0].set_ylabel("GPU memory (GiB)", color=MUTED, fontsize=9)

    fig.suptitle("GPU memory without weight swapping — Prism §7.2 (8 models / 2 GPUs, default trace), "
                 "RTX PRO 6000 96 GB", x=0.06, ha="left", fontsize=13, color=INK, y=1.04)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
