"""Side-by-side figure: Prism 8-model default trace without vs with DRAM weight swapping.

(a) GPU 1 memory without swapping: the four models load one by one; the 4th OOMs.
(b) GPU 1 memory with DRAM swapping during the trace, with every wake/sleep marked.
(c) Requests that can be served in each case.
(d) Cost of each DRAM -> GPU wake, by model.

usage: python analysis/plot_comparison.py [--out results/comparison/prism_8m2g_comparison.png]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOSWAP = os.path.join(ROOT, "results", "prism_8m2g_kvcached_default")
SWAP = os.path.join(ROOT, "results", "prism_8m2g_swap_dram_default")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#b8b7b1"
MIB_TO_GIB = 1 / 1024


def load(run_dir):
    cfg = dict(line.strip().split("=", 1) for line in open(os.path.join(run_dir, "config.txt")) if "=" in line)
    inst = pd.read_csv(os.path.join(run_dir, "instances.tsv"), sep="\t", header=None,
                       names=["name", "model", "gpu", "port"])
    inst["label"] = inst["name"] + " " + inst["model"].str.split("/").str[-1].str.replace("-Instruct", "") \
        .str.replace("-2407", "").str.replace("-v0.2", "").str.replace("Meta-", "")
    return cfg, inst, pd.read_csv(os.path.join(run_dir, "vram.csv"))


def style(ax, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text, sub):
    ax.set_title(text, loc="left", fontsize=11, color=INK, pad=18)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def memory_panel(ax, vram, inst, gpu, t0, t1, colors, cap, cap_label_left=False):
    names = list(inst.loc[inst.gpu == gpu, "name"])
    labels = dict(zip(inst["name"], inst["label"]))
    w = vram[(vram.t >= t0) & (vram.t <= t1)]
    x = w.t - t0
    ax.stackplot(x, *[w[f"{n}_proc_mib"].fillna(0) * MIB_TO_GIB for n in names],
                 colors=[colors[n] for n in names], labels=[labels[n] for n in names], linewidth=0)
    ax.axhline(cap, color=INK, linewidth=1)
    ax.text(x.iloc[0] if cap_label_left else x.iloc[-1], cap + 1, f"GPU capacity {cap:.1f} GiB",
            ha="left" if cap_label_left else "right", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, cap * 1.12)
    ax.set_ylabel("GPU 1 memory (GiB)", color=MUTED, fontsize=9)
    style(ax)
    return x, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "comparison", "prism_8m2g_comparison.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg_n, inst, vram_n = load(NOSWAP)
    cfg_s, _, vram_s = load(SWAP)
    swaps = pd.read_json(os.path.join(SWAP, "swaps.jsonl"), lines=True)
    reqs = pd.read_json(os.path.join(SWAP, "requests.jsonl"), lines=True)
    gpu = int(inst.loc[inst.name == "model_1", "gpu"].iloc[0])
    cap = vram_n[f"gpu{gpu}_total_mib"].iloc[0] * MIB_TO_GIB
    colors = {n: SERIES[i] for i, n in enumerate(inst["name"])}  # one color per model in every panel

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.55, wspace=0.38)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # (a) no swapping: loading the 4 GPU-1 models until the OOM
    t0 = float(cfg_n["launch_start"])
    x, w = memory_panel(ax_a, vram_n, inst, gpu, t0, vram_n.t.iloc[-1], colors, cap, cap_label_left=True)
    peak = w[f"gpu{gpu}_used_mib"].idxmax()
    ax_a.annotate(f"model_4 OOM at {w.loc[peak, f'gpu{gpu}_used_mib'] * MIB_TO_GIB:.1f} GiB\n(before any request)",
                  (x[peak], w.loc[peak, f"gpu{gpu}_used_mib"] * MIB_TO_GIB), xytext=(-12, -50),
                  textcoords="offset points", ha="right", fontsize=8, color=INK,
                  arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax_a.set_xlabel("seconds since first model starts loading", color=MUTED, fontsize=9)
    title(ax_a, "(a) Without swapping", "4 models on GPU 1, loaded one by one — all resident at once")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0, 0.92), fontsize=7.5, frameon=False)

    # (b) DRAM swapping: the trace period, swap events marked
    t0, t1 = float(cfg_s["workload_start"]), float(cfg_s["workload_end"])
    x, _ = memory_panel(ax_b, vram_s, inst, gpu, t0, t1, colors, cap)
    g = swaps[swaps.gpu == gpu]
    for _, ev in g.iterrows():
        wake = ev.action == "wake"
        ax_b.scatter(ev.t_unix - t0, cap * (1.07 if wake else 1.035), marker="^" if wake else "v", s=34,
                     facecolors=colors[ev.instance] if wake else "none", edgecolors=colors[ev.instance],
                     linewidths=1.2, zorder=3)
    ax_b.set_xlabel("seconds since trace start", color=MUTED, fontsize=9)
    title(ax_b, "(b) With DRAM weight swapping",
          "same GPU and models (colors as in a); ▲ wake DRAM→GPU, ▽ sleep GPU→DRAM")

    # (c) requests that can be served
    total = len(reqs)
    unserved = int((reqs.model == "model_4").sum())
    rows = [("Without swapping", total - unserved, unserved, "have a running model"),
            ("DRAM swapping", int(reqs.ok.sum()), 0, "served (measured)")]
    for i, (lbl, ok, bad, ok_text) in enumerate(rows):
        y = len(rows) - 1 - i
        ax_c.barh(y, ok, color=SERIES[0], height=0.5, edgecolor="white", linewidth=2)
        ax_c.text(ok / 2, y, f"{ok} {ok_text}", ha="center", va="center", fontsize=9, color="white")
        if bad:
            ax_c.barh(y, bad, left=ok, color=NEUTRAL, height=0.5, edgecolor="white", linewidth=2, hatch="///")
            ax_c.text(ok + bad / 2, y, f"{bad} no server\n({bad / total:.0%})", ha="center", va="center",
                      fontsize=8.5, color=INK)
    ax_c.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=9, color=INK)
    ax_c.set_xlim(0, total)
    ax_c.set_xlabel(f"requests in the trace (total {total})", color=MUTED, fontsize=9)
    style(ax_c, ygrid=False)
    ax_c.grid(axis="x", color=GRID, linewidth=0.8)
    title(ax_c, "(c) Requests that can be served (both GPUs)",
          "without swapping, model_4 cannot start, so its requests have no server (not replayed)")

    # (d) cost of each wake
    wakes = swaps[swaps.action == "wake"].copy()
    wakes["label"] = wakes.instance.map(dict(zip(inst["name"], inst["label"])))
    wakes = wakes.sort_values("duration_s")
    ax_d.barh(range(len(wakes)), wakes.duration_s, color=[colors[n] for n in wakes.instance], height=0.6,
              edgecolor="white", linewidth=2)
    for i, (d, n) in enumerate(zip(wakes.duration_s, wakes.label)):
        ax_d.text(d + 0.008, i, f"{d:.2f} s", va="center", fontsize=8, color=INK)
    ax_d.set_yticks(range(len(wakes)), wakes.label, fontsize=8, color=INK)
    ax_d.set_xlim(0, wakes.duration_s.max() * 1.25)
    ax_d.set_xlabel("wake time, DRAM → GPU (s)", color=MUTED, fontsize=9)
    style(ax_d, ygrid=False)
    ax_d.grid(axis="x", color=GRID, linewidth=0.8)
    waited = pd.read_json(os.path.join(SWAP, "proxy_requests.jsonl"), lines=True).swap_wait_s
    title(ax_d, "(d) Swap cost in this trace (both GPUs)",
          f"{len(swaps)} swaps, {swaps.duration_s.sum():.1f} s in total; "
          f"{int((waited > 0.1).sum())}/{total} requests waited > 100 ms")

    fig.suptitle("Prism §7.2 workload (8 models / 2 GPUs, default trace) on RTX PRO 6000 96 GB",
                 x=0.07, ha="left", fontsize=13, color=INK, y=0.995)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
