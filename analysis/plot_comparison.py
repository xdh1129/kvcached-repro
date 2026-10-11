"""Prism 8-model default trace on both GPUs: without vs with DRAM weight swapping.

Rows 1-2 (GPU 1, GPU 2): memory per model during the trace, without swapping
(left) and with DRAM swapping (right, every wake/sleep marked). All four panels
share one y scale.
Row 3: (e) GPU 1 startup without swapping: after the 4th model loads, free memory is
below kvcached's safety headroom (1 - KVCACHED_GPU_UTILIZATION of the GPU), so that
model can never map KV; (f) requests served in each run (both measured);
(g) time of each DRAM -> GPU wake.
Without swapping, the panels end shortly after the last request that completed:
requests stuck waiting for KV only end at the client's 1-hour timeout.

usage: python analysis/plot_comparison.py [--out results/comparison/prism_8m2g_comparison.png]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOSWAP = os.path.join(ROOT, "results", "prism_8m2g_noswap_default")
SWAP = os.path.join(ROOT, "results", "prism_8m2g_swap_dram_default")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#b8b7b1"
MIB_TO_GIB = 1 / 1024
LEAD_S = 10  # idle seconds shown before the trace starts
TAIL_S = 10  # seconds shown after the last completed request


def load(run_dir):
    cfg = dict(line.strip().split("=", 1) for line in open(os.path.join(run_dir, "config.txt")) if "=" in line)
    inst = pd.read_csv(os.path.join(run_dir, "instances.tsv"), sep="\t", header=None,
                       names=["name", "model", "gpu", "port"])
    sizes = {"Mistral-Nemo": "12B", "Llama-3.1-8B": "8B", "Mistral-7B": "7B", "Llama-3.2-3B": "3B"}
    inst["label"] = inst["name"] + " " + inst["model"].str.split("/").str[-1].str.replace("-Instruct", "") \
        .str.replace("-2407", "").str.replace("-v0.2", "").str.replace("Meta-", "")
    reqs = pd.read_json(os.path.join(run_dir, "requests.jsonl"), lines=True)
    inst["short"] = inst["name"] + " · " + inst["label"].str.split(" ").str[1].map(sizes)
    return cfg, inst, pd.read_csv(os.path.join(run_dir, "vram.csv")), reqs


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text, sub):
    lines = sub.count("\n") + 1
    ax.set_title(text, loc="left", fontsize=11, color=INK, pad=8 + 11 * lines)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom", linespacing=1.4)


def memory_panel(ax, vram, inst, gpu, t0, t1, colors, cap, x0=None):
    """Stacked per-model memory on one GPU between t0 and t1 (x relative to x0)."""
    names = list(inst.loc[inst.gpu == gpu, "name"])
    labels = dict(zip(inst["name"], inst["label"]))
    w = vram[(vram.t >= t0) & (vram.t <= t1)]
    x = w.t - (x0 if x0 is not None else t0)
    ax.stackplot(x, *[w[f"{n}_proc_mib"].fillna(0) * MIB_TO_GIB for n in names],
                 colors=[colors[n] for n in names], labels=[labels[n] for n in names], linewidth=0)
    ax.axhline(cap, color=INK, linewidth=1)
    ax.text(x.iloc[-1], cap + 1, f"GPU capacity {cap:.1f} GiB", ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, cap * 1.12)
    style(ax)
    return x, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "comparison", "prism_8m2g_comparison.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg_n, inst, vram_n, reqs_n = load(NOSWAP)
    cfg_s, _, vram_s, reqs_s = load(SWAP)
    swaps = pd.read_json(os.path.join(SWAP, "swaps.jsonl"), lines=True)
    failed = pd.read_csv(os.path.join(NOSWAP, "failed.tsv"), sep="\t", header=None,
                         names=["name", "gpu", "port", "s"]) if os.path.getsize(os.path.join(NOSWAP, "failed.tsv")) \
        else pd.DataFrame(columns=["name", "gpu"])
    gpus = sorted(inst.gpu.unique())
    cap = vram_n[f"gpu{gpus[0]}_total_mib"].iloc[0] * MIB_TO_GIB
    colors = {n: SERIES[i] for i, n in enumerate(inst["name"])}  # one color per model in every panel

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.85], hspace=0.75, wspace=1.3)
    mem_axes = {}
    for r, g in enumerate(gpus):
        mem_axes[g] = (fig.add_subplot(gs[r, 0:3]), fig.add_subplot(gs[r, 3:6]))
    util = float(cfg_n.get("kvcached_gpu_utilization", 0.95))
    headroom = cap * (1 - util)
    t_work_n = float(cfg_n["workload_start"])
    t_last_ok = t_work_n + reqs_n.loc[reqs_n.ok, "done_s"].max() + TAIL_S

    def preempted(name):
        w = vram_n[(vram_n.t >= t_work_n) & (vram_n.t <= t_last_ok)][f"{name}_preemptions"].dropna()
        return int(w.iloc[-1] - w.iloc[0]) if len(w) else 0
    ax_e, ax_f, ax_g = fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[2, 2:4]), fig.add_subplot(gs[2, 4:6])

    letters = iter("abcd")
    for g in gpus:
        ax_n, ax_s = mem_axes[g]
        names = list(inst.loc[inst.gpu == g, "name"])
        # without swapping: the trace period on whatever came up
        memory_panel(ax_n, vram_n, inst, g, t_work_n - LEAD_S, t_last_ok, colors, cap, x0=t_work_n)
        down = [n for n in names if n in set(failed.name)]
        stuck = [n for n in names if n not in down and (~reqs_n.loc[reqs_n.model == n, "ok"]).any()]
        pre = {n: preempted(n) for n in names if preempted(n)}
        parts = [f"{n} failed to start" for n in down]
        parts += [f"{n} got no KV memory: {int((reqs_n.model == n).sum())} requests never ran" for n in stuck]
        top = max(pre, key=pre.get) if pre else None
        if top:
            ttft_n = reqs_n.loc[(reqs_n.model == top) & reqs_n.ok, "ttft_s"].median()
            ttft_s = reqs_s.loc[(reqs_s.model == top) & reqs_s.ok, "ttft_s"].median()
            parts.append(f"{top} preempted {pre[top]}×, TTFT p50 {ttft_n:.0f} s (swapping: {ttft_s:.2f} s)")
        sub = "\n".join(parts) or "all 4 models resident, no failures"
        title(ax_n, f"({next(letters)}) GPU {g} — without swapping", sub)
        ax_n.legend(loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=7.5, frameon=False, ncol=4,
                    handlelength=1.2, columnspacing=1.2)
        ax_n.set_ylabel(f"GPU {g} memory (GiB)", color=MUTED, fontsize=9)
        # with DRAM swapping
        t0, t1 = float(cfg_s["workload_start"]), float(cfg_s["workload_end"])
        memory_panel(ax_s, vram_s, inst, g, t0 - LEAD_S, t1, colors, cap, x0=t0)
        ev = swaps[swaps.gpu == g]
        for _, e in ev.iterrows():
            wake = e.action == "wake"
            ax_s.scatter(e.t_unix - t0, cap * (1.07 if wake else 1.035), marker="^" if wake else "v", s=34,
                         facecolors=colors[e.instance] if wake else "none", edgecolors=colors[e.instance],
                         linewidths=1.2, zorder=3)
        n_wake, n_sleep = int((ev.action == "wake").sum()), int((ev.action == "sleep").sum())
        title(ax_s, f"({next(letters)}) GPU {g} — DRAM weight swapping",
              f"all 4 models served\n{n_wake} wakes ▲, {n_sleep} sleeps ▽ (colors as left)")
        for ax in (ax_n, ax_s):
            ax.set_xlabel("seconds since trace start", color=MUTED, fontsize=9)

    # (e) why it fails: GPU 1 startup without swapping, until the trace starts
    g_fail = int(inst.loc[inst.name == "model_1", "gpu"].iloc[0])
    startup = pd.read_csv(os.path.join(NOSWAP, "startup.tsv"), sep="\t")
    on_gpu = startup[startup.gpu == g_fail]
    t0 = float(cfg_n["launch_start"])
    t_loaded = t0 + on_gpu.startup_s.sum() + 20  # GPU-1 instances launch first, one after another
    x, w = memory_panel(ax_e, vram_n, inst, g_fail, t0, t_loaded, colors, cap)
    ax_e.axhspan(cap - headroom, cap, color=NEUTRAL, alpha=0.5, linewidth=0)
    ax_e.text(x.iloc[0], cap - headroom / 2, f"kvcached safety headroom {headroom:.1f} GiB "
              f"({1 - util:.0%})", va="center", fontsize=7.5, color=INK)
    used = on_gpu.gpu_used_mib.iloc[-1] * MIB_TO_GIB
    ax_e.annotate(f"{len(on_gpu)} models loaded: {used:.1f} GiB used\n{cap - used:.1f} GiB free < headroom\n"
                  f"→ no KV for {on_gpu.name.iloc[-1]}",
                  (x.iloc[-1] - 5, used), xytext=(0.03, 0.80), textcoords="axes fraction", ha="left", va="top",
                  fontsize=8, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax_e.set_ylabel(f"GPU {g_fail} memory (GiB)", color=MUTED, fontsize=9)
    ax_e.set_xlabel("seconds since first model starts loading", color=MUTED, fontsize=9)
    title(ax_e, f"(e) GPU {g_fail} startup, no swapping",
          "an earlier identical run hit OOM at this step instead")

    # (f) requests served, both measured
    total = len(reqs_s)
    rows = [("Without\nswapping", int(reqs_n.ok.sum()), int((~reqs_n.ok).sum())),
            ("DRAM\nswapping", int(reqs_s.ok.sum()), int((~reqs_s.ok).sum()))]
    for i, (lbl, ok, bad) in enumerate(rows):
        y = len(rows) - 1 - i
        ax_f.barh(y, ok, color=SERIES[0], height=0.5, edgecolor="white", linewidth=2)
        ax_f.text(ok / 2, y, f"{ok} served", ha="center", va="center", fontsize=9, color="white")
        if bad:
            ax_f.barh(y, bad, left=ok, color=NEUTRAL, height=0.5, edgecolor="white", linewidth=2, hatch="///")
            ax_f.text(ok + bad / 2, y, f"{bad}\nfailed\n({bad / total:.0%})", ha="center", va="center",
                      fontsize=8, color=INK)
    ax_f.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=9, color=INK)
    ax_f.set_xlim(0, total)
    ax_f.set_xlabel(f"requests in the trace (total {total})", color=MUTED, fontsize=9)
    style(ax_f, grid_axis="x")
    failed_models = ", ".join(sorted(reqs_n.loc[~reqs_n.ok, "model"].unique())) or "none"
    title(ax_f, "(f) Requests served (measured)",
          f"failed without swapping: all of {failed_models} (no KV memory, timed out)")

    # (g) cost of each wake
    wakes = swaps[swaps.action == "wake"].copy()
    wakes["label"] = wakes.instance.map(dict(zip(inst["name"], inst["short"])))
    wakes = wakes.sort_values("duration_s")
    ax_g.barh(range(len(wakes)), wakes.duration_s, color=[colors[n] for n in wakes.instance], height=0.6,
              edgecolor="white", linewidth=2)
    for i, d in enumerate(wakes.duration_s):
        ax_g.text(d + 0.008, i, f"{d:.2f} s", va="center", fontsize=8, color=INK)
    ax_g.set_yticks(range(len(wakes)), wakes.label, fontsize=8, color=INK)
    ax_g.set_xlim(0, wakes.duration_s.max() * 1.3)
    ax_g.set_xlabel("wake time, DRAM → GPU (s)", color=MUTED, fontsize=9)
    style(ax_g, grid_axis="x")
    waited = pd.read_json(os.path.join(SWAP, "proxy_requests.jsonl"), lines=True).swap_wait_s
    title(ax_g, "(g) Swap cost (both GPUs)",
          f"{len(swaps)} swaps, {swaps.duration_s.sum():.1f} s total; "
          f"{int((waited > 0.1).sum())}/{total} req. waited >100 ms")

    fig.suptitle("Prism §7.2 workload (8 models / 2 GPUs, default trace) on RTX PRO 6000 96 GB",
                 x=0.07, ha="left", fontsize=13, color=INK, y=0.995)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
