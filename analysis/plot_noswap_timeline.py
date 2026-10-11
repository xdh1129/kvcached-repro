"""Request timeline on the GPU that runs out of memory, Prism 8-model default trace, no swapping.

Four panels share one time axis (seconds since trace start):
  (a) GPU memory per model, with kvcached's safety headroom
  (b) every request as one line: queued (light) then decoding (model color);
      requests that never got a first token run to the right edge (hatched)
  (c) cumulative requests arrived vs completed, per model
  (d) requests waiting in each engine's queue (vLLM metric)

usage: python analysis/plot_noswap_timeline.py [--run results/prism_8m2g_noswap_default] [--gpu N]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#b8b7b1"
MIB_TO_GIB = 1 / 1024
LEAD_S, TAIL_S = 10, 25


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text, sub):
    ax.set_title(text, loc="left", fontsize=11, color=INK, pad=20)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "results", "prism_8m2g_noswap_default"))
    ap.add_argument("--gpu", type=int, default=None, help="GPU to show (default: the one hosting model_1)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "comparison", "prism_8m2g_noswap_timeline.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = dict(line.strip().split("=", 1) for line in open(os.path.join(args.run, "config.txt")) if "=" in line)
    inst = pd.read_csv(os.path.join(args.run, "instances.tsv"), sep="\t", header=None,
                       names=["name", "model", "gpu", "port"])
    inst["label"] = inst["name"] + " " + inst["model"].str.split("/").str[-1].str.replace("-Instruct", "") \
        .str.replace("-2407", "").str.replace("-v0.2", "").str.replace("Meta-", "")
    colors = {n: SERIES[i] for i, n in enumerate(inst["name"])}
    gpu = args.gpu if args.gpu is not None else int(inst.loc[inst.name == "model_1", "gpu"].iloc[0])
    names = list(inst.loc[inst.gpu == gpu, "name"])
    labels = dict(zip(inst["name"], inst["label"]))

    reqs = pd.read_json(os.path.join(args.run, "requests.jsonl"), lines=True)
    reqs = reqs[reqs.model.isin(names)]
    t_work = float(cfg["workload_start"])
    x_end = reqs.loc[reqs.ok, "done_s"].max() + TAIL_S
    stopped_s = float(cfg["workload_end"]) - t_work
    vram = pd.read_csv(os.path.join(args.run, "vram.csv"))
    vram = vram[(vram.t >= t_work - LEAD_S) & (vram.t <= t_work + x_end)]
    vx = vram.t - t_work
    cap = vram[f"gpu{gpu}_total_mib"].iloc[0] * MIB_TO_GIB
    headroom = cap * (1 - float(cfg.get("kvcached_gpu_utilization", 0.95)))

    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6, 1, 0.8], "hspace": 0.42})
    ax_m, ax_r, ax_c, ax_q = axes

    # (a) memory
    ax_m.stackplot(vx, *[vram[f"{n}_proc_mib"].fillna(0) * MIB_TO_GIB for n in names],
                   colors=[colors[n] for n in names], labels=[labels[n] for n in names], linewidth=0)
    ax_m.axhspan(cap - headroom, cap, color=NEUTRAL, alpha=0.5, linewidth=0)
    ax_m.axhline(cap, color=INK, linewidth=1)
    ax_m.text(2, cap + 1, f"GPU capacity {cap:.1f} GiB", ha="left", va="bottom", fontsize=8, color=INK)
    ax_m.text(x_end, cap + 1, f"shaded: kvcached safety headroom {headroom:.1f} GiB", ha="right",
              va="bottom", fontsize=8, color=INK)
    ax_m.set_ylim(0, cap * 1.1)
    ax_m.set_ylabel("GPU memory (GiB)", color=MUTED, fontsize=9)
    ax_m.legend(loc="upper left", fontsize=8, frameon=False, ncol=4, bbox_to_anchor=(0, -0.06),
                handlelength=1.2)
    kv = {n: vram[f"{n}_kv_used_mib"].max() * MIB_TO_GIB for n in names}
    title(ax_m, f"(a) GPU {gpu} memory: weights fill the GPU",
          f"used {vram[f'gpu{gpu}_used_mib'].max() * MIB_TO_GIB:.1f} of {cap:.1f} GiB; "
          "KV in use at most " + ", ".join(f"{n} {v:.2f} GiB" for n, v in kv.items()))
    style(ax_m)

    # (b) one line per request, grouped by model, ordered by arrival
    y, ticks = 0, []
    for n in names:
        d = reqs[reqs.model == n].sort_values("scheduled_s")
        start, block = y, max(len(d), 70)  # minimum block height keeps small groups' labels apart
        y += (block - len(d)) / 2
        for _, q in d.iterrows():
            if q.ok:
                ax_r.hlines(y, q.sent_s, q.first_token_s, color=colors[n], alpha=0.25, linewidth=1.2)
                ax_r.hlines(y, q.first_token_s, q.done_s, color=colors[n], linewidth=1.2)
            else:
                ax_r.hlines(y, q.sent_s, x_end, color=NEUTRAL, linewidth=1.2)
            y += 1
        ticks.append((start + block / 2, f"{labels[n]}\n{len(d)} req, {int(d.ok.sum())} done"))
        y = start + block + 6
        ax_r.axhline(y - 3, color=GRID, linewidth=0.8)
    ax_r.set_yticks([t for t, _ in ticks], [lbl for _, lbl in ticks], fontsize=8, color=INK)
    ax_r.set_ylim(y, -2)
    ax_r.grid(axis="x", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax_r.spines[side].set_visible(False)
    ax_r.tick_params(colors=MUTED, labelsize=8)
    title(ax_r, "(b) Every request, from arrival to completion",
          "light = queued, waiting for KV memory · solid = decoding · gray = never got a first token "
          f"(still queued when the run was stopped at {stopped_s:.0f} s)")

    # (c) cumulative arrived vs completed
    grid_t = np.arange(-LEAD_S, x_end, 0.25)
    finals = {}
    for n in names:
        d = reqs[reqs.model == n]
        arrived = np.searchsorted(np.sort(d.sent_s.values), grid_t, side="right")
        done = np.searchsorted(np.sort(d.loc[d.ok, "done_s"].values), grid_t, side="right")
        ax_c.plot(grid_t, arrived, color=colors[n], linewidth=1.5, linestyle="--")
        ax_c.plot(grid_t, done, color=colors[n], linewidth=2)
        finals.setdefault((int(arrived[-1]), int(done[-1])), []).append(n)
    for (a, dn), ns in finals.items():  # one label per distinct end state, so equal curves share it
        ax_c.annotate(f"{', '.join(ns)}: {a} arrived, {dn} done" + (" each" if len(ns) > 1 else ""),
                      (grid_t[-1], a), xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8,
                      color=INK)
    ax_c.set_ylabel("requests (cumulative)", color=MUTED, fontsize=9)
    title(ax_c, "(c) Requests arrived (dashed) vs completed (solid)",
          "the gap is work the GPU could not run")
    style(ax_c)

    # (d) queue depth and preemptions
    for n in names:
        q = vram[f"{n}_waiting"]
        if q.notna().any():
            ax_q.plot(vx, q, color=colors[n], linewidth=2)
            pre = vram[f"{n}_preemptions"].dropna()
            p = int(pre.iloc[-1] - pre.iloc[0]) if len(pre) else 0
            i = q.idxmax()
            if q.max() > 0:
                ax_q.annotate(f"{n}: up to {int(q.max())} waiting"
                              + (f", {p} preemption{'s' if p != 1 else ''}" if p else ""),
                              (vx[i], q[i]), xytext=(6, 2), textcoords="offset points", fontsize=8, color=INK)
    ax_q.set_ylabel("requests waiting", color=MUTED, fontsize=9)
    ax_q.set_xlabel("seconds since trace start (trace arrivals end at 120 s)", color=MUTED, fontsize=9)
    title(ax_q, "(d) Requests waiting in each engine's queue (vLLM metric)", "")
    style(ax_q)

    for ax in axes:
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.axvline(reqs.scheduled_s.max(), color=MUTED, linewidth=0.8, linestyle=":")
    ax_q.set_xlim(-LEAD_S, x_end)

    fig.suptitle(f"No weight swapping: GPU {gpu} in Prism §7.2 (8 models / 2 GPUs, default trace), "
                 "RTX PRO 6000 96 GB", x=0.07, ha="left", fontsize=13, color=INK, y=0.93)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
