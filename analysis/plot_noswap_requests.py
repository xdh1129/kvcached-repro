"""Request serving timeline on both GPUs, Prism 8-model default trace, no swapping.

One panel per GPU on a shared time axis; every request is one line (light =
queued, solid = decoding, gray = never got a first token). Panel heights follow
the number of rows, so lines have the same thickness on both GPUs.

usage: python analysis/plot_noswap_requests.py [--run results/prism_8m2g_noswap_default]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plot_noswap_timeline import INK, LEAD_S, MUTED, SERIES, TAIL_S, request_panel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_BLOCK = 70


def title(ax, text, sub):
    lines = sub.count("\n") + 1
    ax.set_title(text, loc="left", fontsize=11, color=INK, pad=8 + 12 * lines)
    ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=MUTED, va="bottom", ha="left", linespacing=1.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "results", "prism_8m2g_noswap_default"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "comparison", "prism_8m2g_noswap_requests.png"))
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
    x_end = reqs.loc[reqs.ok, "done_s"].max() + TAIL_S
    stopped_s = float(cfg["workload_end"]) - float(cfg["workload_start"])
    gpus = sorted(inst.gpu.unique())

    # rows per GPU, as request_panel lays them out, to size the panels
    rows = []
    for g in gpus:
        names = inst.loc[inst.gpu == g, "name"]
        rows.append(sum(max(int((reqs.model == n).sum()), MIN_BLOCK) + 6 for n in names))
    height = 3 + sum(rows) / 70
    fig, axes = plt.subplots(len(gpus), 1, figsize=(12, height), sharex=True,
                             gridspec_kw={"height_ratios": rows, "hspace": 0.28})
    fig.subplots_adjust(top=1 - 1.5 / height)  # fixed 1.5 in for the figure title, whatever the height

    for prism_idx, (ax, g) in enumerate(zip(axes, gpus)):
        names = list(inst.loc[inst.gpu == g, "name"])
        r = reqs[reqs.model.isin(names)]
        request_panel(ax, r, names, colors, labels, x_end, min_block=MIN_BLOCK)
        busiest = r.model.value_counts().index[0]
        ok = r[(r.model == busiest) & r.ok]
        stuck = [n for n in names if (~r.loc[r.model == n, "ok"]).any()]
        sub = (f"{int(r.ok.sum())} / {len(r)} requests completed; busiest model {busiest}: "
               f"TTFT p50 {ok.ttft_s.median():.2f} s, last done at {ok.done_s.max():.0f} s")
        if stuck:
            sub += f"\n{', '.join(stuck)}: no request ever got a first token (still queued when stopped at " \
                   f"{stopped_s:.0f} s)"
        title(ax, f"GPU {g} (Prism GPU{prism_idx})", sub)
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.axvline(reqs.scheduled_s.max(), color=MUTED, linewidth=0.8, linestyle=":")
    axes[-1].set_xlim(-LEAD_S, x_end)
    axes[-1].set_xlabel("seconds since trace start (dotted: first and last arrival)", color=MUTED, fontsize=9)

    fig.suptitle("Request serving timeline without weight swapping — Prism §7.2 (8 models / 2 GPUs, "
                 "default trace)", x=0.07, ha="left", fontsize=13, color=INK, y=1 - 0.3 / height)
    fig.text(0.07, 1 - 0.6 / height, "one line per request: light = queued (waiting for KV memory) · solid = decoding · "
             "gray = never got a first token", fontsize=8.5, color=MUTED)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
