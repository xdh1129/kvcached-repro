"""Slide figures (16:9, English) for the six offload-path configs on the Prism trace.

Encoding, shared by every figure:
  color      where the KV goes on sleep: recompute (gray), DRAM (blue), SSD (orange), GDS (aqua)
  line/mark  where the weights come from: DRAM = solid line / circle; SSD or GDS = dashed / triangle
Every series is also labeled directly or in a legend, so color never carries identity alone.

Inputs: results/prism_8m2g_rep2_kvp_<config>/ (one run per config, replication 2).
Outputs: results/comparison/slides/{restore_vs_kv,restore_breakdown,dram_traffic,ttft_cdf}.png

usage: python analysis/plot_paths_slides.py
"""

import csv
import glob
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "comparison", "slides")
PREFIX = "prism_8m2g_rep2_kvp_"
HOT_GPU = 1  # our GPU 1 = Prism's GPU0, the pressured one
GB = 1e9

# config -> (label "weights / KV", KV color, weights from storage?)
GRAY, BLUE, ORANGE, AQUA = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a"
CONFIGS = {
    "dram+recompute": ("DRAM / recompute (Prism)", GRAY, False),
    "dram+dram": ("DRAM / DRAM", BLUE, False),
    "ssd+ssd": ("SSD / SSD", ORANGE, True),
    "gds+gds": ("GDS / GDS", AQUA, True),
    "dram+ssd": ("DRAM / SSD", ORANGE, False),
    "dram+gds": ("DRAM / GDS", AQUA, False),
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 150,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 20, "xtick.labelsize": 18,
    "ytick.labelsize": 18, "legend.fontsize": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "lines.linewidth": 2.5,
})


def style(cfg):
    _, color, storage = CONFIGS[cfg]
    return dict(color=color, linestyle="--" if storage else "-", marker="^" if storage else "o")


def load(cfg):
    d = os.path.join(RES, PREFIX + cfg)
    swaps = [json.loads(l) for l in open(os.path.join(d, "swaps.jsonl"))]
    t0 = min(s["t_unix"] for s in swaps)
    recs = []
    for f in glob.glob(os.path.join(d, "tiered_*.jsonl")):
        name = os.path.basename(f)[len("tiered_"):-len(".jsonl")]
        for line in open(f):
            r = json.loads(line)
            if r["t_unix"] >= t0 - 1:
                recs.append(dict(r, inst=name))
    with open(os.path.join(d, "instances.tsv")) as f:
        hot = {name for name, _, gpu, _ in csv.reader(f, delimiter="\t") if int(gpu) == HOT_GPU}
    reqs = [json.loads(l) for l in open(os.path.join(d, "requests.jsonl"))]
    ttft = np.array(sorted(r["ttft_s"] for r in reqs if r["ok"] and r["model"] in hot))
    return recs, ttft


DATA = {c: load(c) for c in CONFIGS}


def restore_s(r):
    return max(r["weights_end_s"], r.get("kv_end_s", 0))


def headline(fig, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=28, fontweight="bold", va="top")
    fig.text(0.04, 0.885, textwrap.fill(subtitle, 108), fontsize=18, color=INK2, va="top", linespacing=1.3)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# 1. Restore time vs KV size ---------------------------------------------------------------
def fig_restore_vs_kv():
    fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.08))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.73, bottom=0.12)
    panels = [("Weights restored from DRAM", ["dram+dram", "dram+ssd", "dram+gds", "dram+recompute"]),
              ("Weights restored from SSD (POSIX or GDS)", ["ssd+ssd", "gds+gds"])]
    for ax, (title, cfgs) in zip(axes, panels):
        for c in cfgs:
            res = [r for r in DATA[c][0] if r["event"] == "resume"]
            x = np.array([r.get("kv_bytes", 0) / GB for r in res])
            y = np.array([restore_s(r) for r in res])
            st = style(c)
            ax.scatter(x, y, s=150 if c == "dram+recompute" else 110, color=st["color"], marker=st["marker"],
                       edgecolors=SURFACE, linewidths=2, zorder=4 if c == "dram+recompute" else 3,
                       label=CONFIGS[c][0])
            if c != "dram+recompute" and np.ptp(x) > 1:
                k, b = np.polyfit(x, y, 1)
                xs = np.array([0, 12])
                ax.plot(xs, k * xs + b, color=st["color"], linestyle=st["linestyle"], linewidth=2, zorder=2)
                ax.annotate(f"{CONFIGS[c][0]}\n{1 / k:.0f} GB/s KV", (12, k * 12 + b), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=16, color=INK)
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_xlim(-0.3, 12)
        ax.set_xlabel("KV restored with the model (GB)")
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("Restore time (s)")
    axes[0].set_ylim(0, None)
    headline(fig, "Restore time = weights floor + KV size / KV-path bandwidth",
             "One point per wake during the trace (weights: Nemo-12B 25 GB, Llama-8B 17 GB). Lines: least-squares "
             "fits, labeled with the KV path's effective bandwidth. Recompute (gray) restores no KV.")
    for ax in axes:
        ax.margins(x=0)
    axes[1].set_xlim(-0.3, 15.5)
    axes[0].set_xlim(-0.3, 15.5)
    save(fig, "restore_vs_kv.png")


# 2. One restore, broken down -------------------------------------------------------------
def fig_restore_breakdown():
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.24, right=0.97, top=0.70, bottom=0.12)
    rows = list(CONFIGS)[::-1]
    for i, c in enumerate(rows):
        res = [r for r in DATA[c][0] if r["event"] == "resume" and r["inst"] == "model_1"]
        if c == "dram+recompute":
            r = max(res, key=lambda r: r["t_unix"])
        else:  # the model_1 wake whose KV is closest to 8 GB
            r = min(res, key=lambda r: abs(r.get("kv_bytes", 0) / GB - 8))
        _, color, storage = CONFIGS[c]
        h = 0.34
        ax.barh(i + 0.19, r["map_s"], left=0, height=h, color=GRID)
        ax.barh(i + 0.19, r["weights_end_s"] - r["weights_start_s"], left=r["weights_start_s"], height=h,
                color="#52514e" if not storage else color, hatch="//" if storage else None,
                edgecolor=SURFACE, linewidth=0)
        if "kv_end_s" in r:
            ax.barh(i - 0.19, r["kv_end_s"] - r["kv_start_s"], left=r["kv_start_s"], height=h, color=color)
            seq = r["map_s"] + (r["weights_end_s"] - r["weights_start_s"]) + (r["kv_end_s"] - r["kv_start_s"])
            ax.plot([seq, seq], [i - 0.42, i + 0.42], color=INK, linewidth=2, linestyle=":")
            note = f"{restore_s(r):.2f} s  (serial {seq:.2f} s), KV {r['kv_bytes'] / GB:.1f} GB"
        else:
            note = f"{restore_s(r):.2f} s  (KV dropped, re-prefilled later)"
        ax.text(max(restore_s(r), seq if "kv_end_s" in r else 0) + 0.1, i, note, va="center", fontsize=16,
                color=INK)
    ax.set_yticks(range(len(rows)), [CONFIGS[c][0] for c in rows])
    ax.set_xlabel("Seconds from the start of the wake")
    ax.set_xlim(0, 9.5)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=GRID, label="remap"),
                       Patch(color="#52514e", label="weights from DRAM"),
                       Patch(color=BLUE, label="KV (color = its path)"),
                       Line2D([], [], color=INK, linestyle=":", label="serial: one after the other")],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, frameon=False, fontsize=15)
    headline(fig, "Two paths at once: restore time is the longer path, not the sum",
             "One wake of model_1 (Mistral-Nemo-12B, 25 GB weights) per config: the one whose KV is closest to "
             "8 GB. Upper bar = weights (hatched when from storage), lower bar = KV.")
    save(fig, "restore_breakdown.png")


# 3. DRAM traffic -------------------------------------------------------------------------
def dram_split(cfg):
    """GB through host DRAM during the trace, split by direction and by weights(+state) / KV."""
    out = {"sleep_weights": 0, "sleep_kv": 0, "wake_weights": 0, "wake_kv": 0}
    for r in DATA[cfg][0]:
        if r["event"] == "resume":
            out["wake_weights"] += r["weights_host_bytes"] / GB
            out["wake_kv"] += r.get("kv_host_bytes", 0) / GB
        else:
            out["sleep_weights"] += r["state_save_host_bytes"] / GB
            out["sleep_kv"] += r["kv_save_host_bytes"] / GB
    return out


def fig_dram_traffic():
    rows = json.load(open(os.path.join(RES, "comparison", "paths_rep2.json")))
    cfgs = list(CONFIGS)[::-1]
    split = {c: dram_split(c) for c in cfgs}
    fig, axes = plt.subplots(1, 3, sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1.6, 1], wspace=0.2))
    fig.subplots_adjust(left=0.245, right=0.97, top=0.68, bottom=0.12)
    y = np.arange(len(cfgs))
    W_C, KV_C = "#52514e", "#86b6ef"
    for ax, key, title in ((axes[0], "sleep", "Sleep: GPU → DRAM"), (axes[1], "wake", "Wake: DRAM → GPU")):
        w = np.array([split[c][key + "_weights"] for c in cfgs])
        kv = np.array([split[c][key + "_kv"] for c in cfgs])
        ax.barh(y, w, height=0.55, color=W_C, edgecolor=SURFACE, linewidth=2,
                label="weights + runtime state" if key == "wake" else "runtime state (weights are saved once, at startup)")
        ax.barh(y, kv, left=w, height=0.55, color=KV_C, edgecolor=SURFACE, linewidth=2, label="KV")
        for yi, a, b in zip(y, w, kv):
            ax.text(a + b + 5, yi, f"{a + b:.0f}" + (f"  (KV {b:.0f})" if b >= 1 else ""), va="center", fontsize=15)
        ax.set_xlim(0, 470)
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_xlabel("GB")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=W_C, label="weights + runtime state (weights are written to DRAM once, at startup)"),
                            Patch(color=KV_C, label="KV")],
                   loc="lower left", bbox_to_anchor=(0, 1.09), ncol=2, frameon=False, fontsize=15)
    b = axes[2]
    ssd = [rows[c]["storage_io_gb"] for c in cfgs]
    b.barh(y, ssd, height=0.55, color=GRAY)
    for yi, v in zip(y, ssd):
        b.text(v + 6, yi, f"{v:.0f}", va="center", fontsize=15)
    b.set_title("SSD I/O", loc="left", fontweight="bold", pad=10)
    b.set_xlabel("GB")
    b.set_xlim(0, 520)
    axes[0].set_yticks(y, [CONFIGS[c][0] for c in cfgs])
    for ax in axes:
        ax.grid(axis="y", visible=False)
    headline(fig, "Weights dominate DRAM traffic; GDS keeps KV out of DRAM",
             "Trace totals (startup weight saves excluded). Each wake reads the whole model (17-25 GB); a sleep "
             "writes only KV + ~0.6 GB state. POSIX SSD I/O also passes a host bounce buffer.")
    save(fig, "dram_traffic.png")


# 4. TTFT CDF -----------------------------------------------------------------------------
def fig_ttft():
    fig, axes = plt.subplots(1, 2, gridspec_kw=dict(wspace=0.15))
    fig.subplots_adjust(left=0.07, right=0.68, top=0.74, bottom=0.12)
    for c in CONFIGS:
        t = DATA[c][1]
        q = np.arange(1, len(t) + 1) / len(t)
        st = style(c)
        for ax in axes:
            ax.step(t, q, where="post", color=st["color"], linestyle=st["linestyle"], linewidth=2.5,
                    label=f"{CONFIGS[c][0]}\np99 TTFT {np.percentile(t, 99):.1f} s")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Time to first token (s, log scale)")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Fraction of requests")
    axes[0].set_title("All requests", loc="left", fontweight="bold", pad=12)
    axes[1].set_ylim(0.95, 1.0)
    axes[1].set_xlim(0.3, 60)
    axes[1].set_title("Tail (top 5%)", loc="left", fontweight="bold", pad=12)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=16, labelspacing=0.9)
    n = len(DATA["dram+dram"][1])
    headline(fig, "DRAM weights + SSD/GDS KV: the shortest TTFT tail",
             f"Requests to the pressured GPU (4 models, {n} requests). Prism Arena trace, time scale 0.2, "
             "replication 2, one run per config.")
    save(fig, "ttft_cdf.png")


if __name__ == "__main__":
    fig_restore_vs_kv()
    fig_restore_breakdown()
    fig_dram_traffic()
    fig_ttft()
