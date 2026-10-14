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
    """(trace records, startup records, TTFTs), all for the instances on HOT_GPU only.

    Startup records are the sleep each instance does right after launch, before the
    trace; that is where the weights are written to their store, once.
    """
    d = os.path.join(RES, PREFIX + cfg)
    swaps = [json.loads(l) for l in open(os.path.join(d, "swaps.jsonl"))]
    t0 = min(s["t_unix"] for s in swaps)
    with open(os.path.join(d, "instances.tsv")) as f:
        hot = {name for name, _, gpu, _ in csv.reader(f, delimiter="\t") if int(gpu) == HOT_GPU}
    trace, startup = [], []
    for name in hot:
        for line in open(os.path.join(d, f"tiered_{name}.jsonl")):
            r = dict(json.loads(line), inst=name)
            (trace if r["t_unix"] >= t0 - 1 else startup).append(r)
    reqs = [json.loads(l) for l in open(os.path.join(d, "requests.jsonl"))]
    ttft = np.array(sorted(r["ttft_s"] for r in reqs if r["ok"] and r["model"] in hot))
    return trace, startup, ttft


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
            res = [r for r in DATA[c][0] if r["event"] == "resume"]  # GPU 1 wakes
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
             "GPU 1 (3x Mistral-Nemo-12B, 25 GB weights each, + Llama-3.1-8B, 17 GB). One point per wake during "
             "the trace. Lines: least-squares fits, labeled with the KV path's bandwidth. Recompute (gray): no KV.")
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
STORAGE = {"ssd", "gds"}


def traffic(cfg):
    """GB on GPU 1: through host DRAM (sleep/wake, weights+state / KV) and SSD I/O, plus the startup save."""
    trace, startup, _ = DATA[cfg]
    t = dict(startup_dram=0.0, startup_ssd=0.0, sleep_weights=0.0, sleep_kv=0.0, wake_weights=0.0,
             wake_kv=0.0, ssd=0.0)
    for r in startup:  # the one-time weight save (+ state) right after launch
        moved = r.get("weights_save_host_bytes", 0) + r["state_save_host_bytes"] + r["kv_save_host_bytes"]
        t["startup_dram"] += moved / GB
        if r["weights_path"] in STORAGE:
            t["startup_ssd"] += (r["weights_bytes"] + r["state_bytes"]) / GB
    for r in trace:
        if r["event"] == "resume":
            t["wake_weights"] += r["weights_host_bytes"] / GB
            t["wake_kv"] += r.get("kv_host_bytes", 0) / GB
            t["ssd"] += (r["weights_bytes"] if r["weights_path"] in STORAGE else 0) / GB
            t["ssd"] += (r.get("kv_bytes", 0) if r["kv_path"] in STORAGE else 0) / GB
        else:
            t["sleep_weights"] += r["state_save_host_bytes"] / GB
            t["sleep_kv"] += r["kv_save_host_bytes"] / GB
            t["ssd"] += (r["state_bytes"] if r["weights_path"] in STORAGE else 0) / GB
            t["ssd"] += (r["kv_bytes"] if r["kv_path"] in STORAGE else 0) / GB
    return t


def fig_dram_traffic():
    from matplotlib.patches import Patch

    cfgs = list(CONFIGS)[::-1]
    tr = {c: traffic(c) for c in cfgs}
    fig, axes = plt.subplots(1, 3, sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1.6, 1.3], wspace=0.2))
    fig.subplots_adjust(left=0.245, right=0.97, top=0.64, bottom=0.12)
    y = np.arange(len(cfgs))
    W_C, KV_C = "#52514e", "#86b6ef"
    hatch = dict(facecolor=SURFACE, edgecolor=W_C, hatch="///", linewidth=1.5)
    # sleep: startup save (hatched) + trace sleeps; wake: trace wakes
    a = axes[0]
    st = np.array([tr[c]["startup_dram"] for c in cfgs])
    w = np.array([tr[c]["sleep_weights"] for c in cfgs])
    kv = np.array([tr[c]["sleep_kv"] for c in cfgs])
    a.barh(y, st, height=0.55, **hatch)
    a.barh(y, w, left=st, height=0.55, color=W_C, edgecolor=SURFACE, linewidth=2)
    a.barh(y, kv, left=st + w, height=0.55, color=KV_C, edgecolor=SURFACE, linewidth=2)
    for yi, s0, a1, b1 in zip(y, st, w, kv):
        a.text(s0 + a1 + b1 + 5, yi, f"{s0 + a1 + b1:.0f}" + (f"  (trace {a1 + b1:.0f})" if s0 else ""),
               va="center", fontsize=15)
    a.set_title("Sleep: GPU → DRAM", loc="left", fontweight="bold", pad=10)
    b = axes[1]
    w = np.array([tr[c]["wake_weights"] for c in cfgs])
    kv = np.array([tr[c]["wake_kv"] for c in cfgs])
    b.barh(y, w, height=0.55, color=W_C, edgecolor=SURFACE, linewidth=2)
    b.barh(y, kv, left=w, height=0.55, color=KV_C, edgecolor=SURFACE, linewidth=2)
    for yi, a1, b1 in zip(y, w, kv):
        b.text(a1 + b1 + 5, yi, f"{a1 + b1:.0f}" + (f"  (KV {b1:.0f})" if b1 >= 1 else ""), va="center", fontsize=15)
    b.set_title("Wake: DRAM → GPU", loc="left", fontweight="bold", pad=10)
    for ax in axes[:2]:
        ax.set_xlim(0, 420)
        ax.set_xlabel("GB")
    c3 = axes[2]
    st = np.array([tr[c]["startup_ssd"] for c in cfgs])
    io = np.array([tr[c]["ssd"] for c in cfgs])
    c3.barh(y, st, height=0.55, **hatch)
    c3.barh(y, io, left=st, height=0.55, color=GRAY, edgecolor=SURFACE, linewidth=2)
    for yi, s0, v in zip(y, st, io):
        c3.text(s0 + v + 6, yi, f"{s0 + v:.0f}", va="center", fontsize=15)
    c3.set_title("SSD I/O", loc="left", fontweight="bold", pad=10)
    c3.set_xlim(0, 520)
    c3.set_xlabel("GB")
    axes[0].set_yticks(y, [CONFIGS[c][0] for c in cfgs])
    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[0].legend(handles=[Patch(label="startup: weights saved once", **hatch),
                            Patch(color=W_C, label="trace: weights + runtime state"),
                            Patch(color=KV_C, label="trace: KV"),
                            Patch(color=GRAY, label="trace: SSD I/O")],
                   loc="lower left", bbox_to_anchor=(0, 1.09), ncol=2, frameon=False, fontsize=15)
    headline(fig, "Weights are saved once, then read back on every wake",
             "GPU 1 (3x Mistral-Nemo-12B + Llama-3.1-8B). A trace sleep writes only KV + ~0.6 GB state; "
             "a wake reads the whole model (17-25 GB). GDS keeps KV out of DRAM; POSIX SSD passes a host buffer.")
    save(fig, "dram_traffic.png")


# 4. TTFT CDF -----------------------------------------------------------------------------
def fig_ttft():
    fig, axes = plt.subplots(1, 2, gridspec_kw=dict(wspace=0.15))
    fig.subplots_adjust(left=0.07, right=0.68, top=0.74, bottom=0.12)
    for c in CONFIGS:
        t = DATA[c][2]
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
    n = len(DATA["dram+dram"][2])
    headline(fig, "DRAM weights + SSD/GDS KV: the shortest TTFT tail",
             f"Requests to the pressured GPU (4 models, {n} requests). Prism Arena trace, time scale 0.2, "
             "replication 2, one run per config.")
    save(fig, "ttft_cdf.png")


if __name__ == "__main__":
    fig_restore_vs_kv()
    fig_restore_breakdown()
    fig_dram_traffic()
    fig_ttft()
