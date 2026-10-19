"""Slide figures (16:9, English) for the Prism 18-model replays on GPUs 6+7.

Runs: results/prism_18m_2g67_ts0.2_rep2_sched_<config>, all replaying the schedule of
prism_18m_2g67_ts0.2_rep2_ref_dram+ssd. Figures show GPU 6, where every sleep happens
(GPU 7 only wakes its models once at the start). GDS runs use TIERED_GDS_STAGE_MB=256.

Encoding (as analysis/plot_paths_slides.py):
  color      where the KV goes on sleep: dropped (gray), DRAM (blue), SSD (orange), GDS (aqua)
  line/mark  where the weights go: DRAM = solid / circle; SSD or GDS = dashed / triangle
Every series is also labeled, so color never carries identity alone.

Outputs: results/comparison/slides_18m_g67/{wake_time,ttft_cdf,data_moved}.png
usage: python analysis/plot_18m_g67_slides.py
"""

import csv
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
PREFIX = "prism_18m_2g67_ts0.2_rep2_sched_"
OUT = os.path.join(RES, "comparison", "slides_18m_g67")
HOT_GPU = 6
GB, GIB = 1e9, 1 << 30
STORAGE = {"ssd", "gds"}

GRAY, BLUE, ORANGE, AQUA = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# config -> (label "weights / KV", KV color, weights in storage?)
CONFIGS = {
    "vllm": ("vLLM sleep: DRAM / dropped", GRAY, False),
    "dram+dram": ("DRAM / DRAM", BLUE, False),
    "dram+ssd": ("DRAM / SSD", ORANGE, False),
    "dram+gds_stage256": ("DRAM / GDS", AQUA, False),
    "ssd+ssd": ("SSD / SSD", ORANGE, True),
    "gds+gds_stage256": ("GDS / GDS", AQUA, True),
}
GPU6_MODELS = "3x Phi-3-mini, phi-2, PowerLM-3B, 4x Llama-3.2-1B/3B"

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
    with open(os.path.join(d, "instances.tsv")) as f:
        hot = {name for name, _, gpu, _ in csv.reader(f, delimiter="\t") if int(gpu) == HOT_GPU}
    tiered = []  # trace sleeps/wakes only (the save right after launch is before t0)
    for name in hot:
        p = os.path.join(d, f"tiered_{name}.jsonl")
        if os.path.exists(p):
            tiered += [dict(r, inst=name) for r in map(json.loads, open(p)) if r["t_unix"] >= t0 - 1]
    reqs = [json.loads(l) for l in open(os.path.join(d, "requests.jsonl"))]
    summary = json.load(open(os.path.join(d, "summary.json")))
    return {"swaps": [s for s in swaps if s["gpu"] == HOT_GPU], "tiered": tiered, "hot": hot, "summary": summary,
            "ttft": np.array(sorted(r["ttft_s"] for r in reqs if r["ok"] and r["model"] in hot))}


DATA = {c: load(c) for c in CONFIGS}


def headline(fig, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=28, fontweight="bold", va="top")
    fig.text(0.04, 0.885, textwrap.fill(subtitle, 110), fontsize=18, color=INK2, va="top", linespacing=1.3)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# 1. Wake time ---------------------------------------------------------------------------
def fig_wake_time():
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.25, right=0.80, top=0.69, bottom=0.12)
    rows = list(CONFIGS)[::-1]
    rng = np.random.default_rng(0)
    for i, c in enumerate(rows):
        w = np.array([s["duration_s"] for s in DATA[c]["swaps"] if s["action"] == "wake"])
        st = style(c)
        ax.scatter(w, i + rng.uniform(-0.18, 0.18, len(w)), s=80, color=st["color"], marker=st["marker"],
                   edgecolors=SURFACE, linewidths=1.5, alpha=0.9, zorder=3)
        p50, p95 = np.percentile(w, 50), np.percentile(w, 95)
        ax.plot([p50, p50], [i - 0.32, i + 0.32], color=INK, linewidth=3, zorder=4)
        ax.plot([p95, p95], [i - 0.32, i + 0.32], color=INK, linewidth=2, linestyle=":", zorder=4)
        ax.text(1.02, i, f"p50 {p50:.2f} s   p95 {p95:.2f} s", transform=ax.get_yaxis_transform(),
                va="center", fontsize=17, color=INK)
    ax.set_yticks(range(len(rows)), [CONFIGS[c][0] for c in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, None)
    ax.set_xlabel("Wake time on GPU 6 (s): remap + restore of weights and KV")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=INK, linewidth=3, label="median"),
                       Line2D([], [], color=INK, linewidth=2, linestyle=":", label="p95"),
                       Line2D([], [], color=INK2, marker="o", linestyle="", markersize=10, label="one wake")],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=16)
    n = len([s for s in DATA["dram+dram"]["swaps"] if s["action"] == "wake"])
    headline(fig, "DRAM weights keep wakes short; SSD or GDS weights add ~0.7 s",
             f"Labels: where the weights / the KV go on sleep. {n} wakes on GPU 6 ({GPU6_MODELS}), same schedule "
             "for every config. vLLM sleep drops the KV, so its wakes move only weights. GDS with 256 MiB staging.")
    save(fig, "wake_time.png")


# 2. TTFT CDF ----------------------------------------------------------------------------
def fig_ttft():
    fig, axes = plt.subplots(1, 2, gridspec_kw=dict(wspace=0.22))
    fig.subplots_adjust(left=0.07, right=0.68, top=0.74, bottom=0.12)
    for c in CONFIGS:
        t = DATA[c]["ttft"]
        q = np.arange(1, len(t) + 1) / len(t)
        st = style(c)
        for ax in axes:
            ax.step(t, q, where="post", color=st["color"], linestyle=st["linestyle"], linewidth=2.5,
                    label=f"{CONFIGS[c][0]}\np99 TTFT {np.percentile(t, 99):.1f} s")
    ts = [DATA[c]["ttft"] for c in CONFIGS]
    hi = max(t[-1] for t in ts) * 1.3
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Time to first token (s, log scale)")
    axes[0].set_xlim(min(t[0] for t in ts) * 0.8, hi)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Fraction of requests")
    axes[0].set_title("All requests", loc="left", fontweight="bold", pad=12)
    axes[1].set_ylim(0.95, 1.0)
    axes[1].set_xlim(min(np.percentile(t, 95) for t in ts) * 0.85, hi)
    axes[1].set_title("Tail (top 5%)", loc="left", fontweight="bold", pad=12)
    for ax in axes:
        lo, hi_ = ax.get_xlim()
        ticks = [v for v in (0.1, 0.2, 0.5, 1, 2, 4, 5, 6, 8, 10, 20, 50) if lo <= v <= hi_ and (v not in (4, 6, 8) or hi_ / lo < 5)]
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    axes[1].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=16, labelspacing=0.9)
    n = len(DATA["dram+dram"]["ttft"])
    headline(fig, "Same median for every path; only the tail separates them",
             f"Requests to GPU 6 ({n} of 4118). Prism 18-model trace (Arena-Chat), time scale 0.2, replication 2, "
             "one run per config.")
    save(fig, "ttft_cdf.png")


# 3. Data moved --------------------------------------------------------------------------
def moved(cfg):
    """GB on GPU 6 during the trace: GPU -> DRAM and DRAM -> GPU (weights / KV), SSD reads and writes."""
    m = dict(sleep_w=0.0, sleep_kv=0.0, wake_w=0.0, wake_kv=0.0, ssd_write=0.0, ssd_read=0.0,
             sleep_w_t=0.0, sleep_kv_t=0.0, wake_w_t=0.0, wake_kv_t=0.0)  # *_t: through a POSIX bounce buffer
    d = DATA[cfg]
    if cfg == "vllm":  # no plugin logs: vLLM copies the weights pool to DRAM and back on every sleep/wake
        weights = {n: d["summary"]["instances"][n]["weights_gib"] * GIB / GB for n in d["hot"]}
        for s in d["swaps"]:
            key = "sleep_w" if s["action"] == "sleep" else "wake_w"
            m[key] += weights[s["instance"]]
        return m
    for r in d["tiered"]:
        if r["event"] == "suspend":
            m["sleep_w_t" if r["weights_path"] == "ssd" else "sleep_w"] += (
                r.get("weights_save_host_bytes", 0) + r["state_save_host_bytes"]) / GB
            m["sleep_kv_t" if r["kv_path"] == "ssd" else "sleep_kv"] += r["kv_save_host_bytes"] / GB
            if r["weights_path"] in STORAGE and r.get("weights_saved"):
                m["ssd_write"] += r["weights_bytes"] / GB
            if r["kv_path"] in STORAGE:
                m["ssd_write"] += r["kv_bytes"] / GB
        else:
            m["wake_w_t" if r["weights_path"] == "ssd" else "wake_w"] += r["weights_host_bytes"] / GB
            m["wake_kv_t" if r["kv_path"] == "ssd" else "wake_kv"] += r.get("kv_host_bytes", 0) / GB
            if r["weights_path"] in STORAGE:
                m["ssd_read"] += r["weights_bytes"] / GB
            if r["kv_path"] in STORAGE:
                m["ssd_read"] += r.get("kv_bytes", 0) / GB
    return m


def fig_data_moved():
    cfgs = list(CONFIGS)[::-1]
    mv = {c: moved(c) for c in cfgs}
    fig, axes = plt.subplots(1, 3, sharey=True, gridspec_kw=dict(wspace=0.2))
    fig.subplots_adjust(left=0.235, right=0.97, top=0.62, bottom=0.12)
    y = np.arange(len(cfgs))
    W_C, KV_C, R_C, WR_C = "#52514e", "#86b6ef", GRAY, "#c3c2b7"
    hatched = lambda col: dict(facecolor=SURFACE, edgecolor=col, hatch="///", linewidth=1.5)
    solid = lambda col: dict(color=col, edgecolor=SURFACE, linewidth=2)
    # (title, [(key, bar style)]) stacked left to right
    panels = [("Sleep: GPU → DRAM", [("sleep_w", solid(W_C)), ("sleep_w_t", hatched(W_C)),
                                     ("sleep_kv", solid(KV_C)), ("sleep_kv_t", hatched(KV_C))]),
              ("Wake: DRAM → GPU", [("wake_w", solid(W_C)), ("wake_w_t", hatched(W_C)),
                                    ("wake_kv", solid(KV_C)), ("wake_kv_t", hatched(KV_C))]),
              ("SSD I/O", [("ssd_write", solid(WR_C)), ("ssd_read", solid(R_C))])]
    xmax = max(max(sum(m[k] for k, _ in segs) for m in mv.values()) for _, segs in panels) * 1.32
    for ax, (title, segs) in zip(axes, panels):
        left = np.zeros(len(cfgs))
        for key, kw in segs:
            v = np.array([mv[c][key] for c in cfgs])
            nz = v > 0.05  # a zero-width bar would still draw its edge
            ax.barh(y[nz], v[nz], left=left[nz], height=0.55, **kw)
            left += v
        for yi, tot in zip(y, left):
            ax.text(tot + xmax * 0.02, yi, f"{tot:.0f}" if tot >= 0.5 else "0", va="center", fontsize=16)
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("GB")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [CONFIGS[c][0] for c in cfgs])
    axes[0].legend(handles=[Patch(color=W_C, label="weights (+ runtime state)"), Patch(color=KV_C, label="KV"),
                            Patch(label="hatched: passing through a POSIX bounce buffer", **hatched(INK2)),
                            Patch(color=WR_C, label="SSD write"), Patch(color=R_C, label="SSD read")],
                   loc="lower left", bbox_to_anchor=(0, 1.09), ncol=3, frameon=False, fontsize=16)
    headline(fig, "Every sleep re-saves the weights; GDS keeps all of it out of DRAM",
             f"GB moved for GPU 6 during the trace (startup save excluded). vLLM's default re-saves the weights on "
             "every sleep (vLLM row: weights pool size x sleeps/wakes, no plugin log). POSIX SSD I/O also crosses DRAM; GDS does not.")
    save(fig, "data_moved.png")


if __name__ == "__main__":
    fig_wake_time()
    fig_ttft()
    fig_data_moved()
