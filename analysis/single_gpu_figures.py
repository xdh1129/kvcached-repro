"""B2-style figure for the single-GPU experiment: the same wake as the history grows.

Rows are turns, columns are paths. Each panel is GPU 7's PCIe RX (NVML) and md0 reads (pcie.csv,
0.1 s samples) around one model's wake, with the plugin's transfer jobs as bars on top.
Staged GDS bars end when the data stopped moving (without the staging cleanup).

Also: sleep + wake per turn for every model (single3m_sleep_wake_by_turn_<model>.png).

usage: python single_gpu_figures.py [out_dir]     (default results/figures)
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from make_figures import GIB, ROOT, SLOTS, TEXT, TEXT2, jsonl

RUNS = [("dram+dram", "single3m_g7_dram+dram"), ("dram+ssd", "single3m_g7_dram+ssd"),
        ("dram+gds", "single3m_g7_dram+gds"), ("split", "single3m_g7_kvonly_split_f0.20")]
MODELS = [("g7_a3", "Phi-3.5", "phi35"), ("g7_a2", "Qwen3-14B", "qwen14b"), ("g7_a1", "Qwen3-32B", "qwen32b")]
TURNS = [5, 10, 15]
JOBS = {False: (("weights", "weights"), ("kv", "KV")), True: (("dram", "DRAM part"), ("ssd", "SSD part"))}
YMAX = 88


def cell(run, inst, turn, tt, rows, t_pre=0.15, t_post=2.6):
    # wake number n brings the model back for turn n + 1
    e = next(r for r in jsonl(os.path.join(ROOT, run, f"tiered_{inst}.jsonl"))
             if r["event"] == "resume" and r["n"] == turn - 1)
    t_ref = e["t_unix"]
    sel = np.flatnonzero((tt >= t_ref - t_pre) & (tt <= t_ref + t_post))
    jobs = []
    for key, label in JOBS["split_frac" in e]:
        end = e.get(f"{key}_data_end_s", e[f"{key}_end_s"])
        start = e.get(f"{key}_data_start_s", e[f"{key}_start_s"])
        jobs.append((label, start, end))
    return {"t": tt[sel] - t_ref, "rx": [float(rows[i]["gpu7_rx_mbs"]) / 1000 for i in sel],
            "md0": [float(rows[i]["md0_read_mbs"]) / 1000 for i in sel], "jobs": jobs,
            "total_s": e["total_s"], "kv_gib": e["kv_bytes"] / GIB, "weights_gib": e["weights_bytes"] / GIB}


def panel(ax, c, xmax):
    ax.fill_between(c["t"], c["rx"], color=SLOTS[0], alpha=.85, lw=0, label="GPU7 PCIe RX")
    ax.plot(c["t"], c["md0"], color=SLOTS[1], lw=2, ls="--", label="SSD read (md0)")
    for k, (label, a, b) in enumerate(c["jobs"]):
        y = YMAX * (0.95 - 0.08 * k)
        ax.plot([a, b], [y, y], color=TEXT2, lw=4, solid_capstyle="butt")
        ax.text(b + xmax * 0.015, y, label, va="center", fontsize=7.5, color=TEXT2)
    ax.set_xlim(-0.1, xmax)
    ax.set_ylim(0, YMAX)
    ax.grid(axis="x", visible=False)


def figure(inst, model, slug, out_dir):
    data = {}
    for path, run in RUNS:
        rows = list(csv.DictReader(open(os.path.join(ROOT, run, "pcie.csv"))))
        tt = np.array([float(r["t"]) for r in rows])
        for turn in TURNS:
            data[path, turn] = cell(run, inst, turn, tt, rows)
    xmax = max(c["total_s"] for c in data.values()) + 0.35
    fig, axes = plt.subplots(len(TURNS), len(RUNS), figsize=(15.5, 7.8), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.1, "hspace": 0.32})
    for i, turn in enumerate(TURNS):
        for j, (path, _) in enumerate(RUNS):
            c = data[path, turn]
            panel(axes[i, j], c, xmax)
            axes[i, j].set_title(f"{path} · turn {turn} · KV {c['kv_gib']:.1f} GiB · {c['total_s']:.2f} s",
                                 loc="left", fontsize=8.5, color=TEXT)
        axes[i, 0].set_ylabel("GB/s")
    for ax in axes[-1]:
        ax.set_xlabel("seconds since the wake started")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.78))
    fig.suptitle(f"The same wake as the history grows ({model}, {data['dram+dram', 15]['weights_gib']:.1f} GiB weights; "
                 "one GPU; split f = 0.20, SSD holds KV only)", x=0.08, ha="left", fontsize=11, color=TEXT, y=0.95)
    path = os.path.join(out_dir, f"single3m_B2_{slug}.png")
    fig.savefig(path)
    plt.close(fig)
    print(path)
    for (p, t), c in data.items():
        print(f"  {p:10s} turn {t:2d} KV {c['kv_gib']:5.1f} wake {c['total_s']:.2f} s  "
              + "  ".join(f"{l} {a:.2f}-{b:.2f}" for l, a, b in c["jobs"]))


def round_trips(run, inst):
    """{turn: sleep + wake s}: the sleep before a turn's wake moves the same KV as that wake. Pin time excluded."""
    recs = jsonl(os.path.join(ROOT, run, f"tiered_{inst}.jsonl"))
    res = {r["n"]: r for r in recs if r["event"] == "resume"}
    return {r["n"] + 1: r["total_s"] - sum(v for k, v in r.items() if k.endswith("_save_alloc_s")) + res[r["n"]]["total_s"]
            for r in recs if r["event"] == "suspend" and r["n"] >= 1 and r["n"] in res}


def round_trip_figure(inst, model, slug, out_dir):
    colors = {"dram+dram": SLOTS[0], "dram+ssd": SLOTS[1], "dram+gds": SLOTS[2], "split": SLOTS[6]}
    markers = {"dram+dram": "o", "dram+ssd": "s", "dram+gds": "^", "split": "D"}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for path, run in RUNS:
        d = round_trips(run, inst)
        turns = sorted(d)
        ax.plot(turns, [d[t] for t in turns], color=colors[path], marker=markers[path], ms=6, lw=2,
                mec="white", mew=.8, label=path, zorder=4 if path == "split" else 3)
    ax.set_title(f"{model}: sleep + wake of one model swap as the KV grows (one GPU, single run)",
                 loc="left", fontsize=10.5, color=TEXT)
    ax.set_xlabel("turn (the sleep before it + its wake)")
    ax.set_ylabel("sleep + wake, seconds (excl. pin)")
    ax.set_xticks([2, 5, 10, 15])
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, frameon=False, loc="lower right", ncol=2)
    path = os.path.join(out_dir, f"single3m_sleep_wake_by_turn_{slug}.png")
    fig.savefig(path)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figures")
    for m in MODELS:
        figure(*m, out)
        round_trip_figure(*m, out)
