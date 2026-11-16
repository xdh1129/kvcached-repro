"""Sleep + wake time spent in every turn of the two-GPU runs (16:9 slide).

Per run and turn k, on each GPU: the wake times (total_s of the k-th resume of each of the
three models) plus the sleep times, pin time left out (total_s minus the *_save_alloc_s
fields; suspends n >= 1 whose start falls in the turn). Turn k runs from its first request
(requests.jsonl sent_t, over both GPUs) to the first request of turn k + 1 (the last turn:
to workload_end). The line is the mean of the two GPUs, the band their min-max.
The y axis is zoomed to the data.

Output: results/comparison/slides_dual3m_g67/swap_by_turn<suffix>.png
usage: python analysis/plot_dual_swap_by_turn.py [--suffix _x] [run_dir ...]
       (default: dual3m_g67 dram+dram, dram+phoenix, split+phoenix, meet+phoenix)
"""

import argparse
import bisect
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_dual_wake_by_turn import INK2, OUT, ROOT, STYLE, SURFACE  # also sets the slide style

DEFAULT = ["dual3m_g67_dram+dram", "dual3m_g67_dram+phoenix", "dual3m_g67_split+phoenix_f0.20",
           "dual3m_g67_meet+phoenix"]
AGENTS = (1, 2, 3)


def per_turn(d):
    """(path, turns, sleep[turn, gpu], wake[turn, gpu]) in seconds"""
    cfg = dict(l.split("=", 1) for l in open(f"{d}/config.txt").read().splitlines() if "=" in l)
    reqs = [json.loads(l) for l in open(f"{d}/requests.jsonl") if l.strip()]
    turns = sorted({r["turn"] for r in reqs})
    starts = [min(r["sent_t"] for r in reqs if r["turn"] == k) for k in turns]
    gpus = cfg["gpus"].split()
    sleep, wake = np.zeros((len(turns), len(gpus))), np.zeros((len(turns), len(gpus)))
    for j, g in enumerate(gpus):
        for a in AGENTS:
            recs = [json.loads(l) for l in open(f"{d}/tiered_g{g}_a{a}.jsonl")]
            for i, r in enumerate([r for r in recs if r["event"] == "resume"][:len(turns)]):
                wake[i, j] += r["total_s"]
            for r in recs:
                if r["event"] == "suspend" and r["n"] >= 1:
                    i = bisect.bisect_right(starts, r["t_unix"]) - 1
                    if i >= 0:
                        sleep[i, j] += r["total_s"] - sum(v for k, v in r.items() if k.endswith("_save_alloc_s"))
    return cfg["paths"], np.array(turns), sleep, wake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="appended to the output file name")
    ap.add_argument("runs", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    runs = [per_turn(d if os.path.isdir(d) else os.path.join(ROOT, "results", d)) for d in args.runs]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.11)
    lo, hi = float("inf"), 0.0
    for path, turns, sleep, wake in runs:
        label, color, marker = STYLE.get(path, (path, INK2, "o"))
        v = sleep + wake
        ax.fill_between(turns, v.min(axis=1), v.max(axis=1), color=color, alpha=0.18, linewidth=0)
        ax.plot(turns, v.mean(axis=1), color=color, marker=marker, markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
        lo, hi = min(lo, v.min()), max(hi, v.max())
    turns = runs[0][1]
    ax.set_xticks(turns)
    ax.set_xlim(turns[0] - 0.4, turns[-1] + 0.4)
    ax.set_ylim(np.floor(lo - 0.5), np.ceil(hi + 0.5))  # zoomed: y does not start at 0
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("turn")
    ax.set_ylabel("sleep + wake time in the turn, per GPU (s)")
    fig.text(0.04, 0.965, "Sleep + wake time per turn, GPUs 6 and 7 at once", fontsize=26, weight="bold", va="top")
    fig.text(0.04, 0.905, "All sleeps (pin left out) and wakes of the 3 models in the turn. Line: mean of GPUs 6, 7. "
             "y axis does not start at 0", fontsize=18, color=INK2, va="top")
    fig.legend(loc="upper left", bbox_to_anchor=(0.035, 0.87), ncol=4, frameon=False)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"swap_by_turn{args.suffix}.png")
    fig.savefig(out)
    print(out)
    for path, turns, sleep, wake in runs:
        print(f"{path:16s} sleep/turn {np.round(sleep.mean(axis=1), 2).tolist()}")
        print(f"{'':16s} wake/turn  {np.round(wake.mean(axis=1), 2).tolist()}  totals {sleep.mean(axis=1).sum():.1f} + {wake.mean(axis=1).sum():.1f}")


if __name__ == "__main__":
    main()
