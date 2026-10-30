"""Figures for the GPU 6+7 data-movement experiments (agent round-robin + wake storm).

  A1  GPU time by state per offload path            (agent_rr_g67_<paths>)
  A2  arrival -> first token per turn                (agent_rr_g67_<paths>)
  A3  wake time per turn, one panel per model        (agent_rr_g67_<paths>)
  A4  one model switch on both GPUs, per data source (agent_rr_g67_dram+gds)
  A5  turn-24 arrival -> first token split into sleep / wake / prefill, one model (Phi-3.5)
  W2  aggregate rate per data source vs ceilings     (wake_storm_g67_kv5_d0_<paths>)
  W3  wake transfer time vs KV/weights ratio: model + measurements
  B1  PCIe / SSD throughput during one wake of Phi-3.5 (12 GiB KV) on both GPUs, per path
  B2  the same wake at turns 4, 12, 24 (KV grows with the history)
  O1  GPU time by state when GPU 7 sends each step's request 0 / 0.3 / 1 s after GPU 6
  O2  arrival -> first token per turn for those offsets
  O3  PCIe RX / TX of both GPUs during one step, offset 0 vs 0.3 s (full duplex)

Writes results/figures/<name>.png, figures_data.json, and index.html (analysis/figures_page.html
with the data inlined: the interactive version of the same figures).

usage: python analysis/make_figures.py
"""

import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(ROOT, "figures")
GIB = 1 << 30
PATHS = ["dram+dram", "dram+ssd", "dram+gds", "ssd+ssd", "gds+gds", "vllm"]
# categorical slots in fixed order (dataviz reference palette, light mode)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PATH_COLOR = dict(zip(PATHS, SLOTS))
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# measured ceilings: GPU 6+7 shared host uplink, RAID0 (2 NVMe) sequential read
UPLINK_GIBS, RAID_GIBS = 53.7, 12.4
WS_RUN = {p: f"wake_storm_g67_kv5_d0_{p}" + ("_stage256" if "gds" in p else "") for p in PATHS}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2,
                     "ytick.color": TEXT2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def tiered_events(run):
    ev = {}
    for f in glob.glob(os.path.join(ROOT, run, "tiered_*.jsonl")):
        ev[os.path.basename(f)[len("tiered_"):-len(".jsonl")]] = jsonl(f)
    return ev


def agent(p):
    return json.load(open(os.path.join(ROOT, f"agent_rr_g67_{p}", "summary.json")))


# ---------------------------------------------------------------- A1
def a1_data():
    rows = []
    for p in PATHS:
        s = agent(p)
        fr = {k: [] for k in ("prefill", "decode", "load", "offload", "other")}
        for g, st in s["gpus"].items():
            pre = sum(r["prefill_s"] for r in s["requests"] if str(r["gpu"]) == g) / st["window_s"]
            fr["prefill"].append(pre)
            fr["decode"].append(st["compute"] - pre)
            fr["load"].append(st["load"])
            fr["offload"].append(st["offload"])
            fr["other"].append(st["sync"] + st["other"])
        rows.append({"paths": p, "window_s": float(np.mean([st["window_s"] for st in s["gpus"].values()])),
                     **{k: float(np.mean(v)) for k, v in fr.items()}})
    return rows


A1_STATES = [("decode", "Compute: decode"), ("prefill", "Compute: prefill (vllm: re-prefill of history)"),
             ("load", "Wait: load (wake)"), ("offload", "Wait: offload (sleep LRU model)"),
             ("other", "Other (step sync, proxy)")]


def a1_plot(rows):
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for i, (k, label) in enumerate(A1_STATES):
        v = np.array([r[k] * 100 for r in rows])
        ax.barh(y, v, left=left, height=0.62, color=SLOTS[i], label=label, edgecolor="white", linewidth=1.5)
        for yy, l, vv in zip(y, left, v):
            if vv >= 6:
                ax.text(l + vv / 2, yy, f"{vv:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        left += v
    for yy, r in zip(y, rows):
        ax.text(101, yy, f"{r['window_s']:.0f} s", va="center", fontsize=9, color=TEXT2)
    ax.text(101, len(rows) - 0.45, "makespan", fontsize=8.5, color=TEXT2)
    ax.set_yticks(y, [r["paths"] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of GPU time (mean of GPU 6 and 7), %")
    ax.grid(axis="y", visible=False)
    ax.set_title("A1. Where GPU time goes: 8 agents per GPU, 24 turns, LRU swap of weights + KV",
                 loc="left", fontsize=11, color=TEXT, pad=16)
    ax.legend(ncol=2, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17))
    fig.savefig(os.path.join(OUT, "A1_gpu_time_by_state.png"))
    plt.close(fig)


# ---------------------------------------------------------------- A2
def a2_data():
    out = {"turns": list(range(1, 25)), "history_k": [], "series": {}}
    hist = {}
    for p in PATHS:
        reqs = agent(p)["requests"]
        med, parts = [], {"offload": [], "load": [], "prefill": []}
        for t in out["turns"]:
            rs = [r for r in reqs if r["turn"] == t]
            med.append(float(np.median([r["wait_s"] + r["prefill_s"] for r in rs])))
            for k in parts:
                parts[k].append(float(np.median([r[k + "_s"] for r in rs])))
            hist.setdefault(t, []).extend(r["history_len"] for r in rs)
        out["series"][p] = {"ttft": med, **parts}
    out["history_k"] = [float(np.median(hist[t])) / 1000 for t in out["turns"]]
    return out


def a2_plot(d):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    t = d["turns"]
    ends = sorted(((d["series"][p]["ttft"][-1], p) for p in PATHS))
    label_y, prev = {}, -1.0
    for y, p in ends:  # end labels at least 0.26 s apart
        prev = label_y[p] = max(y, prev + 0.26)
    for p in PATHS:
        ys = d["series"][p]["ttft"]
        ax.plot(t, ys, color=PATH_COLOR[p], lw=2, marker="o", ms=3.5, label=p)
        ax.text(t[-1] + 0.4, label_y[p], p, va="center", fontsize=8.5, color=TEXT)
    ax.set_xlim(0.5, 27.5)
    ax.set_ylim(0, None)
    ticks = [1, 4, 8, 12, 16, 20, 24]
    ax.set_xticks(ticks, [f"{x}\n{d['history_k'][x - 1]:.0f}k" for x in ticks])
    ax.set_xlabel("turn  /  history tokens before the turn (median)")
    ax.set_ylabel("arrival → first token, median (s)")
    ax.set_title("A2. Request wait as the conversation grows (every request needs a swap)",
                 loc="left", fontsize=11, color=TEXT)
    ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper left")
    fig.savefig(os.path.join(OUT, "A2_ttft_by_turn.png"))
    plt.close(fig)


# ---------------------------------------------------------------- A3
A3_MODELS = [(1, "Llama-3.2-3B-Instruct"), (2, "Phi-3.5-mini-instruct"), (3, "Meta-Llama-3.1-8B"),
             (4, "Llama-3.1-8B-Instruct")]  # agent i runs pool[(i - 1) % 4]


def a3_data():
    """Median wake (proxy wake = restore of weights + KV) per model and turn: 2 copies x 2 GPUs.

    vllm restores weights only; its history comes back by re-prefill, so it also gets wake + prefill.
    """
    out = {"turns": list(range(1, 25)), "models": []}
    reqs = {p: agent(p)["requests"] for p in PATHS}
    for k, name in A3_MODELS:
        m = {"name": name, "series": {}, "weights_gib": None, "kv_gib": []}
        for p in PATHS:
            load, ready = [], []
            for t in out["turns"]:
                rs = [r for r in reqs[p] if r["turn"] == t and (r["agent"] - 1) % 4 == k - 1]
                load.append(float(np.median([r["load_s"] for r in rs])))
                ready.append(float(np.median([r["load_s"] + r["prefill_s"] for r in rs])))
            m["series"][p] = load
            if p == "vllm":
                m["series"]["vllm_reprefill"] = ready
        for t in out["turns"]:
            rs = [r for r in reqs["dram+dram"] if r["turn"] == t and (r["agent"] - 1) % 4 == k - 1 and r["kv_gib"] is not None]
            m["kv_gib"].append(float(np.median([r["kv_gib"] for r in rs])) if rs else 0.0)
        m["weights_gib"] = float(np.median([r["weights_gib"] for r in reqs["dram+dram"]
                                            if (r["agent"] - 1) % 4 == k - 1 and r["weights_gib"]]))
        out["models"].append(m)
    return out


def a3_plot(d):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.4, "wspace": 0.08})
    t = d["turns"]
    top = max(max(v) for m in d["models"] for v in m["series"].values())
    for ax, m in zip(axes.flat, d["models"]):
        for p in PATHS:
            ax.plot(t, m["series"][p], color=PATH_COLOR[p], lw=1.8, marker="o", ms=2.5, label=p)
        ax.plot(t, m["series"]["vllm_reprefill"], color=PATH_COLOR["vllm"], lw=1.8, ls="--",
                label="vllm wake + re-prefill")
        ax.set_title(f"{m['name']}\n{m['weights_gib']:.1f} GiB weights, KV {m['kv_gib'][-1]:.1f} GiB at turn 24",
                     loc="left", fontsize=9.5, color=TEXT)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_xlim(0.5, 24.5)
        ax.set_ylim(0, top * 1.05)
    for ax in axes[1]:
        ax.set_xlabel("turn")
    for ax in axes[:, 0]:
        ax.set_ylabel("wake time, median (s)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=7, fontsize=8.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("A3. Wake time per turn by model (both GPUs wake at once; KV grows with the history)",
                 x=0.06, ha="left", fontsize=11, color=TEXT, y=0.99)
    fig.savefig(os.path.join(OUT, "A3_wake_time_by_model.png"))
    plt.close(fig)


# ---------------------------------------------------------------- A4
def a4_data(turn=20, agent_idx=3):
    """The step where agent `agent_idx` takes `turn` on both GPUs: sleep of the LRU victim, wake, compute."""
    run = os.path.join(ROOT, "agent_rr_g67_dram+gds")
    s = agent("dram+gds")
    swaps = jsonl(os.path.join(run, "swaps.jsonl"))
    tiered = tiered_events("agent_rr_g67_dram+gds")
    rows = {r["id"]: r for r in s["requests"]}
    t_ref = min(rows[f"{g}:{agent_idx}:{turn}"]["arrive"] for g in (6, 7))
    gpus = {}
    for g in (6, 7):
        r = rows[f"{g}:{agent_idx}:{turn}"]
        segs = []
        for sw in swaps:
            t0, t1 = sw["t_unix"], sw["t_unix"] + sw["duration_s"]
            if sw["gpu"] != g or t1 < r["arrive"] or t0 > r["ready"]:
                continue
            ev = tiered[sw["instance"]]
            if sw["action"] == "sleep":
                rec = min((e for e in ev if e["event"] == "suspend"), key=lambda e: abs(e["t_unix"] - t0))
                a = rec["t_unix"]
                w1 = a + rec["weights_save_s"]
                k0 = w1 + rec["state_save_s"]
                k1 = k0 + rec["kv_save_s"]
                segs += [{"lane": "dram", "kind": "weights", "dir": "offload", "t0": a, "t1": w1,
                          "gib": rec["weights_bytes"] / GIB, "inst": sw["instance"]},
                         {"lane": "ssd", "kind": "kv", "dir": "offload", "t0": k0, "t1": k1,
                          "gib": rec["kv_bytes"] / GIB, "inst": sw["instance"]},
                         {"lane": "phase", "kind": "sleep", "t0": t0, "t1": t1, "inst": sw["instance"]}]
            else:
                rec = min((e for e in ev if e["event"] == "resume"), key=lambda e: abs(e["t_unix"] - t0))
                a = rec["t_unix"]
                segs += [{"lane": "dram", "kind": "weights", "dir": "load", "t0": a + rec["weights_start_s"],
                          "t1": a + rec["weights_end_s"], "gib": rec["weights_bytes"] / GIB, "inst": sw["instance"]},
                         {"lane": "ssd", "kind": "kv", "dir": "load", "t0": a + rec["kv_start_s"],
                          "t1": a + rec["kv_end_s"], "gib": rec["kv_bytes"] / GIB, "inst": sw["instance"]},
                         {"lane": "phase", "kind": "wake", "t0": t0, "t1": t1, "inst": sw["instance"]}]
        segs += [{"lane": "gpu", "kind": "prefill", "t0": r["ready"], "t1": r["ready"] + r["prefill_s"]},
                 {"lane": "gpu", "kind": "decode", "t0": r["ready"] + r["prefill_s"], "t1": r["done"]}]
        for x in segs:
            x["t0"] -= t_ref
            x["t1"] -= t_ref
        gpus[g] = {"segments": segs, "arrive": r["arrive"] - t_ref, "history_len": r["history_len"],
                   "instance": f"g{g}_a{agent_idx}"}
    return {"turn": turn, "agent": agent_idx, "gpus": gpus}


A4_COLOR = {"weights": SLOTS[0], "kv": SLOTS[1], "prefill": SLOTS[2], "decode": "#9ad9c1"}


def a4_plot(d, xmax=4.0):
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    lanes = []
    for g in (6, 7):
        lanes += [(g, "dram", f"GPU{g}  DRAM ↔ GPU"), (g, "ssd", f"GPU{g}  SSD ↔ GPU (GDS)"),
                  (g, "gpu", f"GPU{g}  compute")]
    ypos = {(g, l): len(lanes) - 1 - i for i, (g, l, _) in enumerate(lanes)}
    for g in (6, 7):
        for x in d["gpus"][g]["segments"]:
            if x["lane"] == "phase":
                base = ypos[(g, "gpu")] - 0.45
                ax.axvspan(x["t0"], x["t1"], ymin=0, ymax=0, alpha=0)  # keeps x range honest
                ax.add_patch(plt.Rectangle((x["t0"], base), x["t1"] - x["t0"], 2.9, color="#f0efec", zorder=0))
                ax.text((x["t0"] + x["t1"]) / 2, base + 2.95, f"{x['kind']} {x['inst']}", ha="center",
                        va="bottom", fontsize=7.5, color=TEXT2)
                continue
            y = ypos[(g, x["lane"])]
            t1 = min(x["t1"], xmax)
            hatch = "////" if x.get("dir") == "offload" else None
            ax.barh(y, t1 - x["t0"], left=x["t0"], height=0.56, color=A4_COLOR[x["kind"]],
                    hatch=hatch, edgecolor="white", linewidth=1, zorder=2)
            if "gib" in x and t1 - x["t0"] > 0.18:
                ax.text((x["t0"] + t1) / 2, y, f"{x['gib']:.1f} GiB", ha="center", va="center",
                        fontsize=7.5, color="white", zorder=3,
                        bbox={"facecolor": A4_COLOR[x["kind"]], "edgecolor": "none", "pad": 1.5})
    ax.set_yticks([ypos[(g, l)] for g, l, _ in lanes], [lab for _, _, lab in lanes], fontsize=8.5)
    ax.set_xlim(-0.05, xmax)
    ax.set_ylim(-0.6, len(lanes) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("seconds since the requests arrived (decode cut at the right edge)")
    h = d["gpus"][6]["history_len"]
    ax.set_title(f"A4. One model switch, dram+gds (turn {d['turn']}, agent {d['agent']}, {h / 1000:.0f}k-token history)",
                 loc="left", fontsize=11, color=TEXT)
    from matplotlib.patches import Patch
    handles = [Patch(color=A4_COLOR["weights"], label="weights (DRAM)"), Patch(color=A4_COLOR["kv"], label="KV (SSD, GDS)"),
               Patch(facecolor="white", edgecolor=TEXT2, hatch="////", label="hatched: offload (GPU → store)"),
               Patch(color=A4_COLOR["prefill"], label="prefill"), Patch(color=A4_COLOR["decode"], label="decode")]
    ax.legend(handles=handles, ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.16))
    fig.savefig(os.path.join(OUT, "A4_switch_timeline_dram+gds.png"))
    plt.close(fig)


# ---------------------------------------------------------------- A5
A5_PARTS = [("offload", "sleep the LRU model (offload)", 3), ("load", "wake this model (load)", 2),
            ("prefill", "prefill (vllm: re-prefill of the history)", 1)]


def a5_data(turn=24, model=2):
    """Median split of arrival -> first token at `turn` for the agents running pool model `model`."""
    out = {"turn": turn, "model": dict(A3_MODELS)[model], "rows": []}
    for p in PATHS:
        rs = [r for r in agent(p)["requests"] if r["turn"] == turn and (r["agent"] - 1) % 4 == model - 1]
        row = {"paths": p, "n": len(rs), **{k: float(np.median([r[k + "_s"] for r in rs])) for k, *_ in A5_PARTS}}
        row["other"] = float(np.median([r["other_wait_s"] for r in rs]))
        w, kv = [r["weights_gib"] for r in rs if r["weights_gib"]], [r["kv_gib"] for r in rs if r["kv_gib"]]
        row["weights_gib"] = float(np.median(w)) if w else None  # vllm: no tiered log
        row["kv_gib"] = float(np.median(kv)) if kv else None
        out["rows"].append(row)
    return out


def a5_plot(d):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    rows = d["rows"]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for k, label, slot in A5_PARTS:
        v = np.array([r[k] for r in rows])
        ax.barh(y, v, left=left, height=0.62, color=SLOTS[slot], label=label, edgecolor="white", linewidth=1.5)
        for yy, l, vv in zip(y, left, v):
            if vv >= 0.45:
                ax.text(l + vv / 2, yy, f"{vv:.2f}", ha="center", va="center", fontsize=8.5, color="white")
        left += v
    for yy, tot in zip(y, left):
        ax.text(tot + 0.08, yy, f"{tot:.2f} s", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, [r["paths"] for r in rows])
    ax.set_xlim(0, left.max() * 1.15)
    ax.set_xlabel("seconds, median of 2 copies x 2 GPUs")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"A5. Turn {d['turn']}, {d['model']}: arrival → first token, split by what it waits for",
                 loc="left", fontsize=11, color=TEXT, pad=10)
    ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2))
    fig.savefig(os.path.join(OUT, "A5_turn24_wait_split.png"))
    plt.close(fig)


# ---------------------------------------------------------------- W2
def w2_data():
    """Per wake round (both GPUs at once): bytes each source delivered / time that source was active."""
    rows = []
    for p in PATHS[:-1]:
        wp, kp = p.split("+")
        res = sorted((e for name, ev in tiered_events(WS_RUN[p]).items() if not name.endswith("_bg")
                      for e in ev if e["event"] == "resume" and e["kv_bytes"] > GIB), key=lambda e: e["t_unix"])
        rounds, cur = [], []
        for e in res:
            if cur and e["t_unix"] - cur[0]["t_unix"] > 0.5:
                rounds.append(cur)
                cur = []
            cur.append(e)
        rounds.append(cur)
        rate = {"dram": [], "ssd": []}
        for rd in rounds:
            if len(rd) != 2:
                continue
            spans = {"dram": [], "ssd": []}
            for e in rd:
                a = e["t_unix"]
                spans["dram" if wp == "dram" else "ssd"].append((a + e["weights_start_s"], a + e["weights_end_s"], e["weights_bytes"]))
                spans["dram" if kp == "dram" else "ssd"].append((a + e["kv_start_s"], a + e["kv_end_s"], e["kv_bytes"]))
            for src, sp in spans.items():
                if sp:
                    rate[src].append(sum(b for *_, b in sp) / GIB / (max(t1 for _, t1, _ in sp) - min(t0 for t0, _, _ in sp)))
        rows.append({"paths": p, "dram_gibs": float(np.median(rate["dram"])) if rate["dram"] else 0.0,
                     "ssd_gibs": float(np.median(rate["ssd"])) if rate["ssd"] else 0.0,
                     "rounds": sum(len(r) == 2 for r in rounds)})
    return rows


def w2_plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(rows))[::-1]
    for ax, key, title, color in [
            (axes[0], "dram_gibs", "from DRAM (host uplink, shared by GPU 6+7)", SLOTS[0]),
            (axes[1], "ssd_gibs", "from SSD (RAID0, 2 NVMe)", SLOTS[1])]:
        v = [r[key] for r in rows]
        top = max(v)
        ax.barh(y, v, height=0.6, color=color, edgecolor="white")
        for yy, vv in zip(y, v):
            ax.text(vv + top * 0.02, yy, f"{vv:.1f}" if vv else "not used", va="center", fontsize=8.5,
                    color=TEXT if vv else TEXT2)
        ax.set_xlim(0, top * 1.22)
        ax.set_ylim(-0.5, len(rows) - 0.5)
        ax.set_title(title, fontsize=9.5, color=TEXT, loc="left")
        ax.set_xlabel("GiB/s, both GPUs, while the source is active")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [r["paths"] for r in rows])
    fig.suptitle("W2. Wake storm (both GPUs wake at once, 10.5 GiB weights + 5.4 GiB KV each): aggregate rate per source",
                 x=0.06, ha="left", fontsize=11, color=TEXT, y=1.03)
    fig.savefig(os.path.join(OUT, "W2_source_bandwidth.png"))
    plt.close(fig)


# ---------------------------------------------------------------- W3
def w3_data():
    """Model for two GPUs waking at once: weights from DRAM share the uplink, KV either joins them
    (dram+dram) or comes from the RAID in parallel (dram+gds). Times scaled to 10 GiB of weights."""
    b_dram, b_ssd = UPLINK_GIBS / 2, RAID_GIBS / 2
    ratio = np.linspace(0, 1, 101)
    model = {"ratio": ratio.tolist(), "dram+dram": ((1 + ratio) * 10 / b_dram).tolist(),
             "dram+gds": np.maximum(10 / b_dram, ratio * 10 / b_ssd).tolist(),
             "crossover": b_ssd / (b_dram - b_ssd), "b_dram": b_dram, "b_ssd": b_ssd}
    pts = []
    for src, run_of in [("agent", lambda p: f"agent_rr_g67_{p}"), ("wake storm", lambda p: WS_RUN[p])]:
        for p in ("dram+dram", "dram+gds"):
            res = [e for name, ev in tiered_events(run_of(p)).items() if not name.endswith("_bg")
                   for e in ev if e["event"] == "resume" and e["kv_bytes"] > (1 << 28)]
            r = np.array([e["kv_bytes"] / e["weights_bytes"] for e in res])
            t = np.array([(max(e["weights_end_s"], e["kv_end_s"]) - min(e["weights_start_s"], e["kv_start_s"]))
                          * 10 * GIB / e["weights_bytes"] for e in res])
            for lo in np.arange(0, 1.0, 0.1):
                m = (r >= lo) & (r < lo + 0.1)
                if m.sum() >= 8:
                    pts.append({"source": src, "paths": p, "ratio": float(np.median(r[m])),
                                "time_s": float(np.median(t[m])), "n": int(m.sum())})
    return {"model": model, "points": pts}


def w3_plot(d):
    m = d["model"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for p in ("dram+dram", "dram+gds"):
        ax.plot(m["ratio"], m[p], color=PATH_COLOR[p], lw=2, label=f"{p} model")
        for src, mk in (("agent", "o"), ("wake storm", "D")):
            pts = [x for x in d["points"] if x["paths"] == p and x["source"] == src]
            ax.scatter([x["ratio"] for x in pts], [x["time_s"] for x in pts], color=PATH_COLOR[p], marker=mk,
                       s=38, edgecolor="white", linewidth=1.5, zorder=3,
                       label=f"{p} measured ({src})")
    c = m["crossover"]
    ax.axvline(c, color=TEXT2, ls="--", lw=1)
    ax.text(c + 0.01, 0.05, f"crossover K/W = {c:.2f}\n(B_ssd / (B_dram − B_ssd))", fontsize=8, color=TEXT2,
            transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_xlabel("KV bytes / weights bytes per wake (K/W)")
    ax.set_ylabel("wake transfer time per 10 GiB weights (s)")
    ax.set_title(f"W3. When does GDS help? 2 GPUs wake at once: {m['b_dram']:.1f} GiB/s DRAM and "
                 f"{m['b_ssd']:.1f} GiB/s SSD per GPU", loc="left", fontsize=10.5, color=TEXT)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(os.path.join(OUT, "W3_gds_crossover.png"))
    plt.close(fig)

# ---------------------------------------------------------------- B1 / B2
B_PATHS = ["dram+dram", "dram+ssd", "dram+gds"]
B_TURNS = [4, 12, 24]
B_AGENT = 2  # Phi-3.5-mini: no GQA, ~384 KiB KV per token, so the largest history KV


def bw_data(t_pre=0.15, t_post=2.3):
    """NVML PCIe RX of GPU 6/7 and md0 reads (pcie.csv, 0.1 s samples) around the wake of agent
    B_AGENT at each turn, both GPUs (they wake it in the same step), plus the plugin's weights / KV
    windows. Rates in GB/s (NVML KB/s and block-layer bytes, decimal)."""
    out = {}
    for p in B_PATHS:
        run = os.path.join(ROOT, f"agent_rr_g67_{p}")
        rows = list(csv.DictReader(open(os.path.join(run, "pcie.csv"))))
        tt = np.array([float(r["t"]) for r in rows])
        res = {g: [e for e in jsonl(os.path.join(run, f"tiered_g{g}_a{B_AGENT}.jsonl")) if e["event"] == "resume"]
               for g in (6, 7)}
        for turn in B_TURNS:
            ev = {g: next(e for e in res[g] if e["n"] == turn - 1) for g in (6, 7)}
            t_ref = min(e["t_unix"] for e in ev.values())
            m = (tt >= t_ref - t_pre) & (tt <= t_ref + t_post)
            sel = [rows[i] for i in np.flatnonzero(m)]
            spans = {}
            for g, e in ev.items():
                a = e["t_unix"] - t_ref
                spans[g] = {"weights": [a + e["weights_start_s"], a + e["weights_end_s"]],
                            "kv": [a + e["kv_start_s"], a + e["kv_end_s"]]}
            out[f"{p}|{turn}"] = {
                "paths": p, "turn": turn, "t": [round(float(r["t"]) - t_ref, 3) for r in sel],
                "gpu6": [float(r["gpu6_rx_mbs"]) / 1000 for r in sel], "gpu7": [float(r["gpu7_rx_mbs"]) / 1000 for r in sel],
                "md0": [float(r["md0_read_mbs"]) / 1000 for r in sel], "spans": spans,
                "weights_gib": ev[6]["weights_bytes"] / GIB, "kv_gib": ev[6]["kv_bytes"] / GIB,
                "done_s": max(max(s["weights"][1], s["kv"][1]) for s in spans.values())}
    return {"paths": B_PATHS, "turns": B_TURNS, "agent": B_AGENT, "cells": out}


B_GPU = {"gpu6": SLOTS[0], "gpu7": SLOTS[2]}


def bw_panel(ax, c, strip=True, xmax=2.3, ymax=95):
    t = np.array(c["t"])
    ax.stackplot(t, c["gpu6"], c["gpu7"], colors=[B_GPU["gpu6"], B_GPU["gpu7"]], alpha=.85, step=None,
                 labels=["GPU6 PCIe RX", "GPU7 PCIe RX"], edgecolor="white", linewidth=.6)
    ax.plot(t, c["md0"], color=SLOTS[1], lw=2, ls="--", label="SSD read (md0)")
    if strip:  # plugin windows (both GPUs' union) as thin bars at the top
        for k, (key, lab) in enumerate((("weights", "weights"), ("kv", "KV"))):
            a = min(s[key][0] for s in c["spans"].values())
            b = max(s[key][1] for s in c["spans"].values())
            y = ymax * (0.95 - 0.08 * k)
            ax.plot([a, b], [y, y], color=TEXT2, lw=4, solid_capstyle="butt")
            ax.text(b + 0.03, y, lab, va="center", fontsize=7.5, color=TEXT2)
    ax.set_xlim(-0.1, xmax)
    ax.set_ylim(0, ymax)
    ax.grid(axis="x", visible=False)


def b1_plot(d):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, p in zip(axes, d["paths"]):
        c = d["cells"][f"{p}|24"]
        bw_panel(ax, c)
        ax.set_title(f"{p}  ·  done {c['done_s']:.2f} s", loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("seconds since the wake started")
    axes[0].set_ylabel("GB/s (NVML / block layer)")
    c = d["cells"][f"{d['paths'][0]}|24"]
    fig.suptitle(f"B1. One wake on both GPUs at once: Phi-3.5 at turn 24, {c['weights_gib']:.1f} GiB weights + "
                 f"{c['kv_gib']:.1f} GiB KV per GPU", x=0.08, ha="left", fontsize=11, color=TEXT, y=1.04)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.84))
    fig.savefig(os.path.join(OUT, "B1_wake_bandwidth_timeline.png"))
    plt.close(fig)


def b2_plot(d):
    fig, axes = plt.subplots(len(d["turns"]), 3, figsize=(10.5, 7.4), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.32})
    for i, turn in enumerate(d["turns"]):
        for j, p in enumerate(d["paths"]):
            c = d["cells"][f"{p}|{turn}"]
            ax = axes[i, j]
            bw_panel(ax, c)
            ax.set_title(f"{p} · turn {turn} · KV {c['kv_gib']:.1f} GiB · {c['done_s']:.2f} s",
                         loc="left", fontsize=9, color=TEXT)
        axes[i, 0].set_ylabel("GB/s")
    for ax in axes[-1]:
        ax.set_xlabel("seconds since the wake started")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.84))
    fig.suptitle(f"B2. The same wake as the history grows (Phi-3.5, {d['cells']['dram+dram|24']['weights_gib']:.1f} GiB weights; "
                 "both GPUs at once)", x=0.08, ha="left", fontsize=11, color=TEXT, y=0.95)
    fig.savefig(os.path.join(OUT, "B2_wake_bandwidth_by_turn.png"))
    plt.close(fig)

# ---------------------------------------------------------------- O1 / O2 / O3
O_PATHS = ["dram+dram", "dram+gds"]
OFFSETS = ["0", "0.3", "1"]
OFF_COLOR = dict(zip(OFFSETS, SLOTS))


def off_run(p, off):
    return f"agent_rr_g67_{p}" + ("" if off == "0" else f"_off{off}")


def off_data(step_turn=24, step_agent=2, t_pre=0.2, t_post=3.0):
    gpu_time, ttft, steps = [], {}, {}
    for p in O_PATHS:
        for off in OFFSETS:
            run = off_run(p, off)
            s = json.load(open(os.path.join(ROOT, run, "summary.json")))
            gp = list(s["gpus"].values())
            mean = lambda k: float(np.mean([g[k] for g in gp]))  # noqa: E731
            pre = float(np.mean([sum(r["prefill_s"] for r in s["requests"] if str(r["gpu"]) == g) / st["window_s"]
                                 for g, st in s["gpus"].items()]))
            gpu_time.append({"paths": p, "offset": off, "window_s": mean("window_s"), "prefill": pre,
                             "decode": mean("compute") - pre, "load": mean("load"), "offload": mean("offload"),
                             "sync": mean("sync"), "other": mean("other")})
            ttft[f"{p}|{off}"] = [float(np.median([r["wait_s"] + r["prefill_s"] for r in s["requests"] if r["turn"] == t]))
                                  for t in range(1, 25)]
            # one step: PCIe RX / TX of both GPUs and their sleep / wake windows
            rows = {r["id"]: r for r in s["requests"]}
            t_ref = rows[f"6:{step_agent}:{step_turn}"]["arrive"]
            pc = [r for r in csv.DictReader(open(os.path.join(ROOT, run, "pcie.csv")))
                  if t_ref - t_pre <= float(r["t"]) <= t_ref + t_post]
            swaps = [x for x in jsonl(os.path.join(ROOT, run, "swaps.jsonl"))
                     if x["t_unix"] + x["duration_s"] >= t_ref - t_pre and x["t_unix"] <= t_ref + t_post]
            steps[f"{p}|{off}"] = {
                "t": [round(float(r["t"]) - t_ref, 3) for r in pc],
                **{f"gpu{g}_{d}": [float(r[f"gpu{g}_{d}_mbs"]) / 1000 for r in pc] for g in (6, 7) for d in ("rx", "tx")},
                "phases": [{"gpu": x["gpu"], "action": x["action"], "inst": x["instance"],
                            "t0": x["t_unix"] - t_ref, "t1": x["t_unix"] + x["duration_s"] - t_ref} for x in swaps]}
    return {"paths": O_PATHS, "offsets": OFFSETS, "gpu_time": gpu_time, "ttft": ttft, "steps": steps,
            "step": {"turn": step_turn, "agent": step_agent}}


O1_STATES = [("decode", "Compute: decode"), ("prefill", "Compute: prefill"), ("load", "Wait: load (wake)"),
             ("offload", "Wait: offload (sleep)"), ("sync", "Wait: other GPU (lockstep + offset)")]


def o1_plot(d):
    rows = d["gpu_time"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = np.arange(len(rows))[::-1].astype(float)
    y[:3] += 0.4  # gap between the two paths
    left = np.zeros(len(rows))
    for i, (k, label) in enumerate(O1_STATES):
        v = np.array([r[k] * 100 for r in rows])
        ax.barh(y, v, left=left, height=0.62, color=SLOTS[i], label=label, edgecolor="white", linewidth=1.5)
        for yy, l, vv in zip(y, left, v):
            if vv >= 5:
                ax.text(l + vv / 2, yy, f"{vv:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        left += v
    for yy, r in zip(y, rows):
        ax.text(101, yy, f"{r['window_s']:.0f} s", va="center", fontsize=9, color=TEXT2)
    ax.set_yticks(y, [f"{r['paths']}  Δ{r['offset']} s" for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of GPU time (mean of GPU 6 and 7), %   ·   right: makespan")
    ax.grid(axis="y", visible=False)
    ax.set_title("O1. GPU 7 sends each step's request Δ s after GPU 6", loc="left", fontsize=11, color=TEXT)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.18))
    fig.savefig(os.path.join(OUT, "O1_offset_gpu_time.png"))
    plt.close(fig)


def o2_plot(d):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    t = list(range(1, 25))
    for ax, p in zip(axes, d["paths"]):
        for off in d["offsets"]:
            ax.plot(t, d["ttft"][f"{p}|{off}"], color=OFF_COLOR[off], lw=2, marker="o", ms=3, label=f"Δ {off} s")
        ax.set_title(p, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("turn")
    axes[0].set_ylim(0, 1.08 * max(max(v) for v in d["ttft"].values()))
    axes[0].set_ylabel("arrival → first token, median (s)")
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("O2. Request wait per turn with the GPUs' swaps staggered by Δ", x=0.08, ha="left",
                 fontsize=11, color=TEXT, y=1.02)
    fig.savefig(os.path.join(OUT, "O2_offset_ttft_by_turn.png"))
    plt.close(fig)


def o3_plot(d, p="dram+dram"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True, sharey=True, gridspec_kw={"wspace": 0.06, "hspace": 0.28})
    for j, off in enumerate(["0", "0.3"]):
        st = d["steps"][f"{p}|{off}"]
        t = np.array(st["t"])
        for i, g in enumerate((6, 7)):
            ax = axes[i, j]
            ax.fill_between(t, 0, st[f"gpu{g}_rx"], color=SLOTS[0], alpha=.85, lw=0, label="RX (host → GPU, wake)")
            ax.fill_between(t, 0, -np.array(st[f"gpu{g}_tx"]), color=SLOTS[3], alpha=.85, lw=0, label="TX (GPU → host, sleep)")
            ax.axhline(0, color=TEXT2, lw=.8)
            for ph in st["phases"]:
                if ph["gpu"] == g:
                    ax.axvspan(ph["t0"], ph["t1"], color="#f0efec", zorder=0)
                    ax.text((ph["t0"] + ph["t1"]) / 2, 60, ph["action"], ha="center", fontsize=7.5, color=TEXT2)
            ax.set_ylim(-66, 68)
            ax.set_xlim(-0.2, 3.0)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"Δ {off} s · GPU{g}", loc="left", fontsize=9.5, color=TEXT)
        axes[1, j].set_xlabel("seconds since GPU 6's request arrived")
    for i in range(2):
        axes[i, 0].set_ylabel("GB/s (NVML)")
        axes[i, 0].set_yticks([-60, -40, -20, 0, 20, 40, 60], ["60", "40", "20", "0", "20", "40", "60"])
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle(f"O3. One step ({p}, turn {d['step']['turn']}, agent {d['step']['agent']}): with Δ 0.3 s GPU 6 wakes while "
                 "GPU 7 sleeps, opposite PCIe directions", x=0.06, ha="left", fontsize=10.5, color=TEXT, y=1.0)
    fig.savefig(os.path.join(OUT, "O3_offset_duplex_timeline.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {"A1": a1_data(), "A2": a2_data(), "A3": a3_data(), "A4": a4_data(), "A5": a5_data(), "W2": w2_data(), "W3": w3_data(), "B": bw_data(), "O": off_data(),
            "ceilings": {"uplink_gibs": UPLINK_GIBS, "raid_gibs": RAID_GIBS}}
    a1_plot(data["A1"])
    a2_plot(data["A2"])
    a3_plot(data["A3"])
    a4_plot(data["A4"])
    a5_plot(data["A5"])
    w2_plot(data["W2"])
    w3_plot(data["W3"])
    b1_plot(data["B"])
    b2_plot(data["B"])
    o1_plot(data["O"])
    o2_plot(data["O"])
    o3_plot(data["O"])
    with open(os.path.join(OUT, "figures_data.json"), "w") as f:
        json.dump(data, f, indent=1)
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_page.html")).read()
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(page.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print(json.dumps({k: data[k] for k in ("A1", "W2")}, indent=1))
    print("W3 points", json.dumps(data["W3"]["points"]))
    print("crossover", data["W3"]["model"]["crossover"])


if __name__ == "__main__":
    main()
