"""Figures for the GPU 6+7 data-movement experiments (agent round-robin + wake storm).

  A1  GPU time by state per offload path            (agent_rr_g67_<paths>)
  A2  arrival -> first token per turn                (agent_rr_g67_<paths>)
  A4  one model switch on both GPUs, per data source (agent_rr_g67_dram+gds)
  W2  aggregate rate per data source vs ceilings     (wake_storm_g67_kv5_d0_<paths>)
  W3  wake transfer time vs KV/weights ratio: model + measurements

Writes results/figures/<name>.png, figures_data.json, and index.html (analysis/figures_page.html
with the data inlined: the interactive version of the same figures).

usage: python analysis/make_figures.py
"""

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
    for ax, key, cap, title, color in [
            (axes[0], "dram_gibs", UPLINK_GIBS, "from DRAM (host uplink, shared by GPU 6+7)", SLOTS[0]),
            (axes[1], "ssd_gibs", RAID_GIBS, "from SSD (RAID0, 2 NVMe)", SLOTS[1])]:
        v = [r[key] for r in rows]
        ax.barh(y, v, height=0.6, color=color, edgecolor="white")
        for yy, vv in zip(y, v):
            ax.text(vv + cap * 0.02, yy, f"{vv:.1f}" if vv else "not used", va="center", fontsize=8.5,
                    color=TEXT if vv else TEXT2)
        ax.axvline(cap, color=TEXT2, lw=1.2, ls="--")
        ax.text(cap * 0.98, -0.62, f"ceiling {cap}", ha="right", va="center", fontsize=8, color=TEXT2)
        ax.set_xlim(0, cap * 1.22)
        ax.set_ylim(-0.9, len(rows) - 0.5)
        ax.set_title(title, fontsize=9.5, color=TEXT, loc="left")
        ax.set_xlabel("GiB/s, both GPUs, while the source is active")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [r["paths"] for r in rows])
    fig.suptitle("W2. Wake storm (both GPUs wake at once, 10.5 GiB weights + 5.4 GiB KV each): each source hits its ceiling",
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


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {"A1": a1_data(), "A2": a2_data(), "A4": a4_data(), "W2": w2_data(), "W3": w3_data(),
            "ceilings": {"uplink_gibs": UPLINK_GIBS, "raid_gibs": RAID_GIBS}}
    a1_plot(data["A1"])
    a2_plot(data["A2"])
    a4_plot(data["A4"])
    w2_plot(data["W2"])
    w3_plot(data["W3"])
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
