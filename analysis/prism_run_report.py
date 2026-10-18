"""Report for one run of scripts/run_prism_trace.sh.

Reads <run_dir>/{config.txt,instances.tsv,startup.tsv,vram.csv,requests.jsonl,server_*.log}
and writes:
  summary.json   every number below, machine-readable
  (with SWAP=1 also swaps.jsonl and proxy_requests.jsonl: swap counts, durations,
  per-request swap wait)
  summary.md     the same as tables
  report.png     per-GPU memory over time (stacked per instance), KV in use, queue depth

Phases (from config.txt timestamps):
  baseline  before the first instance starts (other tenants on the GPU)
  idle      all instances loaded, before traffic
  workload  trace replay

usage: python analysis/prism_run_report.py <run_dir>
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GIB_PER_MIB = 1 / 1024

LOG_PATTERNS = {
    "weights_gib": r"Model loading took ([\d.]+) GiB",
    "cudagraph_gib": r"Graph capturing finished in [\d.]+ secs, took ([\d.]+) GiB",
    "torch_peak_bytes": r"torch_peak=(\d+) bytes",
    "kv_capacity_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "engine_init_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
}
ERROR_RE = re.compile(r"(out of memory|OutOfMemoryError|CUDA error[^\n]*|RuntimeError: [^\n]*)", re.I)


def read_config(run_dir):
    cfg = {}
    with open(os.path.join(run_dir, "config.txt")) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            cfg[k] = v
    return cfg


def parse_log(path):
    out = {}
    try:
        text = open(path, errors="replace").read()
    except FileNotFoundError:
        return {"log_missing": True}
    for key, pat in LOG_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1).replace(",", ""))
    errors = sorted({e.strip()[:200] for e in ERROR_RE.findall(text)})
    if errors:
        out["errors"] = errors[:5]
    return out


def pct(values, q):
    return round(float(np.percentile(values, q)), 4) if len(values) else None


def main():
    run_dir = sys.argv[1]
    cfg = read_config(run_dir)
    inst = pd.read_csv(os.path.join(run_dir, "instances.tsv"), sep="\t", header=None,
                       names=["name", "model", "gpu", "port"])
    startup_path = os.path.join(run_dir, "startup.tsv")
    startup = pd.read_csv(startup_path, sep="\t") if os.path.exists(startup_path) else pd.DataFrame()
    vram = pd.read_csv(os.path.join(run_dir, "vram.csv"))
    gpus = sorted(inst["gpu"].unique())

    t_launch = float(cfg.get("launch_start", vram["t"].iloc[0]))
    t_idle = float(cfg["launch_end"]) if "launch_end" in cfg else None
    t_work = float(cfg["workload_start"]) if "workload_start" in cfg else None
    t_end = float(cfg["workload_end"]) if "workload_end" in cfg else None
    phase = {
        "baseline": vram["t"] < t_launch,
        "idle": (vram["t"] >= t_idle) & (vram["t"] < t_work) if t_work else vram["t"] < 0,
        "workload": (vram["t"] >= t_work) & (vram["t"] <= t_end) if t_end else vram["t"] < 0,
    }

    summary = {"config": cfg, "launch_failed": cfg.get("launch_failed") == "1", "gpus": {}, "instances": {}}

    for g in gpus:
        used, total = vram[f"gpu{g}_used_mib"], vram[f"gpu{g}_total_mib"].iloc[0]
        s = {"total_gib": round(total * GIB_PER_MIB, 2),
             "models": list(inst.loc[inst.gpu == g, "name"])}
        for p, mask in phase.items():
            if mask.any():
                s[f"{p}_used_gib_max"] = round(used[mask].max() * GIB_PER_MIB, 2)
        s["peak_used_gib"] = round(used.max() * GIB_PER_MIB, 2)
        s["min_free_gib"] = round((total - used.max()) * GIB_PER_MIB, 2)
        if phase["idle"].any():
            s["idle_free_gib"] = round((total - used[phase["idle"]].max()) * GIB_PER_MIB, 2)
        summary["gpus"][int(g)] = s

    for _, r in inst.iterrows():
        n = r["name"]
        s = {"model": r["model"], "gpu": int(r["gpu"]), "port": int(r["port"]),
             "started": bool(len(startup) and (startup["name"] == n).any())}
        if s["started"]:
            row = startup[startup["name"] == n].iloc[0]
            s["startup_s"] = int(row["startup_s"])
            s["memory_when_healthy_gib"] = round(row["instance_mib"] * GIB_PER_MIB, 2)
        s.update(parse_log(os.path.join(run_dir, f"server_{n}.log")))
        col = f"{n}_proc_mib"
        for p in ("idle", "workload"):
            if phase[p].any() and vram.loc[phase[p], col].notna().any():
                s[f"{p}_memory_gib_max"] = round(vram.loc[phase[p], col].max() * GIB_PER_MIB, 2)
        if phase["workload"].any():
            w = vram[phase["workload"]]
            for key, c in (("kv_used_gib_max", f"{n}_kv_used_mib"), ("kv_prealloc_gib_max", f"{n}_kv_prealloc_mib")):
                if w[c].notna().any():
                    s[key] = round(w[c].max() * GIB_PER_MIB, 2)
            if w[f"{n}_waiting"].notna().any():
                s["max_waiting"] = int(w[f"{n}_waiting"].max())
                s["max_running"] = int(w[f"{n}_running"].max())
            pre = w[f"{n}_preemptions"].dropna()
            if len(pre):
                s["preemptions"] = int(pre.iloc[-1] - pre.iloc[0])
        summary["instances"][n] = s

    req_path = os.path.join(run_dir, "requests.jsonl")
    if os.path.exists(req_path):
        reqs = pd.read_json(req_path, lines=True)
        summary["requests"] = {"total": int(len(reqs)), "ok": int(reqs["ok"].sum()),
                               "failed": int((~reqs["ok"]).sum()), "per_model": {}}
        if (~reqs["ok"]).any():
            summary["requests"]["errors"] = reqs.loc[~reqs["ok"], "error"].astype(str).str[:150] \
                .value_counts().head(5).to_dict()
        for m, d in reqs.groupby("model"):
            ok = d[d["ok"]]
            summary["requests"]["per_model"][m] = {
                "n": int(len(d)), "failed": int((~d["ok"]).sum()),
                **{f"ttft_p{q}_s": pct(ok["ttft_s"], q) for q in (50, 95, 99)},
                **{f"tpot_p{q}_ms": (pct(ok["tpot_s"].dropna() * 1000, q)) for q in (50, 95, 99)},
                "e2e_p50_s": pct(ok["e2e_s"], 50), "e2e_p99_s": pct(ok["e2e_s"], 99),
            }

    swaps = None
    swap_path = os.path.join(run_dir, "swaps.jsonl")
    if os.path.exists(swap_path) and os.path.getsize(swap_path):
        swaps = pd.read_json(swap_path, lines=True)
        sw = {"by_action": {}, "per_gpu_swap_time_s": {}, "per_instance": {}}
        for a, d in swaps.groupby("action"):
            sw["by_action"][a] = {"count": int(len(d)), "duration_p50_s": pct(d["duration_s"], 50),
                                  "duration_p95_s": pct(d["duration_s"], 95),
                                  "duration_max_s": round(float(d["duration_s"].max()), 4)}
        for g, d in swaps.groupby("gpu"):
            sw["per_gpu_swap_time_s"][int(g)] = round(float(d["duration_s"].sum()), 3)
        for n, d in swaps.groupby("instance"):
            sw["per_instance"][n] = {"wakes": int((d["action"] == "wake").sum()),
                                     "sleeps": int((d["action"] == "sleep").sum())}
        preq_path = os.path.join(run_dir, "proxy_requests.jsonl")
        if os.path.exists(preq_path):
            preq = pd.read_json(preq_path, lines=True)
            wait = preq["swap_wait_s"].dropna()
            sw["requests"] = {"total": int(len(preq)), "triggered_wake": int(preq["cold"].fillna(False).sum()),
                              "waited_over_100ms": int((wait > 0.1).sum()),
                              **{f"swap_wait_p{q}_s": pct(wait, q) for q in (50, 95, 99)},
                              "swap_wait_max_s": round(float(wait.max()), 4) if len(wait) else None}
        summary["swaps"] = sw

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    write_markdown(run_dir, summary)
    plot(run_dir, vram, inst, gpus, t_launch, t_idle, t_work, t_end, swaps)
    print(open(os.path.join(run_dir, "summary.md")).read())


def write_markdown(run_dir, s):
    L = [f"# {os.path.basename(run_dir.rstrip('/'))}", ""]
    c = s["config"]
    L += [f"mode `{c.get('mode')}` · vLLM {c.get('vllm')} · kvcached {c.get('kvcached_git')} · "
          f"{c.get('gpu_name')} · time scale {c.get('time_scale')} · replication {c.get('replication')}", ""]
    if s["launch_failed"]:
        L += ["**Launch failed** — not every instance came up; see the instance table and logs.", ""]
    L += ["## GPUs", "", "| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |",
          "|---|---|---|---|---|---|---|"]
    for g, d in s["gpus"].items():
        L.append(f"| {g} | {', '.join(d['models'])} | {d['total_gib']} | {d.get('idle_used_gib_max', '–')} | "
                 f"{d.get('idle_free_gib', '–')} | {d['peak_used_gib']} | {d['min_free_gib']} |")
    L += ["", "## Instances", "",
          "| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | "
          "peak mem GiB | KV used max GiB | max waiting | preemptions | errors |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, d in s["instances"].items():
        L.append(f"| {n} | {d['model'].split('/')[-1]} | {d['gpu']} | {'yes' if d['started'] else 'NO'} | "
                 f"{d.get('startup_s', '–')} | {d.get('weights_gib', '–')} | {d.get('cudagraph_gib', '–')} | "
                 f"{d.get('idle_memory_gib_max', '–')} | {d.get('workload_memory_gib_max', '–')} | "
                 f"{d.get('kv_used_gib_max', '–')} | {d.get('max_waiting', '–')} | {d.get('preemptions', '–')} | "
                 f"{'; '.join(d.get('errors', [])) or '–'} |")
    if "requests" in s:
        r = s["requests"]
        L += ["", f"## Requests — {r['ok']}/{r['total']} ok, {r['failed']} failed", "",
              "| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |",
              "|---|---|---|---|---|---|"]
        for m, d in r["per_model"].items():
            L.append(f"| {m} | {d['n']} | {d['failed']} | {d['ttft_p50_s']} / {d['ttft_p95_s']} / {d['ttft_p99_s']} | "
                     f"{d['tpot_p50_ms']} / {d['tpot_p95_ms']} / {d['tpot_p99_ms']} | {d['e2e_p50_s']} / {d['e2e_p99_s']} |")
        for e, k in r.get("errors", {}).items():
            L.append(f"- {k}× `{e}`")
    if "swaps" in s:
        w = s["swaps"]
        L += ["", "## Weight swaps (DRAM)", "", "| action | count | p50 s | p95 s | max s |", "|---|---|---|---|---|"]
        for a, d in w["by_action"].items():
            L.append(f"| {a} | {d['count']} | {d['duration_p50_s']} | {d['duration_p95_s']} | {d['duration_max_s']} |")
        L += ["", "time spent swapping per GPU (s): "
              + ", ".join(f"GPU {g}: {t}" for g, t in w["per_gpu_swap_time_s"].items()), "",
              "| instance | wakes | sleeps |", "|---|---|---|"]
        for n, d in w["per_instance"].items():
            L.append(f"| {n} | {d['wakes']} | {d['sleeps']} |")
        if "requests" in w:
            q = w["requests"]
            L += ["", f"requests: {q['total']}, triggered a wake: {q['triggered_wake']}, "
                  f"waited > 100 ms for a swap: {q['waited_over_100ms']}; swap wait p50/p95/p99/max s: "
                  f"{q['swap_wait_p50_s']} / {q['swap_wait_p95_s']} / {q['swap_wait_p99_s']} / {q['swap_wait_max_s']}"]
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(run_dir, vram, inst, gpus, t_launch, t_idle, t_work, t_end, swaps=None):
    t0 = t_work or t_launch
    x = vram["t"] - t0
    fig, axes = plt.subplots(3, len(gpus), figsize=(6.5 * len(gpus), 9), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    for col, g in enumerate(gpus):
        names = list(inst.loc[inst.gpu == g, "name"])
        models = dict(zip(inst["name"], inst["model"].str.split("/").str[-1]))
        ax_mem, ax_kv, ax_q = axes[:, col]
        stack = [vram[f"{n}_proc_mib"].fillna(0) * GIB_PER_MIB for n in names]
        ax_mem.stackplot(x, *stack, colors=SERIES[:len(names)], linewidth=0,
                         labels=[f"{n} ({models[n]})" for n in names])
        ax_mem.plot(x, vram[f"gpu{g}_used_mib"] * GIB_PER_MIB, color=INK, linewidth=1, linestyle="--",
                    label="GPU used (all processes)")
        cap = vram[f"gpu{g}_total_mib"].iloc[0] * GIB_PER_MIB
        ax_mem.axhline(cap, color=INK, linewidth=1)
        ax_mem.annotate(f"GPU capacity {cap:.1f} GiB", (x.iloc[-1], cap), xytext=(0, -4),
                        textcoords="offset points", ha="right", va="top", fontsize=8, color=INK)
        ax_mem.set_ylim(0, cap * 1.08)
        ax_mem.set_title(f"GPU {g}", loc="left", fontsize=11, color=INK)
        if swaps is not None:
            for i, n in enumerate(names):
                for action, marker in (("wake", "^"), ("sleep", "v")):
                    ev = swaps[(swaps["instance"] == n) & (swaps["action"] == action)]
                    if len(ev):
                        ax_mem.scatter(ev["t_unix"] - t0, [cap * (1.035 if action == "wake" else 1.0)] * len(ev),
                                       marker=marker, s=18, color=SERIES[i % len(SERIES)],
                                       edgecolors="none" if action == "wake" else SERIES[i % len(SERIES)],
                                       facecolors=SERIES[i % len(SERIES)] if action == "wake" else "none", linewidths=1)
            ax_mem.annotate("▲ wake  ▽ sleep", (1, 1), xycoords="axes fraction", xytext=(0, 4),
                            textcoords="offset points", ha="right", fontsize=7, color=MUTED)
        ax_mem.legend(loc="upper left", fontsize=7, frameon=False)
        for i, n in enumerate(names):
            if vram[f"{n}_kv_used_mib"].notna().any():
                ax_kv.plot(x, vram[f"{n}_kv_used_mib"] * GIB_PER_MIB, color=SERIES[i % len(SERIES)], linewidth=2, label=n)
            if vram[f"{n}_waiting"].notna().any():
                ax_q.plot(x, vram[f"{n}_waiting"], color=SERIES[i % len(SERIES)], linewidth=2, label=n)
        for ax in (ax_mem, ax_kv, ax_q):
            style(ax)
            for t, lbl in ((t_idle, None), (t_work, "trace start"), (t_end, "trace end")):
                if t:
                    ax.axvline(t - t0, color=MUTED, linewidth=0.8, linestyle=":")
                    if ax is ax_mem and lbl:
                        ax.annotate(lbl, (t - t0, cap * 1.02), fontsize=7, color=MUTED, ha="center")
        if col == 0:
            ax_mem.set_ylabel("GPU memory (GiB)", color=MUTED, fontsize=9)
            ax_kv.set_ylabel("KV in use (GiB)", color=MUTED, fontsize=9)
            ax_q.set_ylabel("waiting requests", color=MUTED, fontsize=9)
        ax_q.set_xlabel("seconds since trace start" if t_work else "seconds since launch",
                        color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "report.png"), dpi=130)


if __name__ == "__main__":
    main()
