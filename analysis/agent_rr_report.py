"""Report of an interleaved-agents run (scripts/run_agent_rr.sh).

Joins the client's requests.jsonl, the swap proxy's proxy_requests.jsonl and swaps.jsonl
(same X-Request-Id), and the tiered_restore logs, then reports:

  per request   wait before the request reaches vLLM (proxy arrive -> ready), split into
                offload (sleeps of other agents on its GPU during the wait), load (the
                agent's own wake) and other; then prefill (ready -> first token: new input
                only on a prefix hit, the whole history after vllm's re-prefill), decode;
                prefix-cache hit tokens; weights / KV bytes the wake brought back
  per GPU       timeline over the workload, each instant in one state, first match wins:
                  compute   a request is between ready and done (prefill or decode)
                  load      a wake runs (a request is waiting for it)
                  offload   a sleep runs
                  sync      no request outstanding on this GPU: waiting for the other GPU
                            to finish the step (--sync)
                  other     a request is outstanding but none of the above (proxy, HTTP)
  per turn bin  the same per-request numbers grouped by turn, as the history grows
Writes <run_dir>/summary.md and summary.json.

usage: python analysis/agent_rr_report.py <run_dir>
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

GIB = 1 << 30
STATES = ["compute", "load", "offload", "sync", "other"]
TURN_BINS = [(1, 1), (2, 6), (7, 12), (13, 18), (19, 24)]


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def p50(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main(run_dir):
    cfg = dict(l.strip().split("=", 1) for l in open(os.path.join(run_dir, "config.txt")) if "=" in l)
    reqs = load_jsonl(os.path.join(run_dir, "requests.jsonl"))
    preqs = {r["id"]: r for r in load_jsonl(os.path.join(run_dir, "proxy_requests.jsonl"))}
    swaps = load_jsonl(os.path.join(run_dir, "swaps.jsonl"))
    tiered = {os.path.basename(f)[len("tiered_"):-len(".jsonl")]: load_jsonl(f)
              for f in glob.glob(os.path.join(run_dir, "tiered_*.jsonl"))}
    if not swaps or not reqs:
        print("no swaps or requests recorded")
        return
    t0 = swaps[0]["t_unix"] - swaps[0]["t"]  # proxy clock -> unix
    for s in swaps:
        s["t0"], s["t1"] = s["t_unix"], s["t_unix"] + s["duration_s"]
    inst_name = {}  # (gpu, agent) -> instance
    for line in open(os.path.join(run_dir, "instances.tsv")).read().splitlines():
        name, _, gpu, _ = line.split("\t")
        inst_name[(int(gpu), int(name.split("_a")[1]))] = name

    rows = []
    for r in reqs:
        p = preqs.get(r["id"])
        if not r["ok"] or p is None or "ready_s" not in p:
            continue
        name = inst_name[(r["gpu"], r["agent"])]
        arrive, ready = t0 + p["arrive_s"], t0 + p["ready_s"]
        gpu_swaps = [s for s in swaps if s["gpu"] == r["gpu"]]
        offload = sum(overlap(arrive, ready, s["t0"], s["t1"]) for s in gpu_swaps if s["action"] == "sleep")
        own_wakes = [s for s in gpu_swaps if s["action"] == "wake" and s["instance"] == name
                     and overlap(arrive, ready, s["t0"], s["t1"]) > 0]
        load = sum(overlap(arrive, ready, s["t0"], s["t1"]) for s in own_wakes)
        row = {"id": r["id"], "gpu": r["gpu"], "agent": r["agent"], "turn": r["turn"],
               "history_len": r["history_len"], "hit_tokens": r["hit_tokens"],
               "wait_s": ready - arrive, "offload_s": offload, "load_s": load,
               "other_wait_s": max(0.0, ready - arrive - offload - load),
               "prefill_s": r["first_token_t"] - ready, "decode_s": r["done_t"] - r["first_token_t"],
               "e2e_s": r["done_t"] - arrive, "weights_gib": None, "kv_gib": None,
               "weights_gibs": None, "kv_gibs": None}
        for w in own_wakes:
            rec = next((x for x in tiered.get(name, []) if x["event"] == "resume"
                        and w["t0"] - 0.05 <= x["t_unix"] <= w["t1"]), None)
            if rec:
                row["weights_gib"] = rec["weights_bytes"] / GIB
                row["kv_gib"] = rec["kv_bytes"] / GIB
                wd, kd = rec["weights_end_s"] - rec["weights_start_s"], rec["kv_end_s"] - rec["kv_start_s"]
                row["weights_gibs"] = row["weights_gib"] / wd if wd > 0 else None
                row["kv_gibs"] = row["kv_gib"] / kd if kd > 0 and rec["kv_bytes"] else None
        row["swapped"] = bool(own_wakes)
        row["arrive"], row["ready"], row["done"] = arrive, ready, r["done_t"]
        rows.append(row)

    gpus = sorted({r["gpu"] for r in rows})
    lines = [f"# {os.path.basename(run_dir)}", "",
             f"paths `{cfg.get('paths')}` · GPUs {cfg.get('gpus')} · {cfg.get('copies')} x pool {cfg.get('pool')} per GPU · "
             f"{cfg.get('turns')} turns · input {cfg.get('first_len')} then {cfg.get('turn_len')} tokens · "
             f"output {cfg.get('out_len')} · sync {cfg.get('sync')}"
             + (f" · {', '.join(k + '=' + v for k, v in cfg.items() if k.startswith('tiered_opt_'))}"
                if any(k.startswith('tiered_opt_') for k in cfg) else ""), "",
             f"{len(rows)} requests ok of {len(reqs)}; {sum(r['swapped'] for r in rows)} needed a wake.", ""]
    out = {"run": os.path.basename(run_dir), "config": cfg, "gpus": {}, "turn_bins": {}}

    # GPU timelines on a 1 ms grid
    lines += ["## GPU time by state", "", "| GPU | window s | " + " | ".join(STATES) + " |",
              "|---|---|" + "---|" * len(STATES)]
    for g in gpus:
        mine = [r for r in rows if r["gpu"] == g]
        w0, w1 = min(r["arrive"] for r in rows), max(r["done"] for r in rows)
        grid = np.arange(w0, w1, 0.001)
        state = np.full(len(grid), -1)

        def mark(a, b, k):
            i, j = np.searchsorted(grid, [a, b])
            seg = state[i:j]
            seg[seg == -1] = k

        outstanding = np.zeros(len(grid), bool)
        for r in mine:
            i, j = np.searchsorted(grid, [r["arrive"], r["done"]])
            outstanding[i:j] = True
        for r in mine:
            mark(r["ready"], r["done"], 0)
        for s in swaps:
            if s["gpu"] == g:
                mark(s["t0"], s["t1"], 1 if s["action"] == "wake" else 2)
        state[(state == -1) & ~outstanding] = 3
        state[state == -1] = 4
        frac = {s: float(np.mean(state == k)) for k, s in enumerate(STATES)}
        out["gpus"][g] = {"window_s": w1 - w0, **frac}
        lines.append(f"| {g} | {w1 - w0:.1f} | " + " | ".join(f"{frac[s]:.1%}" for s in STATES) + " |")

    lines += ["", "## Per request, by turn (medians over both GPUs)", "",
              "| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | "
              "weights GiB/s | KV GiB | KV GiB/s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, b in TURN_BINS:
        rs = [r for r in rows if a <= r["turn"] <= b]
        if not rs:
            continue
        col = lambda k: [r[k] for r in rs if r[k] is not None]  # noqa: E731
        stats = {k: p50(col(k)) for k in ("history_len", "hit_tokens", "wait_s", "offload_s", "load_s",
                                          "prefill_s", "decode_s", "weights_gibs", "kv_gib", "kv_gibs")}
        out["turn_bins"][f"{a}-{b}"] = {"n": len(rs), **stats}
        lines.append(f"| {a}-{b} | {len(rs)} | {stats['history_len']:.0f} | {fmt(stats['hit_tokens'], 0)} | "
                     f"{fmt(stats['wait_s'], 3)} | {fmt(stats['offload_s'], 3)} | {fmt(stats['load_s'], 3)} | "
                     f"{fmt(stats['prefill_s'], 3)} | {fmt(stats['decode_s'], 2)} | {fmt(stats['weights_gibs'], 1)} | "
                     f"{fmt(stats['kv_gib'])} | {fmt(stats['kv_gibs'], 1)} |")

    totals = defaultdict(float)
    for r in rows:
        for k in ("offload_s", "load_s", "other_wait_s", "prefill_s", "decode_s"):
            totals[k] += r[k]
    tot = sum(totals.values())
    out["time_split"] = {k: v / tot for k, v in totals.items()}
    lines += ["", "## Where request time goes (sum over all requests)", "",
              "| offload wait | load wait | other wait | prefill | decode |", "|---|---|---|---|---|",
              "| " + " | ".join(f"{totals[k] / tot:.1%}" for k in ("offload_s", "load_s", "other_wait_s",
                                                                  "prefill_s", "decode_s")) + " |"]
    lines += ["", "compute = a request between proxy ready and its last token. load / offload = a wake / "
              "sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). "
              "prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm)."]
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({**out, "requests": rows}, f, indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
