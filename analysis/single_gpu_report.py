"""Compare offload paths of the single-GPU closed-loop runs (scripts/run_single_gpu_suite.sh).

From each run's tiered_<instance>.jsonl, requests.jsonl and proxy_requests.jsonl:
  - whole run: workload time, requests ok, sum of sleep and wake time, pinning time, swap wait
  - per model and turn range: median sleep and wake, pinning listed separately
  - split runs: share of bytes on SSD, and whether the two jobs end together
Sleeps: suspend n >= 1 is the sleep after the instance's turn n (n = 0 is the sleep at startup).
Wakes: the instance's k-th resume brings it back for turn k.
"sleep" is suspend total_s (save + unmap); "pin" is the part of it spent pinning host DRAM.

usage: python single_gpu_report.py <out.md> <run_dir> [<run_dir> ...]
"""

import json
import os
import statistics as st
import sys

GIB = 1 << 30
RANGES = [(2, 5), (6, 10), (11, 15)]
NAMES = {"g7_a1": "Qwen3-32B", "g7_a2": "Qwen3-14B", "g7_a3": "Phi-3.5"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pin_s(r):
    return sum(v for k, v in r.items() if k.endswith("_save_alloc_s"))


def med(xs):
    return st.median(xs) if xs else float("nan")


def analyze(d):
    cfg = dict(l.split("=", 1) for l in open(f"{d}/config.txt").read().splitlines() if "=" in l)
    reqs = load(f"{d}/requests.jsonl")
    proxy = load(f"{d}/proxy_requests.jsonl")
    out = {"name": os.path.basename(d.rstrip("/")), "paths": cfg["paths"],
           "workload_s": float(cfg["workload_end"]) - float(cfg["workload_start"]),
           "ok": sum(r["ok"] for r in reqs), "n": len(reqs),
           "swap_wait_s": sum(p.get("swap_wait_s") or 0 for p in proxy), "inst": {}, "split": []}
    for inst in NAMES:
        recs = load(f"{d}/tiered_{inst}.jsonl")
        sus = [r for r in recs if r["event"] == "suspend" and r["n"] >= 1]
        res = [r for r in recs if r["event"] == "resume"]
        out["inst"][inst] = {"sus": sus, "res": res}
        for r in sus + res:
            if "dram_bytes" in r or "dram_save_bytes" in r:
                out["split"].append((inst, r))
    all_sus = [r for v in out["inst"].values() for r in v["sus"]]
    all_res = [r for v in out["inst"].values() for r in v["res"]]
    out["sleep_sum"], out["wake_sum"] = sum(r["total_s"] for r in all_sus), sum(r["total_s"] for r in all_res)
    out["pin_sum"], out["pin_max"] = sum(map(pin_s, all_sus)), max(map(pin_s, all_sus))
    out["ttft"] = {rng: med([r["first_token_t"] - r["sent_t"] for r in reqs if rng[0] <= r["turn"] <= rng[1]])
                   for rng in RANGES}
    return out


def main(out_path, dirs):
    runs = [analyze(d) for d in dirs]
    L = ["## Whole run", "",
         "| path | requests ok | workload s | sum sleep s | of which pin s (max one) | sum wake s | sleep+wake excl. pin | proxy swap wait s |",
         "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['paths']} | {r['ok']}/{r['n']} | {r['workload_s']:.0f} | {r['sleep_sum']:.1f} | "
                 f"{r['pin_sum']:.1f} ({r['pin_max']:.2f}) | {r['wake_sum']:.1f} | "
                 f"{r['sleep_sum'] - r['pin_sum'] + r['wake_sum']:.1f} | {r['swap_wait_s']:.1f} |")
    L += ["", "## Median time to first token by turn range (s)", "",
          "| path | " + " | ".join(f"turns {a}-{b}" for a, b in RANGES) + " |", "|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['paths']} | " + " | ".join(f"{r['ttft'][x]:.2f}" for x in RANGES) + " |")
    for inst, label in NAMES.items():
        L += ["", f"## {label}: median sleep excl. pin / wake (s) by turn range", "",
              "| path | " + " | ".join(f"turns {a}-{b}: sleep / wake" for a, b in RANGES) + " | KV GiB at last sleep |",
              "|---|---|---|---|---|"]
        for r in runs:
            cells = []
            v = r["inst"][inst]
            for a, b in RANGES:
                s = med([x["total_s"] - pin_s(x) for x in v["sus"] if a <= x["n"] <= b])
                w = med([x["total_s"] for i, x in enumerate(v["res"], 1) if a <= i <= b])
                cells.append(f"{s:.2f} / {w:.2f}")
            L.append(f"| {r['paths']} | " + " | ".join(cells) + f" | {v['sus'][-1]['kv_bytes'] / GIB:.1f} |")
    for r in runs:
        if not r["split"]:
            continue
        L += ["", f"## Split detail: {r['name']}", "",
              "Median over the turn range. gap = SSD data end minus DRAM job end (positive: the SSD finishes last).", "",
              "| model | turns | on SSD GiB (share) | SSD weights GiB | sleep: DRAM end, SSD data end, gap (s) | wake: DRAM end, SSD data end, gap (s) |",
              "|---|---|---|---|---|---|"]
        for inst, label in NAMES.items():
            v = r["inst"][inst]
            for a, b in RANGES:
                sus = [x for x in v["sus"] if a <= x["n"] <= b]
                res = [x for i, x in enumerate(v["res"], 1) if a <= i <= b]
                ssd = med([x["split_ssd_bytes"] / GIB for x in sus])
                share = med([x["split_ssd_bytes"] / x["split_total_bytes"] for x in sus])
                ws = med([x.get("split_ssd_weights_bytes", 0) / GIB for x in sus])
                def line(rs, d_end, s_end):
                    de = med([x[d_end] for x in rs]); se = med([x[s_end] for x in rs])
                    return f"{de:.2f}, {se:.2f}, {se - de:+.2f}"
                L.append(f"| {label} | {a}-{b} | {ssd:.1f} ({share:.0%}) | {ws:.1f} | "
                         f"{line(sus, 'dram_save_end_s', 'ssd_save_data_end_s')} | {line(res, 'dram_end_s', 'ssd_data_end_s')} |")
    text = "\n".join(L) + "\n"
    print(text)
    open(out_path, "w").write(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
