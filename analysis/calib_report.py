"""Per-tier read/write rates from calibration runs (scripts/run_calib_suite.sh).

For each run dir, reads the tiered_restore logs (tiered_<instance>.jsonl) and reports,
per model and per transfer job of the plugin ("weights" and "kv", or "dram" and "ssd"
for the split path):
  write GiB/s  suspend: job bytes / job save time
  read GiB/s   resume: job bytes / (end_s - start_s) of that job's thread
               (staged GDS, when logged: the time the data really moved, without cleanup)
plus the whole sleep (suspend total_s, incl. unmapping) and wake (resume total_s) times
and the share of the bytes on SSD. "weights" includes the small mutable "state" regions.
Only sleeps/wakes moving > 1 GiB of KV count (not the priming ones), and each model's
first such sleep is left out: it allocates the host buffers / files (one-time cost).
Medians. Works on logs of the serial-save layout (weights_save_s + state_save_s, kv_save_s).
Also writes <run_dir>/calib.json.

usage: python calib_report.py <run_dir> [<run_dir> ...] [--out results/comparison/calib.md]
"""

import argparse
import glob
import json
import os
import statistics

GIB = 1 << 30
SSD_STORES = ("ssd", "gds")


def med(xs):
    return round(statistics.median(xs), 2) if xs else None


def save_jobs(r):
    """{job: (bytes, seconds)} of one suspend record."""
    if "save_s" in r:  # parallel-save layout: <job>_save_bytes / <job>_save_s
        out = {}
        for k, v in r.items():
            if k.endswith("_save_bytes") and v:
                job = k[:-len("_save_bytes")]
                out[job] = (v, r[f"{job}_save_data_end_s"] - r[f"{job}_save_data_start_s"]
                            if f"{job}_save_data_end_s" in r else r[f"{job}_save_s"])
        return out
    out = {}  # serial layout
    if r.get("weights_saved") and r.get("weights_save_s"):
        out["weights"] = (r["weights_bytes"] + r.get("state_bytes", 0), r["weights_save_s"] + r.get("state_save_s", 0))
    if r.get("kv_bytes") and r.get("kv_save_s"):
        out["kv"] = (r["kv_bytes"], r["kv_save_s"])
    return out


def load_jobs(r):
    """{job: (bytes, seconds)} of one resume record."""
    out = {}
    for k, v in r.items():
        job = k[:-len("_end_s")]
        if k.endswith("_end_s") and job + "_bytes" in r:
            # staged GDS: when the data moved; <job>_end_s also holds the cleanup, which can block
            out[job] = (r[job + "_bytes"], r[job + "_data_end_s"] - r[job + "_data_start_s"]
                        if job + "_data_end_s" in r else v - r[job + "_start_s"])
    return out


def ssd_share(r):
    if "split_ssd_bytes" in r:
        return r["split_ssd_bytes"] / r["split_total_bytes"]
    total = r["weights_bytes"] + r.get("state_bytes", 0) + r.get("kv_bytes", 0)
    on_ssd = (r["weights_bytes"] + r.get("state_bytes", 0) if r["weights_path"] in SSD_STORES else 0) \
        + (r.get("kv_bytes", 0) if r["kv_path"] in SSD_STORES else 0)
    return on_ssd / total if total else 0


def run_rates(run_dir):
    cfg = dict(l.split("=", 1) for l in open(os.path.join(run_dir, "config.txt")).read().splitlines() if "=" in l)
    names = {l.split("\t")[0]: l.split("\t")[1] for l in open(os.path.join(run_dir, "instances.tsv")).read().splitlines()}
    out = {"run": os.path.basename(run_dir.rstrip("/")), "paths": cfg.get("paths"),
           "parallel": cfg.get("tiered_parallel"), "parallel_save": cfg.get("tiered_opt_TIERED_PARALLEL_SAVE", "0"),
           "split_frac": cfg.get("tiered_opt_TIERED_SPLIT_FRAC"), "models": {}}
    for f in sorted(glob.glob(os.path.join(run_dir, "tiered_*.jsonl"))):
        inst = os.path.basename(f)[len("tiered_"):-len(".jsonl")]
        recs = [json.loads(l) for l in open(f) if l.strip()]
        write, read, w_gib, kv_gib, share, sleep_s, wake_s = {}, {}, [], [], [], [], []
        first = True
        for r in recs:
            if r.get("kv_bytes", 0) <= GIB:  # priming sleeps/wakes
                continue
            if r["event"] == "suspend":
                if first:
                    first = False
                    continue
                for job, (n, s) in save_jobs(r).items():
                    write.setdefault(job, []).append(n / GIB / s)
                sleep_s.append(r["total_s"])
                share.append(ssd_share(r))
            elif r["event"] == "resume":
                for job, (n, s) in load_jobs(r).items():
                    read.setdefault(job, []).append(n / GIB / s)
                w_gib.append((r["weights_bytes"] if "split_frac" in r else
                              r.get("weights_bytes", 0)) / GIB)
                kv_gib.append(r["kv_bytes"] / GIB)
                wake_s.append(r["total_s"])
        out["models"][names.get(inst, inst)] = {
            "weights_gib": med(w_gib), "kv_gib": med(kv_gib), "ssd_share": med(share),
            "write": {j: med(v) for j, v in sorted(write.items())},
            "read": {j: med(v) for j, v in sorted(read.items())},
            "sleep_s": med(sleep_s), "wake_s": med(wake_s),
            "sleep_wake_s": round(med(sleep_s) + med(wake_s), 2) if sleep_s and wake_s else None,
            "n_sleep": len(sleep_s), "n_wake": len(wake_s)}
    with open(os.path.join(run_dir, "calib.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def fmt(rates):
    return ", ".join(f"{j} {v}" for j, v in rates.items()) or "–"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    lines = ["| run | paths | par. save | f | model | W GiB | KV GiB | on SSD | write GiB/s | read GiB/s | sleep s | wake s | sleep+wake s | n |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in args.run_dirs:
        r = run_rates(d)
        for m, s in r["models"].items():
            lines.append(f"| {r['run']} | {r['paths']} | {r['parallel_save']} | {r['split_frac'] or '–'} | "
                         f"{m.split('/')[-1]} | {s['weights_gib']} | {s['kv_gib']} | "
                         f"{s['ssd_share'] if s['ssd_share'] is None else f'{s['ssd_share']:.0%}'} | "
                         f"{fmt(s['write'])} | {fmt(s['read'])} | {s['sleep_s']} | {s['wake_s']} | "
                         f"{s['sleep_wake_s']} | {s['n_sleep']}/{s['n_wake']} |")
    text = ("Medians. Per job: weights (+ state) / kv, or dram / ssd for the split path. write = sleep save,\n"
            "read = wake load; sleep/wake = whole suspend/resume. Sleeps/wakes with > 1 GiB KV, first one left out.\n\n"
            + "\n".join(lines) + "\n")
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
