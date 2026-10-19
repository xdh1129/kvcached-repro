"""Per-piece timing of wakes, for bandwidth-over-time plots (analysis/plot_wake_timeline.py).

Each wake loads weights from DRAM and KV from storage on two threads, as a
TIERED_PARALLEL=1 wake does; the stores' trace lists record every piece moved.
Scenarios (sizes from the Prism 18m runs):
  single     GPU A wakes a Phi-3-mini: 7.8 GB weights + 5 GB KV
  two_gpus   GPU A loads 14.7 GB weights (Mistral-7B); GPU B wakes at the same instant
             with 6.5 GB weights + 3.5 GB KV (the overlap at t=60.7 s of the replication-3 run)
each with the KV through POSIX (ssd) and through GDS (256 MiB staging). Every case runs
--reps times after a barrier; the rep whose wake ends at the median time is kept.

usage: python scripts/bench_wake_timeline.py [--gpus 6 7] [--numa 1] [--reps 5]
         [--out results/bench_wake_timeline/g67.json]
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import statistics
import threading
import time

GB = 1e9
MIB = 1 << 20
ROOT_DIR = "/mnt/raid0/xdh/bench_wake_timeline"


def numa_cpus(node):
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        lo, hi = f.read().strip().split(",")[0].split("-")
    return set(range(int(lo), int(hi) + 1))


def worker(gpu, jobs, reps, numa, barrier, out):
    """jobs: [(label, store, GB)], loaded in parallel on one thread each"""
    os.sched_setaffinity(0, numa_cpus(numa))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TIERED_GDS_STAGE_MB"] = "256"
    import torch

    from tiered_restore.stores import make_store

    torch.cuda.set_device(0)
    loads = []
    for label, store_name, gb in jobs:
        n = int(gb * GB) // (2 * MIB) * 2 * MIB
        buf = torch.randint(0, 255, (n,), dtype=torch.uint8, device="cuda")
        piece = 256 * MIB if label == "weights" else 80 * MIB  # KV: one kvcached compound page
        regions = [(buf.data_ptr() + off, min(piece, n - off), off) for off in range(0, n, piece)]
        store = make_store(store_name, 0, os.path.join(ROOT_DIR, f"gpu{gpu}_{label}"), 8, 64)
        if store_name == "dram":
            store.keep.add(label)
        store.save(label, regions)
        store.load(label, regions)  # warm-up
        loads.append((label, store, regions, buf))
    torch.cuda.synchronize()
    runs = []
    for _ in range(reps):
        for _, store, _, _ in loads:
            store.trace = []
        barrier.wait()
        t0 = time.perf_counter()  # CLOCK_MONOTONIC: comparable across processes
        threads = [threading.Thread(target=s.load, args=(lbl, r)) for lbl, s, r, _ in loads]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        runs.append({"t0": t0, "trace": {lbl: list(s.trace) for lbl, s, _, _ in loads}})
    out.put((gpu, runs))


def run(workers, reps, numa):
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(len(workers)), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(g, jobs, reps, numa, barrier, out)) for g, jobs in workers]
    for p in procs:
        p.start()
    res = dict(out.get() for _ in procs)
    for p in procs:
        p.join()
    shutil.rmtree(ROOT_DIR, ignore_errors=True)
    # per rep: common t0 (earliest start), end = last piece of any GPU; keep the median rep
    reps_out = []
    for i in range(reps):
        t0 = min(res[g][i]["t0"] for g in res)
        rep = {str(g): {lbl: [(s - t0, e - t0, n) for s, e, n in tr] for lbl, tr in res[g][i]["trace"].items()}
               for g in res}
        end = max(e for g in rep.values() for tr in g.values() for _, e, _ in tr)
        reps_out.append((end, rep))
    reps_out.sort(key=lambda x: x[0])
    return reps_out[len(reps_out) // 2][1], [e for e, _ in reps_out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs=2, default=[6, 7])
    ap.add_argument("--numa", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/bench_wake_timeline/g67.json")
    args = ap.parse_args()
    a, b = args.gpus
    result = {"gpus": args.gpus, "single": {}, "two_gpus": {}}
    for kv in ("ssd", "gds"):
        rep, ends = run([(a, [("weights", "dram", 7.8), ("kv", kv, 5.0)])], args.reps, args.numa)
        result["single"][kv] = {"trace": rep, "wake_s": ends}
        print(f"single {kv}: wake s {statistics.median(ends):.3f} (reps {', '.join(f'{e:.3f}' for e in ends)})",
              flush=True)
        rep, ends = run([(a, [("weights", "dram", 14.7)]), (b, [("weights", "dram", 6.5), ("kv", kv, 3.5)])],
                        args.reps, args.numa)
        result["two_gpus"][kv] = {"trace": rep, "wake_s": ends}
        print(f"two_gpus {kv}: both done s {statistics.median(ends):.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
