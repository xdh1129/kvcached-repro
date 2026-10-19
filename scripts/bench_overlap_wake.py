"""Two wakes at the same moment on GPUs A and B, with real model sizes from the trace.

GPU A wakes a model: its weights come from DRAM (and, optionally, its KV too). GPU B
wakes another model at the same instant: weights from DRAM and KV from storage on
two threads, as a TIERED_PARALLEL=1 wake does. B's KV goes through the POSIX path
(dram+ssd) or GDS with staging (dram+gds). Every rep starts both wakes after a
barrier; the script reports the median duration of each transfer over the reps,
and the same wakes run alone.

usage: python scripts/bench_overlap_wake.py [--gpus 6 7] [--numa 1] [--reps 5]
         [--root /mnt/raid0/xdh/bench_overlap_wake] [--out results/bench_overlap_wake/g67.json]
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

# (name, A weights GB, B weights GB, B KV GB) from the replication-3 balanced reference run
EVENTS = [
    ("t=60.7 s: A model_17 / B model_16", 14.7, 6.5, 3.5),
    ("typical: A Phi-3 / B Phi-3, 5 GB KV", 7.8, 7.5, 5.0),
    ("large KV: A Llama-3.1-8B / B Phi-3, 20 GB KV", 16.7, 7.5, 20.0),
]


def numa_cpus(node):
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        lo, hi = f.read().strip().split(",")[0].split("-")
    return set(range(int(lo), int(hi) + 1))


def worker(gpu, jobs, args, barrier, out):
    """jobs: [(label, store, GB)], loaded in parallel on one thread each"""
    os.sched_setaffinity(0, numa_cpus(args.numa))
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
        store = make_store(store_name, 0, os.path.join(args.root, f"gpu{gpu}_{label}"), 8, 64)
        if store_name == "dram":
            store.keep.add(label)
        store.save(label, regions)
        loads.append((label, store, regions, buf))
    torch.cuda.synchronize()
    times = {label: [] for label, *_ in loads}
    for _ in range(args.reps):
        barrier.wait()
        done = {}

        def load(label, store, regions):
            t = time.perf_counter()
            store.load(label, regions)
            done[label] = time.perf_counter() - t

        threads = [threading.Thread(target=load, args=(lbl, s, r)) for lbl, s, r, _ in loads]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for label in done:
            times[label].append(done[label])
    out.put((gpu, {label: statistics.median(v) for label, v in times.items()}))


def run(workers, args):
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(len(workers)), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(g, jobs, args, barrier, out)) for g, jobs in workers]
    for p in procs:
        p.start()
    res = dict(out.get() for _ in procs)
    for p in procs:
        p.join()
    shutil.rmtree(args.root, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs=2, default=[6, 7])
    ap.add_argument("--numa", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--root", default="/mnt/raid0/xdh/bench_overlap_wake")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    a, b = args.gpus
    results = []
    for name, aw, bw, bk in EVENTS:
        a_jobs = [("weights", "dram", aw)]
        row = {"event": name, "a_weights_gb": aw, "b_weights_gb": bw, "b_kv_gb": bk}
        row["a_alone"] = run([(a, a_jobs)], args)[a]
        for kv in ("ssd", "gds"):
            b_jobs = [("weights", "dram", bw), ("kv", kv, bk)]
            row[f"b_alone_{kv}"] = run([(b, b_jobs)], args)[b]
            both = run([(a, a_jobs), (b, b_jobs)], args)
            row[f"a_with_b_{kv}"], row[f"b_with_a_{kv}"] = both[a], both[b]
        results.append(row)
        print(f"== {name}: A weights {aw} GB | B weights {bw} GB + KV {bk} GB (s, median of {args.reps})")
        print(f"   A weights: alone {row['a_alone']['weights']:.3f}, with B KV ssd {row['a_with_b_ssd']['weights']:.3f}, "
              f"with B KV gds {row['a_with_b_gds']['weights']:.3f}")
        for kv in ("ssd", "gds"):
            al, wi = row[f"b_alone_{kv}"], row[f"b_with_a_{kv}"]
            print(f"   B ({kv}): weights alone {al['weights']:.3f} / with A {wi['weights']:.3f}; "
                  f"KV alone {al['kv']:.3f} / with A {wi['kv']:.3f}; "
                  f"wake alone {max(al.values()):.3f} / with A {max(wi.values()):.3f}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
