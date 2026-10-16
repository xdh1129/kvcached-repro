"""Does loading weights from DRAM and KV from SSD at the same time overlap on one GPU?

Uses the same stores as the tiered_restore plugin (DRAM pinned buffer, POSIX SSD
through a host bounce buffer, GDS). On one GPU, it times:
  weights alone    W bytes  DRAM -> GPU
  KV alone         K bytes  SSD (or GDS) -> GPU
  both, parallel   two threads, as in a TIERED_PARALLEL=1 wake
  both, sequential weights then KV, as in TIERED_PARALLEL=0
Parallel close to max(alone) means the two transfers overlap; close to the sum means they queue.

usage: CUDA_VISIBLE_DEVICES=<gpu> python scripts/bench_parallel_restore.py
         [--weights-gib 23.4] [--kv-gib 8] [--reps 3] [--root /mnt/raid0/xdh/bench_parallel] [--out file.json]
"""

import argparse
import json
import os
import shutil
import statistics
import threading
import time

import torch

from tiered_restore.stores import make_store

GIB = 1 << 30
MIB = 1 << 20


def regions(base, total, piece):
    out, off = [], 0
    while off < total:
        n = min(piece, total - off)
        out.append((base + off, n, off))
        off += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-gib", type=float, default=23.4)  # one Mistral-Nemo-12B
    ap.add_argument("--kv-gib", type=float, default=8.0)  # a large frozen KV in the trace (7-10.6 GiB)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--root", default="/mnt/raid0/xdh/bench_parallel")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    w_bytes = int(args.weights_gib * GIB) // (2 * MIB) * 2 * MIB
    k_bytes = int(args.kv_gib * GIB) // (2 * MIB) * 2 * MIB
    torch.cuda.set_device(0)
    w_gpu = torch.randint(0, 255, (w_bytes,), dtype=torch.uint8, device="cuda")
    k_gpu = torch.randint(0, 255, (k_bytes,), dtype=torch.uint8, device="cuda")
    # weights: 256 MiB regions; KV: 80 MiB regions (one kvcached compound page in the trace runs)
    w_reg = regions(w_gpu.data_ptr(), w_bytes, 256 * MIB)
    k_reg = regions(k_gpu.data_ptr(), k_bytes, 80 * MIB)

    os.makedirs(args.root, exist_ok=True)
    stores = {p: make_store(p, 0, os.path.join(args.root, p), 8, 64) for p in ("dram", "ssd", "gds")}
    # one-time saves (also pins the DRAM buffer)
    stores["dram"].save("weights", w_reg)
    for p in ("ssd", "gds"):
        stores[p].save("kv", k_reg)
    torch.cuda.synchronize()

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    def load_w():
        stores["dram"].load("weights", w_reg)

    def parallel(kv_path):
        ends = {}

        def run(name, fn):
            fn()
            ends[name] = time.perf_counter()

        t = time.perf_counter()
        ths = [threading.Thread(target=run, args=("w", load_w)),
               threading.Thread(target=run, args=("kv", lambda: stores[kv_path].load("kv", k_reg)))]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        torch.cuda.synchronize()
        return time.perf_counter() - t, ends["w"] - t, ends["kv"] - t

    res = {"gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "weights_gib": w_bytes / GIB, "kv_gib": k_bytes / GIB}
    w_alone = statistics.median(timed(load_w) for _ in range(args.reps))
    res["weights_alone_s"] = w_alone
    for kv_path in ("ssd", "gds"):
        kv_alone = statistics.median(timed(lambda: stores[kv_path].load("kv", k_reg)) for _ in range(args.reps))
        seq = statistics.median(timed(lambda: (load_w(), stores[kv_path].load("kv", k_reg)))
                                for _ in range(args.reps))
        runs = [parallel(kv_path) for _ in range(args.reps)]
        par = statistics.median(r[0] for r in runs)
        res[kv_path] = {
            "kv_alone_s": kv_alone, "sequential_s": seq, "parallel_s": par,
            "parallel_weights_done_s": statistics.median(r[1] for r in runs),
            "parallel_kv_done_s": statistics.median(r[2] for r in runs),
            "max_alone_s": max(w_alone, kv_alone), "sum_alone_s": w_alone + kv_alone,
        }
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    shutil.rmtree(args.root, ignore_errors=True)


if __name__ == "__main__":
    main()
