"""Cross-GPU overlap: GPU A loads weights from DRAM while GPU B loads KV from storage.

GPUs 6 and 7 share one PCIe switch, and its x16 uplink to the CPU, with the NVMe RAID.
DRAM -> GPU A uses the uplink downstream. KV for GPU B through the POSIX path
(NVMe -> host -> GPU B) also comes down that uplink; through GDS it stays inside
the switch. Each case runs one process per GPU; after a barrier every process
repeats its load for --seconds and reports its GiB/s.

Cases: A weights alone, B KV alone (ssd, gds), A weights + B KV (ssd, gds),
A weights + B weights (both from DRAM: the uplink limit).

usage: python scripts/bench_cross_gpu.py [--gpus 6 7] [--numa 1] [--weights-gib 8] [--kv-gib 8]
         [--seconds 8] [--root /mnt/raid0/xdh/bench_cross_gpu] [--out results/bench_cross_gpu/g67.json]
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time

GIB = 1 << 30
MIB = 1 << 20


def numa_cpus(node):
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        lo, hi = f.read().strip().split(",")[0].split("-")
    return set(range(int(lo), int(hi) + 1))


def worker(gpu, store_name, gib, args, barrier, out):
    os.sched_setaffinity(0, numa_cpus(args.numa))  # host buffers first-touched on the GPUs' NUMA node
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TIERED_GDS_STAGE_MB"] = "256"
    import torch

    from tiered_restore.stores import make_store

    torch.cuda.set_device(0)
    n = int(gib * GIB) // (256 * MIB) * 256 * MIB
    buf = torch.randint(0, 255, (n,), dtype=torch.uint8, device="cuda")
    regions = [(buf.data_ptr() + off, 256 * MIB, off) for off in range(0, n, 256 * MIB)]
    store = make_store(store_name, 0, os.path.join(args.root, f"gpu{gpu}"), 8, 64)
    store.save("x", regions)
    store.load("x", regions)  # warm-up (DRAM: the buffer is dropped after a load, so save again)
    if store_name == "dram":
        store.keep.add("x")
        store.save("x", regions)
    torch.cuda.synchronize()
    barrier.wait()
    t0 = time.perf_counter()
    done = 0
    while time.perf_counter() - t0 < args.seconds:
        store.load("x", regions)
        done += n
    torch.cuda.synchronize()
    out.put((gpu, store_name, done / (time.perf_counter() - t0) / GIB))


def run_case(jobs, args):
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(len(jobs)), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(*j, args, barrier, out)) for j in jobs]
    for p in procs:
        p.start()
    res = [out.get() for _ in procs]
    for p in procs:
        p.join()
    shutil.rmtree(args.root, ignore_errors=True)
    return {f"gpu{g}_{s}": round(bw, 2) for g, s, bw in sorted(res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs=2, default=[6, 7])
    ap.add_argument("--numa", type=int, default=1)
    ap.add_argument("--weights-gib", type=float, default=8)
    ap.add_argument("--kv-gib", type=float, default=8)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--root", default="/mnt/raid0/xdh/bench_cross_gpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    a, b = args.gpus
    w, k = ("dram", args.weights_gib), args.kv_gib
    cases = {
        "A weights (DRAM) alone": [(a, *w)],
        "B weights (DRAM) alone": [(b, *w)],
        "B KV (ssd) alone": [(b, "ssd", k)],
        "B KV (gds) alone": [(b, "gds", k)],
        "A weights (DRAM) + B KV (ssd)": [(a, *w), (b, "ssd", k)],
        "A weights (DRAM) + B KV (gds)": [(a, *w), (b, "gds", k)],
        "A weights (DRAM) + B weights (DRAM)": [(a, *w), (b, *w)],
    }
    results = {}
    for name, jobs in cases.items():
        results[name] = run_case(jobs, args)
        print(f"{name:40s} " + "  ".join(f"{k} {v:.1f} GiB/s" for k, v in results[name].items()), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
