"""Do several processes loading from the NVMe RAID at the same time slow each other down?

Each worker process owns one GPU buffer (weights-like regions, 24-400 MiB) and one
store file, written before the timed part. All workers then load (or save) at
once, after a barrier, through the same stores as the tiered_restore plugin.
Reports each worker's GiB/s and the aggregate (total bytes / wall time).

usage: python scripts/bench_gds_multiproc.py --gpus 1 2 [--store gds|ssd] [--stage-mb 256]
         [--gib 7.3] [--op load|save] [--reps 3] [--root /mnt/raid0/xdh/bench_multiproc] [--out file.json]
  --gpus 1 1 runs two workers on the same GPU.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import time

GIB = 1 << 30
MIB = 1 << 20


def worker(i, gpu, args, barrier, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TIERED_GDS_STAGE_MB"] = str(args.stage_mb)
    import torch

    from tiered_restore.stores import make_store

    torch.cuda.set_device(0)
    rng = random.Random(i)
    sizes, total = [], int(args.gib * GIB)
    while sum(sizes) < total // 2:
        sizes.append(rng.choice([24, 32, 40, 48]) * MIB)
    while sum(sizes) < total:
        sizes.append(rng.choice([96, 192, 400]) * MIB)
    buf = torch.randint(0, 255, (sum(sizes),), dtype=torch.uint8, device="cuda")
    regions, off = [], 0
    for n in sizes:
        regions.append((buf.data_ptr() + off, n, off))
        off += n
    store = make_store(args.store, 0, os.path.join(args.root, f"w{i}"), args.threads, 64)
    store.save("weights", regions)
    torch.cuda.synchronize()
    fn = store.load if args.op == "load" else store.save
    times = []
    for _ in range(args.reps):
        barrier.wait()
        t = time.perf_counter()
        fn("weights", regions)
        torch.cuda.synchronize()
        times.append((t, time.perf_counter()))
    out.put((i, gpu, off, times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--store", default="gds", choices=["gds", "ssd"])
    ap.add_argument("--stage-mb", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gib", type=float, default=7.3)
    ap.add_argument("--op", default="load", choices=["load", "save"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--root", default="/mnt/raid0/xdh/bench_multiproc")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(len(args.gpus)), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, g, args, barrier, out)) for i, g in enumerate(args.gpus)]
    for p in procs:
        p.start()
    res = sorted(out.get() for _ in procs)
    for p in procs:
        p.join()
    shutil.rmtree(args.root, ignore_errors=True)

    reps = []
    for r in range(args.reps):
        start = min(t[r][0] for _, _, _, t in res)
        end = max(t[r][1] for _, _, _, t in res)
        per = [n / (t[r][1] - t[r][0]) / GIB for _, _, n, t in res]
        reps.append({"aggregate_gibs": sum(n for _, _, n, _ in res) / (end - start) / GIB, "per_worker_gibs": per})
    best = max(reps, key=lambda x: x["aggregate_gibs"])
    print(f"{args.store} stage={args.stage_mb} {args.op} gpus={args.gpus}: aggregate {best['aggregate_gibs']:.1f} GiB/s, "
          f"per worker {' '.join(f'{x:.1f}' for x in best['per_worker_gibs'])}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "reps": reps}, f, indent=1)


if __name__ == "__main__":
    main()
