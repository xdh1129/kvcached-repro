"""Sample PCIe traffic and busy time of a set of GPUs, and NVMe throughput.

DCGM's profiling counters need nv-hostengine, which is not running on this shared
host, so this uses NVML and /sys/block instead:
  gpu<N>_rx_mbs, gpu<N>_tx_mbs  PCIe RX (host/NVMe -> GPU) and TX, MB/s
                                (nvmlDeviceGetPcieThroughput: NVML averages over ~20 ms)
  gpu<N>_util                   nvmlDeviceGetUtilizationRates().gpu: % of the last
                                sample period with any kernel running (not SM occupancy)
  gpu<N>_used_mib               device-wide used memory
  <dev>_read_mbs, <dev>_write_mbs   block-layer throughput of each NVMe / md device,
                                from /sys/block/<dev>/stat (GDS reads show up here too)
Writes <run_dir>/pcie.csv with t = unix time of the sample.

Each nvmlDeviceGetPcieThroughput call blocks ~24 ms, so one loop over 2 GPUs x 2
directions takes ~100 ms. --nvml-threads samples each (GPU, direction) on its own thread
(still ~24 ms each, in parallel) into <run_dir>/pcie_nvml.csv (t, gpu, dir, mbs; t = when
the call returned), and the main loop, which can then run at e.g. --interval 0.01, copies
the latest value into pcie.csv. Phoenix reads bypass md0: they show on the members only.

usage: python pcie_monitor.py <run_dir> --gpus 6 7 [--interval 0.1] [--nvml-threads]
                              [--block nvme1n1 nvme2n1 md0]
"""

import argparse
import csv
import os
import signal
import threading
import time

import pynvml

MIB = 1024 * 1024


def block_sectors(dev):
    """(sectors read, sectors written) of a block device; sectors are 512 B."""
    with open(f"/sys/block/{dev}/stat") as f:
        v = f.read().split()
    return int(v[2]), int(v[6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--block", nargs="*", default=["nvme1n1", "nvme2n1", "md0"])
    ap.add_argument("--nvml-threads", action="store_true",
                    help="sample PCIe rx/tx on one thread per GPU and direction (pcie_nvml.csv)")
    args = ap.parse_args()

    pynvml.nvmlInit()
    handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in args.gpus}
    devs = [d for d in args.block if os.path.exists(f"/sys/block/{d}/stat")]
    cols = ["t"]
    for g in args.gpus:
        cols += [f"gpu{g}_rx_mbs", f"gpu{g}_tx_mbs", f"gpu{g}_util", f"gpu{g}_used_mib"]
    for d in devs:
        cols += [f"{d}_read_mbs", f"{d}_write_mbs"]

    stop = False

    def on_term(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    latest = {(g, d): 0.0 for g in args.gpus for d in ("rx", "tx")}  # MB/s
    nvml_threads = []
    if args.nvml_threads:
        nf = open(os.path.join(args.run_dir, "pcie_nvml.csv"), "w", newline="")
        nw, nlock = csv.writer(nf), threading.Lock()
        nw.writerow(["t", "gpu", "dir", "mbs"])
        kinds = {"rx": pynvml.NVML_PCIE_UTIL_RX_BYTES, "tx": pynvml.NVML_PCIE_UTIL_TX_BYTES}

        def sampler(g, d):
            while not stop:
                v = round(pynvml.nvmlDeviceGetPcieThroughput(handles[g], kinds[d]) / 1000, 1)  # KB/s -> MB/s
                t = time.time()
                latest[(g, d)] = v
                with nlock:
                    nw.writerow([round(t, 4), g, d, v])

        nvml_threads = [threading.Thread(target=sampler, args=k, daemon=True) for k in latest]
        for th in nvml_threads:
            th.start()

    prev_t, prev = time.time(), {d: block_sectors(d) for d in devs}
    with open(os.path.join(args.run_dir, "pcie.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        next_t = time.time()
        while not stop:
            row = []
            for g, h in handles.items():
                if args.nvml_threads:
                    rx, tx = latest[(g, "rx")], latest[(g, "tx")]
                else:
                    rx = round(pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1000, 1)
                    tx = round(pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1000, 1)
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                used = pynvml.nvmlDeviceGetMemoryInfo(h).used // MIB
                row += [rx, tx, util, used]
            t = time.time()
            dt = max(t - prev_t, 1e-6)
            for d in devs:
                r, wr = block_sectors(d)
                row += [round((r - prev[d][0]) * 512 / 1e6 / dt, 1), round((wr - prev[d][1]) * 512 / 1e6 / dt, 1)]
                prev[d] = (r, wr)
            prev_t = t
            w.writerow([round(t, 4)] + row)
            f.flush()
            if args.nvml_threads:
                with nlock:
                    nf.flush()
            next_t += args.interval
            time.sleep(max(0.0, next_t - time.time()))


if __name__ == "__main__":
    main()
