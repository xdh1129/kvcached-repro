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

usage: python pcie_monitor.py <run_dir> --gpus 6 7 [--interval 0.1] [--block nvme1n1 nvme2n1 md0]
"""

import argparse
import csv
import os
import signal
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

    prev_t, prev = time.time(), {d: block_sectors(d) for d in devs}
    with open(os.path.join(args.run_dir, "pcie.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        next_t = time.time()
        while not stop:
            row = []
            for g, h in handles.items():
                rx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES)  # KB/s
                tx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                used = pynvml.nvmlDeviceGetMemoryInfo(h).used // MIB
                row += [round(rx / 1000, 1), round(tx / 1000, 1), util, used]
            t = time.time()
            dt = max(t - prev_t, 1e-6)
            for d in devs:
                r, wr = block_sectors(d)
                row += [round((r - prev[d][0]) * 512 / 1e6 / dt, 1), round((wr - prev[d][1]) * 512 / 1e6 / dt, 1)]
                prev[d] = (r, wr)
            prev_t = t
            w.writerow([round(t, 3)] + row)
            f.flush()
            next_t += args.interval
            time.sleep(max(0.0, next_t - time.time()))


if __name__ == "__main__":
    main()
