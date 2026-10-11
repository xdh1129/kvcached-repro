"""Sample GPU memory and KV-cache usage of the instances in a run directory.

Reads <run_dir>/instances.tsv ("name model gpu port") for the instance set and
re-reads <run_dir>/servers.tsv ("name port pgid") every sample, so it can start
before the servers and record model loading too. Writes <run_dir>/vram.csv:
  t                        unix time
  gpu<N>_used_mib          device-wide used memory (NVML), includes other tenants
  gpu<N>_total_mib
  <inst>_proc_mib          physical memory held by the instance's processes
  <inst>_kv_limit_mib      kvcached: KV budget        (empty in static mode)
  <inst>_kv_used_mib       kvcached: KV pages in use  (empty in static mode)
  <inst>_kv_prealloc_mib   kvcached: pre-mapped spare pages
  <inst>_kv_usage_pct      vLLM /metrics kv_cache_usage_perc * 100
  <inst>_running, <inst>_waiting   vLLM scheduler queue depth
  <inst>_preemptions       vLLM cumulative preempted requests

usage: python vram_monitor.py <run_dir> [--interval S]
"""

import argparse
import csv
import os
import re
import signal
import time
import urllib.request

import pynvml

MIB = 1024 * 1024
METRIC_RE = re.compile(
    r"^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc|num_requests_running|num_requests_waiting"
    r"|num_preemptions_total)\{[^}]*\}\s+([0-9.eE+-]+)$",
    re.M,
)


def read_instances(run_dir):
    with open(os.path.join(run_dir, "instances.tsv")) as f:
        return [(name, int(gpu), int(port))
                for name, _, gpu, port in (line.split("\t") for line in f.read().splitlines() if line)]


def read_pgids(run_dir):
    try:
        with open(os.path.join(run_dir, "servers.tsv")) as f:
            return {name: int(pgid) for name, _, pgid in (line.split() for line in f if line.strip())}
    except FileNotFoundError:
        return {}


def read_kvcached_shm(ipc_name):
    try:
        from kvcached.cli.utils import get_kv_cache_limit
    except ImportError:
        return None
    info = get_kv_cache_limit(ipc_name)
    if info is None or info.total_size <= 0:
        return None
    return info


def scrape_metrics(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=0.3) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for key, val in METRIC_RE.findall(text):
        key = "kv_usage" if key.endswith("usage_perc") else key
        out[key] = out.get(key, 0.0) + float(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    instances = read_instances(args.run_dir)
    gpus = sorted({g for _, g, _ in instances})
    pynvml.nvmlInit()
    handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in gpus}

    cols = ["t"]
    for g in gpus:
        cols += [f"gpu{g}_used_mib", f"gpu{g}_total_mib"]
    for name, _, _ in instances:
        cols += [f"{name}_{c}" for c in ("proc_mib", "kv_limit_mib", "kv_used_mib", "kv_prealloc_mib",
                                         "kv_usage_pct", "running", "waiting", "preemptions")]

    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    with open(os.path.join(args.run_dir, "vram.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        while not stop:
            t0 = time.time()
            row = {"t": f"{t0:.3f}"}
            pgids = read_pgids(args.run_dir)
            proc_bytes = {}
            for g, h in handles.items():
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                row[f"gpu{g}_used_mib"] = mem.used // MIB
                row[f"gpu{g}_total_mib"] = mem.total // MIB
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                    try:
                        pgid = os.getpgid(p.pid)
                    except ProcessLookupError:
                        continue
                    if p.usedGpuMemory:
                        proc_bytes[pgid] = proc_bytes.get(pgid, 0) + p.usedGpuMemory

            for name, _, port in instances:
                if name not in pgids:
                    continue
                row[f"{name}_proc_mib"] = proc_bytes.get(pgids[name], 0) // MIB
                info = read_kvcached_shm(f"kvc_{name}_{port}")
                if info is not None:
                    row[f"{name}_kv_limit_mib"] = info.total_size // MIB
                    row[f"{name}_kv_used_mib"] = info.used_size // MIB
                    row[f"{name}_kv_prealloc_mib"] = info.prealloc_size // MIB
                m = scrape_metrics(port)
                if "kv_usage" in m:
                    row[f"{name}_kv_usage_pct"] = round(m["kv_usage"] * 100, 2)
                if "num_requests_running" in m:
                    row[f"{name}_running"] = int(m["num_requests_running"])
                    row[f"{name}_waiting"] = int(m["num_requests_waiting"])
                if "num_preemptions_total" in m:
                    row[f"{name}_preemptions"] = int(m["num_preemptions_total"])

            w.writerow(row)
            f.flush()
            time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
