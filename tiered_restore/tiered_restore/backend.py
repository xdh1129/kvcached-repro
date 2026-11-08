"""Tiered sleep-mode backend: weights and KV each go to DRAM, SSD or GDS.

Defaults follow vLLM's own sleep mode; each departure from it is opt-in (env):
  TIERED_WEIGHTS_SAVE  every (default): the whole cumem "weights" pool is saved on
                       every sleep, as vLLM's cumem backend does. once: split the pool
                       at the first sleep; static regions (parameters/buffers only)
                       are saved once, mutable ones (runner/sampler state) every sleep.
  TIERED_KV_MERGE=1    merge consecutive kvcached pages into one region (fewer,
                       larger I/Os); default 0: one region per page.
  TIERED_WEIGHTS_PACE=1  pace DRAM weights during a parallel wake with GDS KV.
  TIERED_PARALLEL_SAVE=1  sleep saves the weights and the KV on two threads at once
                       (default 0: weights, then KV), as TIERED_PARALLEL does for wakes.
  TIERED_TRACE=1       resume records carry per-piece transfer times (diagnostics).
  TIERED_SPLIT_FRAC=f  split placement (needs TIERED_WEIGHTS=dram, TIERED_KV=gds|ssd):
                       instead of weights -> DRAM and KV -> SSD, the bytes of one sleep
                       are laid out in order weights, state, KV pages; the first
                       1 - f of them go to DRAM, the last f (rounded down to 2 MiB) to
                       the KV path's SSD store, so a weight-heavy model may put weights
                       on SSD and a KV-heavy one keeps part of its KV in DRAM. The two
                       parts move on two threads ("dram", "ssd" jobs), in parallel when
                       TIERED_PARALLEL / TIERED_PARALLEL_SAVE are on. Recomputed every
                       sleep (the KV grows). Not with TIERED_WEIGHTS_SAVE=once.

suspend(level=1):
  1. Weights go to the weights path (see TIERED_WEIGHTS_SAVE).
  2. KV: the in-use kvcached pages (pages holding blocks of the frozen
     requests) are saved to the KV store and unmapped. With TIERED_KV=recompute
     nothing is saved: the requests were preempted and re-prefill on wake.
  3. Every cumem allocation is unmapped and released.
resume():
  1. Physical memory is mapped back at the same virtual addresses (cumem
     handles, kvcached page offsets), so CUDA graphs and block tables stay valid.
  2. Weights and KV are loaded on two threads at the same time (TIERED_PARALLEL=1),
     or one after the other.

Each suspend/resume appends one JSON line to TIERED_LOG with the bytes and
times of each path, and with host_bytes: how many of those bytes passed
through host DRAM (DRAM store, or the SSD bounce buffer; 0 for GDS).
"""

import bisect
import gc
import itertools
import json
import os
import threading
import time

import torch
from torch._subclasses.fake_tensor import FakeTensor
from vllm.device_allocator.cumem import CuMemAllocator, create_and_map, unmap_and_release
from vllm.device_allocator.sleep_mode_backend import SleepModeBackend
from vllm.logger import init_logger

from . import KV_TENSORS
from .stores import make_store

logger = init_logger("vllm.tiered_restore")


def _env(name, default):
    return os.environ.get(name, default)


class _CudaBytes:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def _gpu_bytes(ptr, n) -> torch.Tensor:
    """uint8 tensor view of raw GPU memory (no copy)."""
    return torch.as_tensor(_CudaBytes(ptr, n), device="cuda")


def _groups(regions, size):
    """Split regions into consecutive groups of about `size` bytes (a region is never split)."""
    out, cur, n = [], [], 0
    for r in regions:
        cur.append(r)
        n += r[1]
        if n >= size:
            out.append(cur)
            cur, n = [], 0
    return out + ([cur] if cur else [])


SPLIT_ALIGN = 2 << 20  # cumem / kvcached granularity: cut points stay O_DIRECT-aligned


def _split_parts(parts, ssd_bytes):
    """Cut [(kind, regions)] at total - ssd_bytes: (head parts, tail parts), order kept.

    A region crossing the cut becomes two regions with the same store offsets it had,
    so each kind keeps one flat byte space per store (the tail's file is sparse below it).
    """
    left = sum(n for _, regions in parts for _, n, _ in regions) - ssd_bytes
    head, tail = [], []
    for kind, regions in parts:
        h, t = [], []
        for ptr, n, off in regions:
            if left >= n:
                h.append((ptr, n, off))
                left -= n
            elif left > 0:
                h.append((ptr, left, off))
                t.append((ptr + left, n - left, off + left))
                left = 0
            else:
                t.append((ptr, n, off))
        if h:
            head.append((kind, h))
        if t:
            tail.append((kind, t))
    return head, tail


def _checksum(ptr, n) -> tuple[int, int]:
    """Plain and position-weighted sums of a region's 32-bit words (detects changed or shifted bytes)."""
    w = _gpu_bytes(ptr, n).view(torch.int32)
    total, weighted = 0, 0
    step = 32 << 20
    for i in range(0, w.numel(), step):
        c = w[i:i + step].to(torch.int64)
        total += int(c.sum())
        weighted += int((c * torch.arange(i % 9973, i % 9973 + c.numel(), device=c.device, dtype=torch.int64)
                         .remainder_(9973).add_(1)).sum())
    return total, weighted


class TieredBackend(SleepModeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.weights_path = _env("TIERED_WEIGHTS", "dram")
        self.kv_path = _env("TIERED_KV", "dram")
        self.parallel = _env("TIERED_PARALLEL", "1") == "1"
        self.parallel_save = _env("TIERED_PARALLEL_SAVE", "0") == "1"
        self.save_once = _env("TIERED_WEIGHTS_SAVE", "every") == "once"
        self.split_frac = float(_env("TIERED_SPLIT_FRAC", "0") or 0)
        if self.split_frac and (self.weights_path != "dram" or self.kv_path not in ("gds", "ssd")
                                or self.save_once or not 0 < self.split_frac < 1):
            raise ValueError("TIERED_SPLIT_FRAC needs 0 < f < 1, TIERED_WEIGHTS=dram, TIERED_KV=gds|ssd "
                             "and weights saved every sleep")
        self.load_jobs = {}  # job name -> (store, [(kind, regions)]) restored by the next wake
        self.kv_merge = _env("TIERED_KV_MERGE", "0") == "1"
        self.root = _env("TIERED_DIR", f"/mnt/raid0/xdh/tiered/{os.getpid()}")
        self.log_path = _env("TIERED_LOG", "")
        self.device = torch.cuda.current_device()
        threads, chunk_mb = int(_env("TIERED_IO_THREADS", "8")), int(_env("TIERED_CHUNK_MB", "64"))
        self.stores = {}
        for path in {self.weights_path, self.kv_path} - {"recompute"}:
            self.stores[path] = make_store(path, self.device, self.root, threads, chunk_mb)
        if self.save_once and self.weights_path == "dram":
            self.stores["dram"].keep.add("weights")  # saved once: the host copy must outlive each wake
        # [(gpu_ptr, nbytes, store_offset)] of the weights pool: (static, mutable) with save-once,
        # (whole pool, []) otherwise
        self.weight_regions = self.state_regions = None
        self.kv_saved = []  # [(group_id, offsets, regions)] of the current sleep
        self.n_suspend = 0
        # TIERED_VERIFY=1: checksum every weight and KV region before release and after load.
        self.verify = _env("TIERED_VERIFY", "0") == "1"
        self.sums = {}

    # -- KV pages owned by kvcached --

    def _kv_pages(self):
        """(group_id, logical offsets, regions) of every in-use kvcached page."""
        from kvcached.pool_registry import get_registered_kv_cache_pools
        from kvcached.utils import CONTIGUOUS_LAYOUT

        if not CONTIGUOUS_LAYOUT:
            raise NotImplementedError("tiered backend supports kvcached's contiguous KV layout only")
        out, store_off = [], 0
        for mgr, _ in get_registered_kv_cache_pools():
            pages = sorted(set(mgr.avail_pages) | set(mgr.full_pages))
            if not pages:
                continue
            compound = mgr.page_size * mgr.num_layers * mgr.num_kv_buffers
            base = KV_TENSORS[mgr.group_id][0].data_ptr()
            offsets = [pid * compound for pid in pages]
            # TIERED_KV_MERGE=1: consecutive pages are contiguous both in GPU memory and in the
            # store, so they merge into one region (GDS: ~12 GiB/s on >= 256 MiB I/Os, ~7 on 80 MiB).
            regions = []
            for off in offsets:
                if self.kv_merge and regions and regions[-1][0] + regions[-1][1] == base + off:
                    ptr, n, soff = regions[-1]
                    regions[-1] = (ptr, n + compound, soff)
                else:
                    regions.append((base + off, compound, store_off))
                store_off += compound
            out.append((mgr.group_id, offsets, regions))
        return out

    # -- suspend / resume --

    def _split_weight_pool(self, regions):
        """Split the "weights" cumem pool into static and mutable regions.

        vLLM tags everything allocated during load_model as "weights". Besides the
        parameters this includes runner/sampler state that changes while serving
        (e.g. a (max_num_seqs, vocab) int32 buffer). A region is static when every
        live CUDA tensor inside it is an nn.Module parameter or buffer.
        """
        size = dict((p, n) for p, n, _ in regions)
        starts = sorted(size)
        gc.unfreeze()  # vLLM gc.freeze()s after startup; frozen objects are invisible to get_objects()
        try:
            objs = gc.get_objects()
            module_ptrs = {t.untyped_storage().data_ptr() for o in objs if isinstance(o, torch.nn.Module)
                           for t in itertools.chain(o.parameters(recurse=False), o.buffers(recurse=False))
                           if not isinstance(t, FakeTensor)}
            mutable = set()
            for o in objs:
                # FakeTensors (torch.compile tracing leftovers) own no memory
                if isinstance(o, torch.Tensor) and o.is_cuda and not isinstance(o, FakeTensor):
                    p = o.untyped_storage().data_ptr()
                    i = bisect.bisect_right(starts, p) - 1
                    if p not in module_ptrs and i >= 0 and p < starts[i] + size[starts[i]]:
                        mutable.add(starts[i])
        finally:
            gc.freeze()
        static = [(p, size[p]) for p in starts if p not in mutable]
        return static, [(p, size[p]) for p in starts if p in mutable]

    @staticmethod
    def _layout(pairs):
        regions, off = [], 0
        for ptr, n in pairs:
            regions.append((ptr, n, off))
            off += n
        return regions

    # -- suspend / resume --

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("tiered backend implements sleep level 1 only")
        from kvcached import vmm_ops

        self._state = "SUSPENDED"
        rec = {"event": "suspend", "t_unix": time.time(), "n": self.n_suspend,
               "weights_path": self.weights_path, "kv_path": self.kv_path}
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        alloc = CuMemAllocator.get_instance()
        live = [(ptr, d) for ptr, d in alloc.pointer_to_data.items() if not d.is_asleep]
        wstore = self.stores[self.weights_path]

        pool = self._layout(sorted((ptr, d.handle[1]) for ptr, d in live if d.tag == "weights"))
        if not self.save_once:
            self.weight_regions, self.state_regions = pool, []  # whole pool, every sleep (vLLM)
        elif self.weight_regions is None:  # save-once: split at the first sleep
            static, mutable = self._split_weight_pool(pool)
            self.weight_regions, self.state_regions = self._layout(static), self._layout(mutable)
        save_weights = not self.save_once or not self.n_suspend
        rec["weights_saved"] = save_weights
        rec["weights_bytes"] = sum(n for _, n, _ in self.weight_regions)
        sizes = sorted(n for _, n, _ in self.weight_regions)
        rec.update(weights_regions=len(sizes),
                   weights_region_mib_p50=sizes[len(sizes) // 2] / (1 << 20) if sizes else 0,
                   weights_bytes_in_regions_lt_64mib=sum(n for n in sizes if n < 64 << 20),
                   state_regions=len(self.state_regions),
                   state_bytes=sum(n for _, n, _ in self.state_regions))

        # recompute: vLLM's cache reset already preempted the frozen requests; nothing to keep
        self.kv_saved = [] if self.kv_path == "recompute" else self._kv_pages()
        kv_regions = [r for _, _, regions in self.kv_saved for r in regions]
        rec.update(kv_pages=len(kv_regions), kv_bytes=sum(n for _, n, _ in kv_regions))

        w_parts = [("weights", self.weight_regions), ("state", self.state_regions)]
        kv_parts = [("kv", kv_regions)]
        if self.split_frac:
            total = sum(n for _, regions in w_parts + kv_parts for _, n, _ in regions)
            ssd = int(self.split_frac * total) // SPLIT_ALIGN * SPLIT_ALIGN
            head, tail = _split_parts(w_parts + kv_parts, ssd)
            save_jobs = self.load_jobs = {"dram": (self.stores["dram"], head),
                                          "ssd": (self.stores[self.kv_path], tail)}
            rec.update(split_frac=self.split_frac, split_total_bytes=total, split_ssd_bytes=ssd,
                       **{f"split_ssd_{kind}_bytes": sum(n for _, n, _ in regions) for kind, regions in tail})
        else:
            save_jobs = {"weights": (wstore, w_parts if save_weights else w_parts[1:]),
                         "kv": (self.stores[self.kv_path], kv_parts)}
            self.load_jobs = {"weights": (wstore, w_parts), "kv": (self.stores[self.kv_path], kv_parts)}
        rec["parallel_save"] = self.parallel_save
        t = time.perf_counter()
        results = self._run_jobs(save_jobs, to_gpu=False, parallel=self.parallel_save, t0=t)
        rec["save_s"] = time.perf_counter() - t
        for job, st in results.items():
            rec.update({f"{job}_save_{k}": v for k, v in st.items()})
            rec[f"{job}_save_s"] = st["end_s"] - st["start_s"]
        if not self.split_frac:  # fields of the serial layout, kept for the analysis scripts
            rec.update(state_save_s=0.0, state_save_host_bytes=0)
        if self.verify:
            self.sums = {r[0]: _checksum(*r[:2]) for r in self.weight_regions + self.state_regions + kv_regions}

        t = time.perf_counter()
        for group_id, offsets, _ in self.kv_saved:
            if not vmm_ops.unmap_from_kv_tensors(offsets, group_id=group_id):
                raise RuntimeError(f"kvcached unmap failed for group {group_id}")
        released = 0
        for _, d in live:
            unmap_and_release(d.handle)
            d.is_asleep = True
            released += d.handle[1]
        gc.collect()
        torch.cuda.empty_cache()
        rec.update(release_s=time.perf_counter() - t, cumem_released_bytes=released,
                   total_s=time.perf_counter() - t0)
        self.n_suspend += 1
        self._log(rec)

    def resume(self, tags: list[str] | None = None) -> None:
        if tags is not None and set(tags) != {"weights", "kv_cache"}:
            raise ValueError("tiered backend wakes everything at once (tags=None)")
        from kvcached import vmm_ops

        self._state = "RESUMING"
        rec = {"event": "resume", "t_unix": time.time(), "n": self.n_suspend - 1,
               "weights_path": self.weights_path, "kv_path": self.kv_path, "parallel": self.parallel}
        t0 = time.perf_counter()
        alloc = CuMemAllocator.get_instance()
        for d in alloc.pointer_to_data.values():
            if d.is_asleep:
                create_and_map(d.handle)
                d.is_asleep = False
        for group_id, offsets, _ in self.kv_saved:
            if not vmm_ops.map_to_kv_tensors(offsets, group_id=group_id):
                raise RuntimeError(f"kvcached map failed for group {group_id}")
        rec["map_s"] = time.perf_counter() - t0

        kv_regions = [r for _, _, regions in self.kv_saved for r in regions]
        # job name -> (store, [(kind, regions)]), as laid out by the last sleep
        jobs = dict(self.load_jobs)
        # Negative control for the correctness test: zero the KV pages instead of loading them.
        skip_kv = os.path.exists(os.path.join(self.root, "DEBUG_SKIP_KV_LOAD"))
        rec["debug_skip_kv_load"] = skip_kv
        if skip_kv:
            for ptr, n, _ in kv_regions:
                _gpu_bytes(ptr, n).zero_()
            jobs = {job: (store, [(kind, regions) for kind, regions in parts if kind != "kv"])
                    for job, (store, parts) in jobs.items()}
        has_kv = any(kind == "kv" and regions for _, parts in jobs.values() for kind, regions in parts)
        pace = self._weights_pace(kv_regions) if has_kv and self.parallel and not self.split_frac else None
        rec["weights_pace_gibs"] = pace
        if self.split_frac:
            rec.update(split_frac=self.split_frac, weights_bytes=sum(n for _, n, _ in self.weight_regions),
                       state_bytes=sum(n for _, n, _ in self.state_regions),
                       kv_bytes=sum(n for _, n, _ in kv_regions))
        trace = _env("TIERED_TRACE", "0") == "1"
        if trace:
            for store in self.stores.values():
                store.trace = []
                if hasattr(store, "stage_trace"):
                    store.stage_trace = []
        results = self._run_jobs(jobs, to_gpu=True, parallel=self.parallel, t0=t0, pace=pace)
        if trace:
            rec["trace"] = {name: [(round(a - t0, 5), round(b - t0, 5), n) for a, b, n in store.trace]
                            for name, store in self.stores.items()}
            for name, store in self.stores.items():
                if getattr(store, "stage_trace", None):
                    rec["stage_trace"] = [[round(x - t0, 5) for x in row[:4]] + [row[4]] for row in store.stage_trace]
                store.trace = None
        torch.cuda.synchronize()
        rec["total_s"] = time.perf_counter() - t0  # before the optional checksums
        if self.verify:
            rec["verify"] = self._verify(kv_regions)
        for path, st in results.items():
            rec.update({f"{path}_{k}": v for k, v in st.items()})
        self.kv_saved = []
        self._state = "RUNNING"
        self._log(rec)

    def _run_jobs(self, jobs, to_gpu, parallel, t0, pace=None):
        """Save (to_gpu=False) or load every job's parts: one thread per job when parallel.

        Returns {job: {bytes, host_bytes, store counters..., start_s, end_s}} (times since t0)
        for the jobs that moved anything. pace (GiB/s, loads only) paces the "weights" job.
        """
        results, errors = {}, []

        def run(job):
            store, parts = jobs[job]
            t = time.perf_counter()
            moved = {"bytes": 0, "host_bytes": 0}
            try:
                for kind, regions in parts:
                    if not regions:
                        continue
                    paced = pace and job == "weights"
                    # paced weights: load in groups of ~256 MiB, sleeping to hold the target rate
                    for g in _groups(regions, 256 << 20) if paced else [regions]:
                        st = store.load(kind, g) if to_gpu else store.save(kind, g)
                        for k, v in st.items():
                            moved[k] = moved.get(k, 0) + v  # bytes, host_bytes, store counters
                        if paced:
                            ahead = moved["bytes"] / (pace * (1 << 30)) - (time.perf_counter() - t)
                            if ahead > 0:
                                time.sleep(ahead)
            except BaseException as e:  # re-raised on the calling thread
                errors.append(e)
                return
            if moved["bytes"]:
                results[job] = dict(moved, start_s=t - t0, end_s=time.perf_counter() - t0)

        if parallel:
            threads = [threading.Thread(target=run, args=(job,)) for job in jobs]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        else:
            for job in jobs:
                run(job)
        if errors:
            raise errors[0]
        return results

    def _weights_pace(self, kv_regions):
        """Target weights rate (GiB/s) for a parallel DRAM-weights wake whose KV comes over GDS, or None.

        At full speed the DRAM -> GPU weight copy (~53 GiB/s) takes the GPU's whole PCIe
        link, and the GDS KV read makes little progress until it ends: the wake then lasts
        about weights + KV. Paced so that both end together (rate = weights / (KV / GDS rate)),
        the two share the link and the wake lasts about the KV read alone. Measured on a
        23.4 GiB + 8 GiB wake: 0.99 s unpaced, 0.75 s at 35 GiB/s, 0.65 s for the KV alone.
        Off by default; TIERED_WEIGHTS_PACE=1 enables it. TIERED_GDS_GIBS (12) is the expected GDS rate,
        TIERED_PACE_MAX_GIBS (45) the rate above which pacing is skipped.
        """
        if self.weights_path != "dram" or self.kv_path != "gds" or _env("TIERED_WEIGHTS_PACE", "0") != "1":
            return None
        kv = sum(n for _, n, _ in kv_regions)
        w = sum(n for _, n, _ in self.weight_regions + self.state_regions)
        if not kv:
            return None
        kv_s = kv / (float(_env("TIERED_GDS_GIBS", "12")) * (1 << 30))  # expected KV read time
        rate = w / kv_s / (1 << 30)  # GiB/s
        return None if rate >= float(_env("TIERED_PACE_MAX_GIBS", "45")) else rate

    def _verify(self, kv_regions):
        out = {}
        for kind, regions in (("weights", self.weight_regions), ("state", self.state_regions),
                              ("kv", kv_regions)):
            out[f"{kind}_regions"] = len(regions)
            out[f"{kind}_mismatch"] = sum(_checksum(ptr, n) != self.sums[ptr] for ptr, n, _ in regions)
        return out

    def _log(self, rec) -> None:
        logger.info("tiered %s", json.dumps(rec))
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
