"""Where offloaded GPU bytes live: host DRAM, SSD through a host bounce buffer, or GDS.

A store holds one flat byte space per kind ("weights" or "kv"). The caller
passes regions as (gpu_ptr, nbytes, store_offset). Every size and offset is a
multiple of 2 MiB (cumem and kvcached page granularity), so O_DIRECT alignment
holds. save() copies GPU -> store, load() copies store -> GPU. Each call returns
the bytes moved and how many of them passed through host DRAM.

Setting a store's ``trace`` to a list (off: None, the default) appends one
(t_start, t_end, nbytes) per transfer piece, in time.perf_counter() seconds:
per region for DRAM (CUDA events, no extra syncs), per chunk or staged span for files.
"""

import fcntl
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from cuda.bindings import runtime as cudart

MIB = 1 << 20


def _check(err) -> None:
    err = err[0] if isinstance(err, tuple) else err
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error {err}")


def _chunks(regions, chunk):
    for gpu_ptr, nbytes, off in regions:
        for i in range(0, nbytes, chunk):
            yield gpu_ptr + i, min(chunk, nbytes - i), off + i


class DramStore:
    """Pinned host buffer per kind; one CUDA stream does every copy.

    Default, as vLLM's cumem backend: a kind's buffer is pinned at the save
    (torch pin_memory, which reuses freed pinned blocks through torch's caching host
    allocator) and dropped after the load. Kinds in `keep` (weights saved once)
    keep their buffer. Each save reports the time spent pinning (alloc_s).

    TIERED_DRAM_KV_RESERVE_GB (default 0) > 0: the KV buffer is instead reserved at
    that size on the first save and kept, growing by 1.5x when a save needs more.
    Pinning 8-12 GB takes ~2-3 s, which a reserve moves out of the sleeps.
    """

    name = "dram"

    def __init__(self, device: int, **_):
        self.device = device
        self.buffers: dict[str, torch.Tensor] = {}
        self.stream = None
        self.kv_reserve = int(float(os.environ.get("TIERED_DRAM_KV_RESERVE_GB", "0")) * (1 << 30))
        self.keep = {"kv"} if self.kv_reserve else set()  # kinds whose buffer outlives the load
        self.trace = None

    def _buffer(self, kind, size):
        """(buffer, seconds spent pinning new host memory)"""
        buf = self.buffers.get(kind)
        if buf is not None and buf.numel() >= size:
            return buf, 0.0
        t = time.perf_counter()
        want = size
        if kind == "kv" and self.kv_reserve:
            want = max(size, self.kv_reserve if buf is None else buf.numel() * 3 // 2)
        self.buffers[kind] = None
        self.buffers[kind] = buf = torch.empty(want, dtype=torch.uint8, pin_memory=True)
        return buf, time.perf_counter() - t

    def _copy(self, kind, regions, to_gpu):
        torch.cuda.set_device(self.device)
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        size = max((off + n for _, n, off in regions), default=0)
        buf, alloc_s = self._buffer(kind, size)
        base = buf.data_ptr()
        s = self.stream.cuda_stream
        events = None
        if self.trace is not None:
            start = torch.cuda.Event(enable_timing=True)
            start.record(self.stream)
            t_wall, events = time.perf_counter(), []
        for gpu_ptr, n, off in regions:
            if to_gpu:
                _check(cudart.cudaMemcpyAsync(gpu_ptr, base + off, n, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, s))
            else:
                _check(cudart.cudaMemcpyAsync(base + off, gpu_ptr, n, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, s))
            if events is not None:
                events.append((torch.cuda.Event(enable_timing=True), n))
                events[-1][0].record(self.stream)
        self.stream.synchronize()
        if events is not None:  # the copies run back to back on one stream
            prev = 0.0
            for ev, n in events:
                end = start.elapsed_time(ev) / 1e3
                self.trace.append((t_wall + prev, t_wall + end, n))
                prev = end
        if to_gpu and kind not in self.keep:
            self.buffers[kind] = None  # as vLLM drops its CPU backup after waking
        moved = sum(n for _, n, _ in regions)
        return {"bytes": moved, "host_bytes": moved, "alloc_s": alloc_s}

    def save(self, kind, regions):
        return self._copy(kind, regions, to_gpu=False)

    def load(self, kind, regions):
        return self._copy(kind, regions, to_gpu=True)


class _FileStore:
    """One O_DIRECT file per kind under ``root``; work is split into chunks over a thread pool."""

    def __init__(self, device: int, root: str, threads: int, chunk_mb: int):
        self.device, self.root, self.chunk = device, root, chunk_mb * MIB
        os.makedirs(root, exist_ok=True)
        self.fds: dict[str, int] = {}
        self.trace = None
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix=f"tiered-{self.name}",
                                       initializer=torch.cuda.set_device, initargs=(device,))

    def _fd(self, kind):
        if kind not in self.fds:
            path = os.path.join(self.root, f"{kind}.bin")
            self.fds[kind] = os.open(path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)
            self._opened(kind, self.fds[kind])
        return self.fds[kind]

    def _opened(self, kind, fd):
        pass

    def _io_traced(self, kind, gpu_ptr, n, off, to_gpu):
        t = time.perf_counter()
        self._io(kind, gpu_ptr, n, off, to_gpu)
        if self.trace is not None:
            self.trace.append((t, time.perf_counter(), n))

    def _run(self, kind, regions, to_gpu):
        self._fd(kind)
        list(self.pool.map(lambda c: self._io_traced(kind, *c, to_gpu), _chunks(regions, self.chunk)))
        moved = sum(n for _, n, _ in regions)
        return {"bytes": moved, "host_bytes": moved if self.via_host else 0}

    def save(self, kind, regions):
        return self._run(kind, regions, to_gpu=False)

    def load(self, kind, regions):
        return self._run(kind, regions, to_gpu=True)


class SsdStore(_FileStore):
    """POSIX O_DIRECT I/O into a per-thread pinned bounce buffer, then cudaMemcpy."""

    name = "ssd"
    via_host = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local = threading.local()

    def _thread_state(self):
        st = self.local
        if not hasattr(st, "buf"):
            torch.cuda.set_device(self.device)
            st.buf = torch.empty(self.chunk, dtype=torch.uint8, pin_memory=True)
            st.view = memoryview(st.buf.numpy())
            st.stream = torch.cuda.Stream()
        return st

    def _io(self, kind, gpu_ptr, n, off, to_gpu):
        st, fd = self._thread_state(), self.fds[kind]
        s = st.stream.cuda_stream
        if to_gpu:
            got = os.preadv(fd, [st.view[:n]], off)
            if got != n:
                raise IOError(f"short read {got}/{n} at {off} in {kind}")
            _check(cudart.cudaMemcpyAsync(gpu_ptr, st.buf.data_ptr(), n, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, s))
            st.stream.synchronize()
        else:
            _check(cudart.cudaMemcpyAsync(st.buf.data_ptr(), gpu_ptr, n, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, s))
            st.stream.synchronize()
            if os.pwritev(fd, [st.view[:n]], off) != n:
                raise IOError(f"short write at {off} in {kind}")


class _HostGate:
    """Host-wide mutual exclusion across processes (flock), shared by the threads of one process.

    The first thread of a process to enter takes the file lock; the last to leave
    releases it. Threads of the same process (weights and KV loads) run together.
    """

    def __init__(self, path: str):
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        self.cv = threading.Lock()
        self.users = 0

    def enter(self) -> float:
        t = time.perf_counter()
        with self.cv:
            if self.users == 0:
                fcntl.flock(self.fd, fcntl.LOCK_EX)  # other threads of this process wait on cv meanwhile
            self.users += 1
        return time.perf_counter() - t

    def leave(self) -> None:
        with self.cv:
            self.users -= 1
            if self.users == 0:
                fcntl.flock(self.fd, fcntl.LOCK_UN)


class GdsStore(_FileStore):
    """cuFile reads/writes straight between the NVMe and GPU memory (no host bounce).

    Each I/O chunk is registered with cuFileBufRegister just for that I/O. An
    unregistered GPU buffer makes cuFile stage every 1 MiB through internal GPU
    bounce buffers and copy them on a CUDA stream. That copy competes with other
    processes' kernels on the GPU: measured 13.3 -> 2.3 GiB/s on a GPU busy with
    another model, unless an idle GPU near the NVMe is visible to host the bounce
    buffers. A registered chunk is DMAed in place and keeps ~12.6 GiB/s (chunks of
    256 MiB, registration cost included). Registering a whole multi-GiB region at
    once does not help (2.6 GiB/s), hence per-chunk registration. The memory is
    remapped on every wake, so registrations cannot outlive one I/O.

    TIERED_GDS_GATE=1 (default 0) serializes GDS transfers of different processes
    host-wide (flock on TIERED_GDS_LOCK, default <parent of TIERED_DIR>/.gds.lock). Two processes
    reading concurrently got only ~2-3 GiB/s each (about 6 in total) in this
    Python path, while one at a time gets ~13; they share the same SSDs anyway.
    Cause (results/bench_gds_multiproc): direct reads into GPUs behind different
    PCIe switches of NUMA 0 (GPU 1 with 2 or 3; the NVMe is on NUMA 1) collapse
    to ~3.4 GiB/s in total, gdsio included, unless GPU 6 or 7 (the NVMe's switch)
    is visible for cuFile to bounce through. Same switch or one GPU per socket: ~13.

    TIERED_GDS_STAGE_MB=n (default 0, off): the regions (cumem allocations, median
    ~48 MiB for weights) cap each cuFile I/O at one region's size. Instead, the
    store file is read/written in n MiB spans through TIERED_GDS_STAGE_BUFS (default
    2) cudaMalloc'd, registered GPU staging buffers, with device-to-device copies
    between a span and the regions it covers. The staging buffers exist only for
    one save/load, so a sleeping process holds no extra memory. 7.3 GiB in 128
    regions of 24-400 MiB, one GPU: read 4.8 -> 12.5, write 6.3 -> 10.5 GiB/s
    with 256 MiB x 2 (128 x 4 and 512 x 4 about the same, 8 buffers slower).
    """

    name = "gds"
    via_host = False

    def __init__(self, device: int, root: str, threads: int, chunk_mb: int):
        import cuda.bindings.cufile as cufile

        self.cufile = cufile
        cufile.driver_open()
        self.handles: dict[str, int] = {}
        self.descrs = {}
        self.lock = threading.Lock()
        self.unregistered = 0  # chunks that could not be registered (slow bounce-buffer path)
        super().__init__(device, root, threads, int(os.environ.get("TIERED_GDS_CHUNK_MB", "256")))
        self.stage = int(os.environ.get("TIERED_GDS_STAGE_MB", "0")) * MIB
        self.stage_bufs = int(os.environ.get("TIERED_GDS_STAGE_BUFS", "2"))
        self.local = threading.local()
        self.gate = None
        if os.environ.get("TIERED_GDS_GATE", "0") == "1":
            self.gate = _HostGate(os.environ.get("TIERED_GDS_LOCK") or
                                  os.path.join(os.path.dirname(os.path.normpath(root)), ".gds.lock"))

    def _run(self, kind, regions, to_gpu):
        before = self.unregistered
        wait = self.gate.enter() if self.gate else 0.0
        try:
            st = self._run_staged(kind, regions, to_gpu) if self.stage else super()._run(kind, regions, to_gpu)
        finally:
            if self.gate:
                self.gate.leave()
        st["unregistered_chunks"] = self.unregistered - before
        st["gate_wait_s"] = wait
        return st

    def _spans(self, regions):
        """[(file_off, nbytes, [(gpu_ptr, n, offset_in_span)])]: contiguous file ranges of at most `stage` bytes"""
        spans = []
        for gpu_ptr, nbytes, off in sorted(regions, key=lambda r: r[2]):
            i = 0
            while i < nbytes:
                if not spans or spans[-1][0] + spans[-1][1] != off + i or spans[-1][1] == self.stage:
                    spans.append([off + i, 0, []])
                span = spans[-1]
                n = min(nbytes - i, self.stage - span[1])
                span[2].append((gpu_ptr + i, n, span[1]))
                span[1] += n
                i += n
        return spans

    def _run_staged(self, kind, regions, to_gpu):
        self._fd(kind)
        torch.cuda.set_device(self.device)
        spans = self._spans(regions)
        bufs = queue.Queue()
        t = time.perf_counter()
        for _ in range(min(self.stage_bufs, len(spans))):
            err, ptr = cudart.cudaMalloc(self.stage)
            _check(err)
            self.cufile.buf_register(ptr, self.stage, 0)
            bufs.put(ptr)
        setup_s = time.perf_counter() - t

        def io(span):
            file_off, n, pieces = span
            st = self.local
            if not hasattr(st, "stream"):
                st.stream = torch.cuda.Stream()
            s = st.stream.cuda_stream
            buf = bufs.get()
            t = time.perf_counter()
            try:
                d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
                if to_gpu:
                    self._io_registered(kind, buf, n, file_off, True)
                    for gpu_ptr, m, at in pieces:
                        _check(cudart.cudaMemcpyAsync(gpu_ptr, buf + at, m, d2d, s))
                    st.stream.synchronize()
                else:
                    for gpu_ptr, m, at in pieces:
                        _check(cudart.cudaMemcpyAsync(buf + at, gpu_ptr, m, d2d, s))
                    st.stream.synchronize()
                    self._io_registered(kind, buf, n, file_off, False)
                if self.trace is not None:
                    self.trace.append((t, time.perf_counter(), n))
            finally:
                bufs.put(buf)

        try:
            list(self.pool.map(io, spans))
        finally:
            while not bufs.empty():
                ptr = bufs.get()
                self.cufile.buf_deregister(ptr)
                _check(cudart.cudaFree(ptr))
        return {"bytes": sum(n for _, n, _ in regions), "host_bytes": 0, "stage_spans": len(spans),
                "stage_setup_s": setup_s}

    def _io_registered(self, kind, gpu_ptr, n, off, to_gpu):
        fn = self.cufile.read if to_gpu else self.cufile.write
        got = fn(self.handles[kind], gpu_ptr, n, off, 0)
        if got != n:
            raise IOError(f"cuFile {'read' if to_gpu else 'write'} {got}/{n} at {off} in {kind}")

    def _opened(self, kind, fd):
        cf = self.cufile
        d = cf.Descr()
        d.type = cf.FileHandleType.OPAQUE_FD
        d.handle.fd = fd
        d.fs_ops = 0  # must be zeroed, or cuFileHandleRegister segfaults
        self.descrs[kind] = d
        self.handles[kind] = cf.handle_register(d.ptr)

    def _io(self, kind, gpu_ptr, n, off, to_gpu):
        fh = self.handles[kind]
        fn = self.cufile.read if to_gpu else self.cufile.write
        try:
            self.cufile.buf_register(gpu_ptr, n, 0)
        except self.cufile.cuFileError:
            # e.g. memory from cuMemCreate without gpuDirectRDMACapable: fall back to bounce buffers
            with self.lock:
                self.unregistered += 1
            got = fn(fh, gpu_ptr, n, off, 0)
        else:
            try:
                got = fn(fh, gpu_ptr, n, off, 0)
            finally:
                self.cufile.buf_deregister(gpu_ptr)
        if got != n:
            raise IOError(f"cuFile {'read' if to_gpu else 'write'} {got}/{n} at {off} in {kind}")


def make_store(name: str, device: int, root: str, threads: int, chunk_mb: int):
    cls = {"dram": DramStore, "ssd": SsdStore, "gds": GdsStore}[name]
    return cls(device=device, root=root, threads=threads, chunk_mb=chunk_mb)
