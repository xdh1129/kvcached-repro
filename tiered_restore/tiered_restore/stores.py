"""Where offloaded GPU bytes live: host DRAM, SSD through a host bounce buffer, or GDS.

A store holds one flat byte space per kind ("weights" or "kv"). The caller
passes regions as (gpu_ptr, nbytes, store_offset). Every size and offset is a
multiple of 2 MiB (cumem and kvcached page granularity), so O_DIRECT alignment
holds. save() copies GPU -> store, load() copies store -> GPU. Each call returns
the bytes moved and how many of them passed through host DRAM.
"""

import fcntl
import os
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
    """Pinned host buffer per kind; one CUDA stream does every copy."""

    name = "dram"

    def __init__(self, device: int, **_):
        self.device = device
        self.buffers: dict[str, torch.Tensor] = {}
        self.stream = None

    def _buffer(self, kind, size):
        buf = self.buffers.get(kind)
        if buf is None or buf.numel() < size:
            # Pinning is slow (seconds for tens of GB): grow geometrically so KV saves rarely re-pin.
            grown = size if buf is None or kind == "weights" else max(size, buf.numel() * 3 // 2)
            self.buffers[kind] = None
            self.buffers[kind] = buf = torch.empty(grown, dtype=torch.uint8, pin_memory=True)
        return buf

    def _copy(self, kind, regions, to_gpu):
        torch.cuda.set_device(self.device)
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        size = max((off + n for _, n, off in regions), default=0)
        base = self._buffer(kind, size).data_ptr()
        s = self.stream.cuda_stream
        for gpu_ptr, n, off in regions:
            if to_gpu:
                _check(cudart.cudaMemcpyAsync(gpu_ptr, base + off, n, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, s))
            else:
                _check(cudart.cudaMemcpyAsync(base + off, gpu_ptr, n, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, s))
        self.stream.synchronize()
        moved = sum(n for _, n, _ in regions)
        return {"bytes": moved, "host_bytes": moved}

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

    def _run(self, kind, regions, to_gpu):
        self._fd(kind)
        list(self.pool.map(lambda c: self._io(kind, *c, to_gpu), _chunks(regions, self.chunk)))
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

    GDS transfers of different processes are serialized host-wide (flock on
    TIERED_GDS_LOCK, default <parent of TIERED_DIR>/.gds.lock). Two processes
    reading concurrently got only ~2-3 GiB/s each (about 6 in total) in this
    Python path, while one at a time gets ~13; they share the same SSDs anyway.
    gdsio does not show this (6.5 + 6.5); the cause is not known yet.
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
        self.gate = _HostGate(os.environ.get("TIERED_GDS_LOCK") or
                              os.path.join(os.path.dirname(os.path.normpath(root)), ".gds.lock"))

    def _run(self, kind, regions, to_gpu):
        before = self.unregistered
        wait = self.gate.enter()
        try:
            st = super()._run(kind, regions, to_gpu)
        finally:
            self.gate.leave()
        st["unregistered_chunks"] = self.unregistered - before
        st["gate_wait_s"] = wait
        return st

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
