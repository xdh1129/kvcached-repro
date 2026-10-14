"""Where offloaded GPU bytes live: host DRAM, SSD through a host bounce buffer, or GDS.

A store holds one flat byte space per kind ("weights" or "kv"). The caller
passes regions as (gpu_ptr, nbytes, store_offset). Every size and offset is a
multiple of 2 MiB (cumem and kvcached page granularity), so O_DIRECT alignment
holds. save() copies GPU -> store, load() copies store -> GPU. Each call returns
the bytes moved and how many of them passed through host DRAM.
"""

import os
import threading
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
            self.buffers[kind] = buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
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


class GdsStore(_FileStore):
    """cuFile reads/writes straight between the NVMe and GPU memory (no host bounce)."""

    name = "gds"
    via_host = False

    def __init__(self, *args, **kwargs):
        import cuda.bindings.cufile as cufile

        self.cufile = cufile
        cufile.driver_open()
        self.handles: dict[str, int] = {}
        self.descrs = {}
        super().__init__(*args, **kwargs)

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
        got = fn(fh, gpu_ptr, n, off, 0)
        if got != n:
            raise IOError(f"cuFile {'read' if to_gpu else 'write'} {got}/{n} at {off} in {kind}")


def make_store(name: str, device: int, root: str, threads: int, chunk_mb: int):
    cls = {"dram": DramStore, "ssd": SsdStore, "gds": GdsStore}[name]
    return cls(device=device, root=root, threads=threads, chunk_mb=chunk_mb)
