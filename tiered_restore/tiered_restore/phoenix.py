"""PhoenixStore: GDS writes, Phoenix P2P reads straight from the md RAID0 member NVMes.

Phoenix (github.com/xPU-IO/phoenix, phoenixfs.ko + libphoenix) DMAs NVMe -> GPU with
O_DIRECT on any fd. On kernel >= 6.2 that fails on md0 (EREMOTEIO: the md queue lacks
QUEUE_FLAG_PCI_P2PDMA), but works on the member disks, which have the flag. So:

  save: exactly GdsStore (cuFile through ext4 on md0; never a write to a member disk).
  load: FIEMAP the store file (physical offsets on md0), map every piece through the
        raid0 layout (chunk c -> member c % n at data_offset + (c // n) * chunk), and
        read it with phxfs_read_batch from the member opened O_RDONLY | O_DIRECT,
        into the GPU regions. Verified byte-exact against reads through ext4
        (phoenix-eval/raid0_direct).

Only valid while the file is not being written: a load always follows the save of the
same sleep, which has completed. Unwritten extents (fallocated, never written) load as
zeros; a hole inside a requested range is an error.

Env:
  TIERED_PHX_LIB   libphoenix.so (default: the 16-worker build, 13.2 GiB/s on one GPU)
  PHX_STAGING_SIZE_MB  read by libphoenix (staging pool, default 256)
The device is opened (phxfs_open, which maps a staging pool on the GPU) for each load
and closed after it, so a sleeping process holds no Phoenix GPU memory, like GDS staging.
load_many reads several kinds (e.g. a split tail of weights + KV) in one such session.
"""

import bisect
import ctypes
import fcntl
import os
import struct
import threading
import time

import torch
from cuda.bindings import runtime as cudart

from .stores import GdsStore, _check

DEFAULT_LIB = "/home/poc/xdh/phoenix-eval/phoenix/build_t16/libphoenix.so"
FS_IOC_FIEMAP = 0xC020660B
FIEMAP_FLAG_SYNC = 0x1
FIEMAP_EXTENT_LAST = 0x1
FIEMAP_EXTENT_UNWRITTEN = 0x800
FIEMAP_EXTENT_MERGED = 0x1000
_HDR = struct.Struct("=QQIIII")       # struct fiemap (32 bytes)
_EXT = struct.Struct("=QQQQQIIII")    # struct fiemap_extent (56 bytes)


class _Req(ctypes.Structure):  # phxfs_io_req_t
    _fields_ = [("fd", ctypes.c_int), ("device_id", ctypes.c_int), ("buf", ctypes.c_void_p),
                ("buf_offset", ctypes.c_int64), ("nbytes", ctypes.c_size_t),
                ("f_offset", ctypes.c_int64), ("result", ctypes.c_ssize_t)]


def _fiemap(fd, cap=4096):
    """[(logical, physical, length, unwritten)] of the whole file (FIEMAP_FLAG_SYNC)."""
    out, start = [], 0
    while True:
        buf = bytearray(_HDR.size + cap * _EXT.size)
        _HDR.pack_into(buf, 0, start, (1 << 64) - 1 - start, FIEMAP_FLAG_SYNC, 0, cap, 0)
        fcntl.ioctl(fd, FS_IOC_FIEMAP, buf, True)
        mapped = _HDR.unpack_from(buf, 0)[3]
        if not mapped:
            return out
        last = False
        for i in range(mapped):
            lo, ph, ln, _, _, flags, _, _, _ = _EXT.unpack_from(buf, _HDR.size + i * _EXT.size)
            if flags & ~(FIEMAP_EXTENT_LAST | FIEMAP_EXTENT_UNWRITTEN | FIEMAP_EXTENT_MERGED):
                raise IOError(f"unsupported FIEMAP extent flags {flags:#x}")
            out.append((lo, ph, ln, bool(flags & FIEMAP_EXTENT_UNWRITTEN)))
            last = bool(flags & FIEMAP_EXTENT_LAST)
        if last:
            return out
        start = out[-1][0] + out[-1][2]


def _md_layout(path):
    """(chunk, [(member fd, data_offset)]) of the md raid0 device holding path."""
    st = os.stat(path)
    sysd = f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}/md"
    rd = lambda p: open(os.path.join(sysd, p)).read().strip()
    if rd("level") != "raid0":
        raise ValueError(f"{path} is not on an md raid0 device")
    n, chunk = int(rd("raid_disks")), int(rd("chunk_size"))
    sizes = {rd(f"rd{i}/size") for i in range(n)}
    if len(sizes) != 1:
        raise ValueError("md raid0 with unequal members (several zones) is not supported")
    members = []
    for i in range(n):
        dev = os.readlink(os.path.join(sysd, f"rd{i}"))
        assert dev.startswith("dev-")
        fd = os.open(f"/dev/{dev[4:]}", os.O_RDONLY | os.O_DIRECT)  # read-only, always
        members.append((fd, int(rd(f"rd{i}/offset")) * 512))
    return chunk, members


class PhoenixStore(GdsStore):
    name = "phoenix"
    via_host = False

    def __init__(self, device: int, root: str, threads: int, chunk_mb: int):
        super().__init__(device, root, threads, chunk_mb)
        self.lib = ctypes.CDLL(os.environ.get("TIERED_PHX_LIB", DEFAULT_LIB))
        L = self.lib
        L.phxfs_open.argtypes = [ctypes.c_int]
        L.phxfs_close.argtypes = [ctypes.c_int]
        L.phxfs_regmem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        L.phxfs_deregmem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        L.phxfs_read_batch.argtypes = [ctypes.POINTER(_Req), ctypes.c_int]
        L.phxfs_get_map_mode.argtypes = [ctypes.c_int]
        self.layout = None  # (chunk, [(member fd, data_offset)]), on the first load
        self.phx_lock = threading.Lock()

    def save(self, kind, regions):
        return super().save(kind, regions)  # GDS through ext4 + md0

    def _pieces(self, kind, regions):
        """[(member fd, member offset, gpu_ptr, nbytes)] and [(gpu_ptr, n)] to zero, for regions."""
        chunk, members = self.layout
        nd = len(members)
        ext = _fiemap(self.fds[kind])
        starts = [e[0] for e in ext]
        pieces, zero = [], []
        for gpu_ptr, nbytes, off in regions:
            pos, end = off, off + nbytes
            while pos < end:
                i = bisect.bisect_right(starts, pos) - 1
                if i < 0 or pos >= ext[i][0] + ext[i][2]:
                    raise IOError(f"{kind}: file offset {pos} is a hole (never written)")
                lo, ph, ln, unwritten = ext[i]
                n = min(end, lo + ln) - pos
                dst = gpu_ptr + (pos - off)
                if unwritten:
                    zero.append((dst, n))
                else:
                    x = ph + (pos - lo)  # physical address on md0
                    while n:
                        c, inn = divmod(x, chunk)
                        m = min(n, chunk - inn)
                        fd, doff = members[c % nd]
                        pieces.append((fd, doff + (c // nd) * chunk + inn, dst, m))
                        x += m; dst += m; n -= m; pos += m
                    continue
                pos += n
        return pieces, zero

    def load(self, kind, regions):
        return self.load_many([(kind, regions)])

    def load_many(self, parts):
        """Load [(kind, regions)] in one Phoenix session: one open, one read batch, one close.

        Closing waits for the whole device (it frees the staging pool), so a second session
        started after it would sit behind any other copy on the GPU, e.g. the DRAM job."""
        for kind, _ in parts:
            self._fd(kind)
        if self.layout is None:
            self.layout = _md_layout(os.path.join(self.root, f"{parts[0][0]}.bin"))
        torch.cuda.set_device(self.device)
        L, dev = self.lib, self.device
        t0 = time.perf_counter()
        pieces, zero = [], []
        for kind, regs in parts:
            p, z = self._pieces(kind, regs)
            pieces += p
            zero += z
        regions = [r for _, regs in parts for r in regs]
        t_map = time.perf_counter()
        with self.phx_lock:  # one Phoenix load at a time per process (one staging pool per device)
            if L.phxfs_open(dev) != 0:
                raise RuntimeError(f"phxfs_open({dev}) failed: is phoenixfs.ko loaded and readable?")
            regs = []
            try:
                t_open = time.perf_counter()
                tgt = ctypes.c_void_p()
                for gpu_ptr, n, _ in regions:
                    if L.phxfs_regmem(dev, gpu_ptr, n, ctypes.byref(tgt)) != 0:
                        raise RuntimeError(f"phxfs_regmem({gpu_ptr:#x}, {n}) failed")
                    regs.append((gpu_ptr, n))
                reqs = (_Req * len(pieces))()
                # buf = the region start that contains the piece (Phoenix looks the registration up by it)
                starts = sorted(r[0] for r in regions)
                for i, (fd, doff, dst, m) in enumerate(pieces):
                    rbase = starts[bisect.bisect_right(starts, dst) - 1]
                    reqs[i].fd, reqs[i].device_id, reqs[i].buf = fd, dev, rbase
                    reqs[i].buf_offset, reqs[i].nbytes, reqs[i].f_offset = dst - rbase, m, doff
                    reqs[i].result = 0
                t_io = time.perf_counter()
                bad = L.phxfs_read_batch(reqs, len(pieces)) if pieces else 0
                t_end = time.perf_counter()
                if bad:
                    first = next(r for r in reqs if r.result != r.nbytes)
                    raise IOError(f"phxfs_read_batch: {bad}/{len(pieces)} requests failed "
                                  f"(first: fd {first.fd} off {first.f_offset} -> {first.result})")
            finally:
                for gpu_ptr, n in regs:
                    L.phxfs_deregmem(dev, gpu_ptr, n)
                L.phxfs_close(dev)
        for dst, n in zero:
            _check(cudart.cudaMemset(dst, 0, n))
        if self.trace is not None:
            self.trace.append((t_io, t_end, sum(p[3] for p in pieces)))
        moved = sum(n for _, n, _ in regions)
        return {"bytes": moved, "host_bytes": 0, "phx_pieces": len(pieces), "phx_zero_bytes": sum(n for _, n in zero),
                "phx_fiemap_s": t_map - t0, "phx_open_s": t_open - t_map, "phx_io_s": t_end - t_io,
                "data_start_t": t_io, "data_end_t": t_end}
