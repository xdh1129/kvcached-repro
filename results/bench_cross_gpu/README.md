# Cross-GPU overlap: GPU A loads weights from DRAM while GPU B loads KV (2026-10-19)

scripts/bench_cross_gpu.py: one process per GPU, pinned to NUMA 1 CPUs, 8 GiB per load in
256 MiB regions, each process repeats its load for 8 s after a barrier. KV: SsdStore (POSIX
O_DIRECT through a host bounce buffer) or GdsStore with TIERED_GDS_STAGE_MB=256.

| case | GPUs 6+7 (NVMe's switch) | GPUs 4+5 (NUMA 1, other switch) |
|---|---|---|
| A weights (DRAM) alone | 53.7 | 53.7 |
| B KV (ssd) alone | 13.1 | 13.1 |
| B KV (gds) alone | 12.4 | 12.4-12.7 |
| A weights (DRAM) + B KV (ssd) | **41.0** + 12.7 | **40.7** + 13.0-13.1 |
| A weights (DRAM) + B KV (gds) | **53.7** + 12.3 | **53.7** + 12.7 |
| A weights (DRAM) + B weights (DRAM) | 26.9 + 26.9 | 26.9 + 26.9 |

(GiB/s per process.)

- Host -> GPU traffic of both GPUs shares one limit of ~53.7 GiB/s (two DRAM loads get 26.9 each).
- POSIX KV for GPU B comes down that same path, so A's DRAM weights drop 53.7 -> ~41 (-24%).
- GDS KV for GPU B does not: A keeps 53.7 while B gets 12.3-12.7.
- nvidia-smi dmon on 4+5 (DRAM + GDS case): GPU 4 rx ~62 GB/s, GPU 5 rx ~15.5 GB/s, 77.6 GB/s in
  total, more than one PCIe 5.0 x16 link carries; two DRAM loads give 31 + 31 GB/s. So on 4+5
  the NVMe -> GPU 5 traffic also avoids the host link of the 80:01.1 switch. The switches are
  Broadcom PEX in synthetic mode ("Virtual PCIe Placeholder Endpoint"), so sysfs shows a virtual
  tree; the physical path between the NVMe switch (e0) and the 80 switch is not visible. Not
  verified further.
