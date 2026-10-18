# Concurrent GDS reads from several processes (2026-10-19)

Machine: 2x AMD EPYC 9355. NVMe RAID0 (nvme1n1, nvme2n1) under root port e0:01.1 on NUMA 1,
behind the same PCIe switch as GPUs 6 and 7. GPUs 0-3 on NUMA 0 (switches 0-1, 2-3), 4-7 on NUMA 1.

## scripts/bench_gds_multiproc.py (GdsStore, TIERED_GDS_STAGE_MB=256, 7.3 GiB per process, one GPU visible per process)

| GPUs (one process each) | op | aggregate GiB/s | per process |
|---|---|---|---|
| 1 | load | 12.6 | 12.6 |
| 4 | load | 12.7 | 12.7 |
| 1, 1 (same GPU) | load | 12.9 | 6.4, 6.7 |
| 2, 3 (same switch) | load | 12.9 | 6.5, 6.5 |
| 4, 5 (same switch) | load | 12.9 | 7.1, 6.5 |
| 1, 4 | load | 12.9 | 6.5, 6.5 |
| 2, 5 | load | 13.0 | 6.7, 6.6 |
| 3, 4 | load | 12.9 | 6.4, 6.7 |
| **1, 2** | load | **3.4** | 1.7, 1.7 |
| **1, 3** | load | **3.5** | 1.7, 1.8 |
| **1, 2, 3, 4** | load | **6.0** | 1.5, 1.5, 1.8, 2.9 |
| 1, 2 | save | 11.0 | 5.8, 5.6 |
| ssd store: 1 / 1,2 / 1,2,3,4 | load | 13.2 / 13.2 / 13.2 | |

## gdsio (8 threads, 256 MiB I/O, 8 GiB file per process, read)

| CUDA_VISIBLE_DEVICES of the two processes | GiB/s each |
|---|---|
| all GPUs (-d 1, -d 2) | 6.1, 6.1 |
| 1 / 2 | 1.6, 1.4 |
| 1,2 / 1,2 | 1.7, 1.4 |
| 1,3 / 2,3 | 1.6, 1.5 |
| 1,4 / 2,4 | 1.7, 1.6 |
| 1,7 / 2,7 | 6.3, 6.2 |

With GPU 7 visible, cuFile bounces the reads through GPU 7 (behind the NVMe's switch):
nvidia-smi dmon shows ~1.1 GiB allocated on GPU 7 and ~15 GB/s rx and tx on it, while
GPU 1 receives ~7.5 GB/s. Without it, cuFile DMAs NVMe -> GPU directly (cufile stats:
nvfs=128, posix=0 in both cases).

## Conclusion

Direct NVMe -> GPU reads are fine for one destination (12.6 GiB/s, even across the socket)
and for two GPUs behind one switch, or one per socket. They collapse to ~3.4 GiB/s in
total when two processes read at once into GPUs behind *different switches of NUMA 0*
(GPU 1 with 2 or 3), the socket remote from the NVMe. Writes (GPU -> NVMe) are not
affected, nor is the POSIX path (host bounce). This is a platform P2P limitation, not
our code: gdsio shows it too. Our Prism 2-GPU runs used GPUs 1 and 2, exactly that case.
