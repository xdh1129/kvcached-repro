# Two wakes at the same moment on GPUs 6 and 7 (2026-10-19)

scripts/bench_overlap_wake.py. GPU A (6) wakes a model whose weights come from DRAM. GPU B (7)
wakes another model at the same instant: its weights from DRAM and its KV from storage on two
threads (TIERED_PARALLEL=1). B's KV through the POSIX path (dram+ssd) or GDS with 256 MiB
staging (dram+gds). Sizes from the replication-3 balanced reference run. Median of 5 reps, seconds.

| event | A weights: alone / B KV ssd / B KV gds | B KV: alone / with A (ssd) | B KV: alone / with A (gds) | B wake with A: ssd / gds | both done: ssd / gds |
|---|---|---|---|---|---|
| t=60.7 s: A 14.7 GB weights; B 6.5 GB weights + 3.5 GB KV | 0.255 / 0.414 / 0.368 | 0.340 / 0.457 | 0.283 / 0.288 | 0.457 / 0.288 (-37%) | 0.457 / 0.368 (-19%) |
| typical: A 7.8 GB; B 7.5 GB + 5 GB KV | 0.135 / 0.270 / 0.265 | 0.464 / 0.592 | 0.389 / 0.396 | 0.592 / 0.396 (-33%) | 0.592 / 0.396 (-33%) |
| large KV: A 16.7 GB; B 7.5 GB + 20 GB KV | 0.290 / 0.473 / 0.420 | 1.534 / 1.662 | 1.445 / 1.458 | 1.662 / 1.458 (-12%) | 1.662 / 1.458 (-12%) |

- A's weights always slow down when B wakes, because B's own weights also come from DRAM over
  the shared uplink (two DRAM loads split ~53.7 GiB/s). GDS KV on B removes only the KV share
  of that contention: A's weights 0.414 -> 0.368 s and 0.473 -> 0.420 s (-11%).
- The larger gain is on B itself: with POSIX the KV competes with A's weights for the uplink
  (3.5 GB KV: 0.340 -> 0.457 s); with GDS it does not (0.283 -> 0.288 s).
- Even alone, B's wake is faster with GDS KV (0.283 vs 0.340 s): with POSIX, B's KV competes with
  B's own DRAM weights for the same uplink.
