# Agent workload, GPUs 6+7: staggering the two GPUs' swaps

GPU 7 sends each step's request Δ s after GPU 6 (`OFFSET_S`, `--sync` lockstep otherwise unchanged).
Δ 0 = the original runs `agent_rr_g67_dram+dram` / `agent_rr_g67_dram+gds`; others `agent_rr_g67_<paths>_off<Δ>`.
8 agents per GPU, 24 turns, 384/384 requests ok in every run. GDS: TIERED_GDS_STAGE_MB=256.

## GPU time (mean of GPU 6 and 7)

| paths | Δ s | makespan s | compute | wait: load | wait: offload | wait: other GPU |
|---|---|---|---|---|---|---|
| dram+dram | 0 | 985 | 73.4% | 11.6% | 14.3% | 0.7% |
| dram+dram | 0.3 | **946** | 76.5% | 6.7% | 10.1% | 6.7% |
| dram+dram | 1 | 1067 | 67.9% | 5.7% | 8.0% | 18.4% |
| dram+gds | 0 | 1077 | 67.6% | 12.4% | 19.3% | 0.7% |
| dram+gds | 0.3 | **1037** | 70.3% | 8.2% | 14.3% | 7.1% |
| dram+gds | 1 | 1138 | 64.0% | 6.8% | 11.1% | 17.9% |

## Per swap (tiered_restore logs, medians over both GPUs)

| paths | Δ s | GPU6 wake ∩ GPU7 wake | GPU6 wake ∩ GPU7 sleep | wake xfer, turns 19-24 | sleep, turns 20-24 | weights GiB/s per wake | arrival → first token, turn 24 |
|---|---|---|---|---|---|---|---|
| dram+dram | 0 | 99% | 1% | 0.69 s | 0.79 s | 22.5 | 1.80 s |
| dram+dram | 0.3 | 18% | 66% | 0.38 s | 0.49 s | 41.9 | 1.16 s |
| dram+dram | 1 | 0% | 12% | 0.35 s | 0.42 s | 44.6 | 1.03 s |
| dram+gds | 0 | 98% | 1% | 0.58 s | 1.33 s | 27.2 | 2.26 s |
| dram+gds | 0.3 | 21% | 69% | 0.36 s | 0.90 s | 45.1 | 1.59 s |
| dram+gds | 1 | 0% | 39% | 0.36 s | 0.73 s | 44.9 | 1.35 s |

## Reading

- Δ 0: both GPUs sleep together then wake together, same PCIe direction on the shared host uplink:
  ~27 GiB/s each for weights.
- Δ 0.3 s: GPU 6's wake (host → GPU, RX) mostly overlaps GPU 7's sleep (GPU → host, TX). PCIe is
  full duplex, so neither slows the other: weights reach the single-GPU rate (~42-45 GiB/s) and the
  load/offload waits drop by about a third. Makespan −4% for both paths.
- Δ 1 s removes the remaining overlap but gains little more on transfers; the lockstep then makes
  GPU 6 wait ~1 s per step, so the makespan grows. That cost belongs to the lockstep design, not to
  the data movement: per-request waits keep improving (turn-24 wait 1.80 → 1.16 → 1.03 s for
  dram+dram). dram+gds gains more from Δ 1 at late turns, because its KV (GDS) phase is long and
  0.3 s does not separate it.
- Figures: results/figures/O1_offset_gpu_time.png, O2_offset_ttft_by_turn.png, O3_offset_duplex_timeline.png.
