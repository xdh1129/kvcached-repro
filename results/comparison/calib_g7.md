Rates in GiB/s (medians). W = weights (+ state), write = sleep save, read = wake load.

| run | paths | parallel | model | W GiB | KV GiB | W write | KV write | W read | KV read | sleep s | wake s | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calib_g7_kv8_dram+dram_p0 | dram+dram | 0 | Qwen3-32B | 61.68 | 8.12 | 52.67 | 52.66 | 53.72 | 53.7 | 1.52 | 1.48 | 4/3 |
| calib_g7_kv8_dram+dram_p0 | dram+dram | 0 | Qwen3-14B | 28.16 | 8.91 | 52.66 | 52.66 | 53.7 | 53.69 | 0.81 | 0.78 | 4/3 |
| calib_g7_kv8_dram+dram_p0 | dram+dram | 0 | Phi-3.5-mini-instruct | 7.31 | 9.88 | 52.63 | 52.66 | 53.62 | 53.67 | 0.39 | 0.36 | 3/3 |
| calib_g7_kv8_gds+gds_p0 | gds+gds | 0 | Qwen3-32B | 61.68 | 8.12 | 10.97 | 10.46 | 13.01 | 12.29 | 6.6 | 5.59 | 4/3 |
| calib_g7_kv8_gds+gds_p0 | gds+gds | 0 | Qwen3-14B | 28.16 | 8.91 | 10.87 | 10.54 | 12.99 | 12.55 | 3.56 | 2.97 | 4/3 |
| calib_g7_kv8_gds+gds_p0 | gds+gds | 0 | Phi-3.5-mini-instruct | 7.31 | 9.94 | 10.42 | 10.61 | 12.43 | 12.52 | 1.74 | 1.43 | 3/3 |
| calib_g7_kv8_dram+gds_p1 | dram+gds | 1 | Qwen3-32B | 61.68 | 8.12 | 52.67 | 10.8 | 47.54 | 6.23 | 2.13 | 1.48 | 4/3 |
| calib_g7_kv8_dram+gds_p1 | dram+gds | 1 | Qwen3-14B | 28.16 | 8.91 | 52.66 | 10.9 | 41.22 | 12.56 | 1.48 | 0.8 | 4/3 |
| calib_g7_kv8_dram+gds_p1 | dram+gds | 1 | Phi-3.5-mini-instruct | 7.31 | 9.94 | 52.63 | 11.01 | 42.55 | 12.49 | 1.12 | 0.84 | 3/3 |
| calib_g7_kv8_gds+dram_p1 | gds+dram | 1 | Qwen3-32B | 61.68 | 8.12 | 11.03 | 52.61 | 13.08 | 42.31 | 5.93 | 4.89 | 4/3 |
| calib_g7_kv8_gds+dram_p1 | gds+dram | 1 | Qwen3-14B | 28.16 | 8.91 | 10.98 | 52.61 | 12.96 | 42.09 | 2.84 | 2.27 | 4/3 |
| calib_g7_kv8_gds+dram_p1 | gds+dram | 1 | Phi-3.5-mini-instruct | 7.31 | 9.94 | 10.41 | 52.59 | 12.35 | 42.03 | 0.96 | 0.64 | 3/3 |

GPU 7 alone, Qwen3-32B / Qwen3-14B / Phi-3.5-mini, ~8-10 GiB primed KV per model, KV pages of
128 MiB compound, TIERED_GDS_STAGE_MB=256. parallel=0: weights then KV; parallel=1: both at once
(wake only; sleep always saves weights, then KV). First save per model left out (host alloc).

## Reading

- DRAM alone: ~53 GiB/s both ways, same for weights and KV (large KV pages: no fragmentation penalty).
- GDS alone (md RAID0 of two NVMe): read ~12.5-13, write ~10.4-11 GiB/s, same for weights and KV.
- DRAM || GDS on one GPU: the sum stays ~55 GiB/s, the GPU's PCIe link. While both run, GDS keeps
  ~13.1 and DRAM drops to ~42 GiB/s, for all three models (trace run: calib_g7_kv8_dram+gds_p1_trace).
- The "kv 6.2" of Qwen3-32B in the table is a measurement artifact, not a slow transfer: its GDS data
  ended at 0.82 s (13.1 GiB/s), but the KV job's end_s came 0.66 s later, when the DRAM copies ended.
  Only the staging cleanup (cuFile deregister, cudaFree) runs in that interval.
- Staging: TIERED_GDS_STAGE_MB=0 is worse (KV read 4.5-5.3 GiB/s), 4 buffers give nothing over 2.
- Consequence for the split path: a wake cannot beat (W+K)/~55 GiB/s. Read share that costs nothing:
  13.1 / (42.5 + 13.1) = ~24%, all models. Sleep (parallel save) not yet measured.

## Stage 1: both directions under overlap (calib_g7_kv8_dram+gds_p1_psave, trace, parallel save)

Rates inside the window where DRAM and GDS both run, medians over all models:

| direction | GDS | DRAM | sum | X/T = GDS / (GDS + DRAM) |
|---|---|---|---|---|
| wake (read) | ~13.05 | ~40.9 | ~54.0 | ~0.24 |
| sleep (write) | ~8.9 | ~43.5 | ~52.5 | ~0.17 |

f = min(0.24, 0.17) = 0.17: the write side limits it. Very stable across the three models.
