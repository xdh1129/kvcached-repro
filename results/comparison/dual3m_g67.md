## Whole run

| path | requests ok | workload s | per GPU: sleep excl. pin / pin / wake (s) | sleep+wake excl. pin, both GPUs |
|---|---|---|---|---|
| dram+dram | 90/90 | 334 | g6 66.2 / 12.9 / 66.4; g7 66.3 / 12.9 / 66.5 | 265.5 |
| dram+gds | 90/90 | 333 | g6 74.9 / 0.0 / 70.3; g7 74.2 / 0.0 / 70.0 | 289.3 |
| dram+phoenix | 90/90 | 333 | g6 74.6 / 0.0 / 70.1; g7 73.8 / 0.0 / 70.0 | 288.4 |
| split | 90/90 | 323 | g6 65.1 / 6.1 / 62.5; g7 65.3 / 6.2 / 62.4 | 255.3 |
| split+phoenix | 90/90 | 322 | g6 65.4 / 5.0 / 62.8; g7 65.6 / 5.1 / 62.5 | 256.3 |
| split+phoenix+w | 90/90 | 336 | g6 83.1 / 5.2 / 59.0; g7 82.4 / 5.2 / 59.2 | 283.8 |

## Wake overlap between the GPUs

Pairs = same agent's k-th wake on both GPUs. overlap = shared time / the shorter wake.

| path | pairs | median overlap | pairs with overlap >= 50% | median start offset ms |
|---|---|---|---|---|
| dram+dram | 45 | 100% | 45/45 | 4 |
| dram+gds | 45 | 100% | 45/45 | 5 |
| dram+phoenix | 45 | 100% | 45/45 | 4 |
| split | 45 | 100% | 45/45 | 4 |
| split+phoenix | 45 | 100% | 45/45 | 4 |
| split+phoenix+w | 45 | 100% | 45/45 | 7 |

## Qwen3-32B: median sleep excl. pin / wake (s) by turn range, both GPUs

| path | turns 2-5 | turns 6-10 | turns 11-15 | KV GiB at last sleep |
|---|---|---|---|---|
| dram+dram | 2.58 / 2.54 | 2.68 / 2.64 | 2.84 / 2.73 | 8.0 |
| dram+gds | 2.54 / 2.49 | 2.54 / 2.50 | 2.57 / 2.44 | 8.0 |
| dram+phoenix | 2.54 / 2.49 | 2.52 / 2.49 | 2.59 / 2.43 | 8.0 |
| split | 2.55 / 2.49 | 2.55 / 2.50 | 2.56 / 2.48 | 8.0 |
| split+phoenix | 2.55 / 2.49 | 2.56 / 2.49 | 2.58 / 2.49 | 8.0 |
| split+phoenix+w | 3.10 / 2.25 | 3.24 / 2.34 | 3.54 / 2.41 | 8.0 |

## Qwen3-14B: median sleep excl. pin / wake (s) by turn range, both GPUs

| path | turns 2-5 | turns 6-10 | turns 11-15 | KV GiB at last sleep |
|---|---|---|---|---|
| dram+dram | 1.19 / 1.16 | 1.29 / 1.23 | 1.33 / 1.30 | 4.7 |
| dram+gds | 1.18 / 1.15 | 1.20 / 1.15 | 1.30 / 1.16 | 4.7 |
| dram+phoenix | 1.18 / 1.15 | 1.20 / 1.12 | 1.31 / 1.15 | 4.7 |
| split | 1.18 / 1.15 | 1.18 / 1.15 | 1.30 / 1.15 | 4.7 |
| split+phoenix | 1.18 / 1.14 | 1.18 / 1.14 | 1.26 / 1.15 | 4.7 |
| split+phoenix+w | 1.46 / 1.05 | 1.59 / 1.07 | 1.94 / 1.11 | 4.7 |

## Phi-3.5: median sleep excl. pin / wake (s) by turn range, both GPUs

| path | turns 2-5 | turns 6-10 | turns 11-15 | KV GiB at last sleep |
|---|---|---|---|---|
| dram+dram | 0.45 / 0.41 | 0.62 / 0.57 | 0.77 / 0.74 | 11.9 |
| dram+gds | 0.75 / 0.44 | 1.40 / 1.02 | 2.08 / 1.70 | 11.9 |
| dram+phoenix | 0.73 / 0.42 | 1.41 / 1.05 | 2.09 / 1.72 | 11.9 |
| split | 0.54 / 0.40 | 0.73 / 0.53 | 0.90 / 0.66 | 11.9 |
| split+phoenix | 0.56 / 0.39 | 0.76 / 0.57 | 0.92 / 0.69 | 11.9 |
| split+phoenix+w | 0.60 / 0.41 | 0.76 / 0.55 | 0.91 / 0.71 | 11.9 |

## SSD read rate during wakes (GiB/s, median over wakes with an SSD part)

| path | Qwen3-32B | Qwen3-14B | Phi-3.5 | all |
|---|---|---|---|---|
| dram+gds | 6.6 (3.8 GiB) | 6.5 (2.3 GiB) | 6.5 (5.9 GiB) | 6.5 |
| dram+phoenix | 6.3 (3.8 GiB) | 6.2 (2.3 GiB) | 6.4 (5.9 GiB) | 6.4 |
| split | 6.4 (3.8 GiB) | 6.4 (2.3 GiB) | 6.4 (2.6 GiB) | 6.4 |
| split+phoenix | 6.2 (3.8 GiB) | 6.2 (2.3 GiB) | 6.3 (2.6 GiB) | 6.2 |
| split+phoenix+w | 6.3 (13.1 GiB) | 6.5 (6.1 GiB) | 6.3 (2.6 GiB) | 6.3 |

## DRAM held by tiered (logical bytes, all 6 instances, during the workload)

| path | peak total GiB | at peak: weights / state / KV GiB | time-weighted mean GiB |
|---|---|---|---|
| dram+dram | 243.4 | 194.3 / 0.0 / 49.1 | 128.9 |
| dram+gds | 194.3 | 194.3 / 0.0 / 0.0 | 114.1 |
| dram+phoenix | 194.3 | 194.3 / 0.0 / 0.0 | 114.0 |
| split | 210.4 | 194.3 / 0.0 / 16.1 | 118.4 |
| split+phoenix | 210.4 | 194.3 / 0.0 / 16.1 | 118.3 |
| split+phoenix+w | 194.7 | 178.7 / 0.0 / 16.1 | 107.3 |

## Monitor

| path | pcie.csv median interval ms | peak NVMe read, both members (GiB/s, 10 ms) | peak GPU RX (GiB/s, NVML) |
|---|---|---|---|
| dram+dram | 10.0 | 0.0 | g6 58.0, g7 57.9 |
| dram+gds | 10.0 | 14.2 | g6 58.6, g7 58.6 |
| dram+phoenix | 10.0 | 13.7 | g6 59.2, g7 60.2 |
| split | 10.0 | 13.8 | g6 58.0, g7 58.0 |
| split+phoenix | 10.0 | 13.5 | g6 57.8, g7 57.9 |
| split+phoenix+w | 10.0 | 13.8 | g6 58.5, g7 58.5 |
