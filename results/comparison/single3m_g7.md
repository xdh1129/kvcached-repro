## Whole run

| path | requests ok | workload s | sum sleep s | of which pin s (max one) | sum wake s | sleep+wake excl. pin | proxy swap wait s |
|---|---|---|---|---|---|---|---|
| dram+dram | 45/45 | 268 | 44.9 | 9.7 (2.51) | 35.0 | 70.2 | 80.3 |
| dram+ssd | 45/45 | 282 | 49.8 | 0.0 (0.00) | 44.2 | 94.0 | 94.4 |
| dram+gds | 45/45 | 269 | 41.1 | 0.0 (0.00) | 38.9 | 80.0 | 80.4 |
| split | 45/45 | 281 | 51.6 | 4.9 (2.50) | 41.2 | 87.9 | 93.1 |

## Median time to first token by turn range (s)

| path | turns 2-5 | turns 6-10 | turns 11-15 |
|---|---|---|---|
| dram+dram | 2.60 | 2.68 | 2.61 |
| dram+ssd | 2.58 | 3.06 | 3.35 |
| dram+gds | 2.48 | 2.75 | 2.60 |
| split | 2.81 | 3.15 | 3.41 |

## Qwen3-32B: median sleep excl. pin / wake (s) by turn range

| path | turns 2-5: sleep / wake | turns 6-10: sleep / wake | turns 11-15: sleep / wake | KV GiB at last sleep |
|---|---|---|---|---|
| dram+dram | 1.38 / 1.34 | 1.42 / 1.39 | 1.50 / 1.44 | 8.0 |
| dram+ssd | 1.53 / 1.39 | 1.74 / 1.57 | 2.04 / 1.78 | 8.0 |
| dram+gds | 1.39 / 1.35 | 1.45 / 1.40 | 1.50 / 1.44 | 8.0 |
| split | 1.73 / 1.44 | 1.81 / 1.63 | 2.06 / 1.83 | 8.0 |

## Qwen3-14B: median sleep excl. pin / wake (s) by turn range

| path | turns 2-5: sleep / wake | turns 6-10: sleep / wake | turns 11-15: sleep / wake | KV GiB at last sleep |
|---|---|---|---|---|
| dram+dram | 0.63 / 0.62 | 0.67 / 0.65 | 0.70 / 0.68 | 4.7 |
| dram+ssd | 0.73 / 0.65 | 0.87 / 0.77 | 1.03 / 0.90 | 4.7 |
| dram+gds | 0.65 / 0.63 | 0.68 / 0.65 | 0.72 / 0.69 | 4.7 |
| split | 0.84 / 0.69 | 0.95 / 0.81 | 1.08 / 0.94 | 4.7 |

## Phi-3.5: median sleep excl. pin / wake (s) by turn range

| path | turns 2-5: sleep / wake | turns 6-10: sleep / wake | turns 11-15: sleep / wake | KV GiB at last sleep |
|---|---|---|---|---|
| dram+dram | 0.25 / 0.21 | 0.32 / 0.29 | 0.41 / 0.37 | 11.9 |
| dram+ssd | 0.45 / 0.30 | 0.84 / 0.60 | 1.23 / 0.94 | 11.9 |
| dram+gds | 0.38 / 0.24 | 0.74 / 0.54 | 1.10 / 0.86 | 11.9 |
| split | 0.32 / 0.25 | 0.40 / 0.29 | 0.51 / 0.38 | 11.9 |

## Split detail: single3m_g7_split_f0.20

Median over the turn range. gap = SSD data end minus DRAM job end (positive: the SSD finishes last).

| model | turns | on SSD GiB (share) | SSD weights GiB | sleep: DRAM end, SSD data end, gap (s) | wake: DRAM end, SSD data end, gap (s) |
|---|---|---|---|---|---|
| Qwen3-32B | 2-5 | 12.7 (20%) | 10.8 | 1.14, 1.53, +0.38 | 1.31, 1.43, +0.12 |
| Qwen3-32B | 6-10 | 13.2 (20%) | 8.9 | 1.18, 1.59, +0.42 | 1.31, 1.62, +0.30 |
| Qwen3-32B | 11-15 | 13.7 (20%) | 6.7 | 1.18, 1.84, +0.66 | 1.32, 1.82, +0.50 |
| Qwen3-14B | 2-5 | 5.9 (20%) | 4.7 | 0.53, 0.72, +0.20 | 0.60, 0.68, +0.08 |
| Qwen3-14B | 6-10 | 6.2 (20%) | 3.5 | 0.54, 0.82, +0.28 | 0.60, 0.80, +0.20 |
| Qwen3-14B | 11-15 | 6.5 (20%) | 2.3 | 0.54, 0.96, +0.43 | 0.61, 0.93, +0.32 |
| Phi-3.5 | 2-5 | 2.0 (20%) | 0.0 | 0.27, 0.34, +0.07 | 0.20, 0.24, +0.04 |
| Phi-3.5 | 6-10 | 2.8 (20%) | 0.0 | 0.28, 0.38, +0.09 | 0.29, 0.27, -0.02 |
| Phi-3.5 | 11-15 | 3.6 (20%) | 0.0 | 0.32, 0.43, +0.11 | 0.37, 0.34, -0.03 |

## Caveat on the split run (f = 0.20)

For Qwen3-32B and Qwen3-14B the tail holds weights and KV, moved as two consecutive staged GDS calls
inside the SSD job. The SSD data end matches "DRAM job end + KV tail / ~13 GiB/s" in all six cells
(e.g. 32B turns 11-15, wake: 1.32 + 7.0/13 = 1.86 vs 1.82 measured), i.e. the KV part starts only when
the DRAM job has ended: consistent with the staging cleanup after the first call blocking on the DRAM
copies. Phi-3.5's tail is KV only (one call) and shows no such gap. Not yet confirmed by a fixed run.
