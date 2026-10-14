# Offload paths on the Prism trace (prism_8m2g_rep2_sched_*)

## Requests

| config | ok / total | TTFT p50 / p95 / p99 s (GPU 1) | E2E p50 / p99 s (GPU 1) | TPOT p50 ms (GPU 1) | vLLM preemptions (GPU 1) | peak GiB (GPU 1) |
|---|---|---|---|---|---|---|
| dram+recompute | 1508 / 1508 | 0.24 / 0.66 / 4.52 | 22.10 / 75.39 | 79.4 | 392 | 91.5 |
| dram+dram | 1508 / 1508 | 0.24 / 0.64 / 0.87 | 21.38 / 65.36 | 78.5 | 17 | 91.75 |
| ssd+ssd | 1508 / 1508 | 0.24 / 0.87 / 2.43 | 21.00 / 62.85 | 78.3 | 17 | 91.13 |
| gds+gds | 1508 / 1508 | 0.24 / 1.10 / 3.51 | 20.87 / 62.00 | 75.8 | 9 | 90.82 |
| dram+ssd | 1508 / 1508 | 0.24 / 0.63 / 0.86 | 21.38 / 66.29 | 78.5 | 21 | 91.49 |
| dram+gds | 1508 / 1508 | 0.24 / 0.67 / 0.89 | 21.27 / 67.98 | 78.2 | 25 | 91.48 |

## Swaps and restores

| config | wakes | sleeps (busy) | frozen requests | restores with KV | KV GB p50 | restore s p50 / p95 / max | restore with KV s p50 | weights path s p50 | KV path s p50 | swap time s (GPU 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| dram+recompute | 14 | 7 (6) | 372 | 0 / 14 | nan | 0.424 / 0.514 / 0.514 | nan | 0.365 | nan | 6.0 |
| dram+dram | 14 | 7 (6) | 354 | 6 / 14 | 4.20 | 0.435 / 0.701 / 0.731 | 0.591 | 0.374 | 0.203 | 7.2 |
| ssd+ssd | 14 | 7 (6) | 336 | 6 / 14 | 4.19 | 1.877 / 3.045 / 3.355 | 2.201 | 1.799 | 2.089 | 25.5 |
| gds+gds | 14 | 7 (5) | 272 | 5 / 14 | 5.96 | 2.993 / 4.700 / 4.766 | 4.176 | 2.920 | 4.041 | 38.9 |
| dram+ssd | 14 | 7 (6) | 354 | 6 / 14 | 4.20 | 0.438 / 1.185 / 1.303 | 0.783 | 0.370 | 0.709 | 11.0 |
| dram+gds | 14 | 7 (6) | 350 | 6 / 14 | 4.20 | 0.476 / 1.729 / 1.754 | 0.952 | 0.376 | 0.876 | 13.8 |

## Data moved during the trace (GB)

| config | KV offloaded | DRAM: GPU -> host | DRAM: host -> GPU | SSD I/O |
|---|---|---|---|---|
| dram+recompute | 0.00 | 4.15 | 269.42 | 0.00 |
| dram+dram | 30.58 | 34.73 | 300.49 | 0.00 |
| ssd+ssd | 29.88 | 34.03 | 299.79 | 333.82 |
| gds+gds | 23.76 | 0.00 | 0.00 | 321.57 |
| dram+ssd | 30.52 | 34.67 | 300.43 | 61.52 |
| dram+gds | 30.35 | 4.15 | 269.42 | 61.19 |
