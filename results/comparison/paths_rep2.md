# Offload paths on the Prism trace (prism_8m2g_rep2_kvp_*)

## Requests

| config | ok / total | TTFT p50 / p95 / p99 s (GPU 1) | E2E p50 / p99 s (GPU 1) | TPOT p50 ms (GPU 1) | vLLM preemptions (GPU 1) | peak GiB (GPU 1) |
|---|---|---|---|---|---|---|
| dram+recompute | 1508 / 1508 | 0.24 / 1.72 / 27.74 | 23.45 / 78.55 | 78.1 | 504 | 91.55 |
| dram+dram | 1508 / 1508 | 0.25 / 1.34 / 6.15 | 22.54 / 60.75 | 79.9 | 43 | 91.47 |
| ssd+ssd | 1508 / 1508 | 0.25 / 2.06 / 7.10 | 22.12 / 60.52 | 80.5 | 52 | 91.07 |
| gds+gds | 1508 / 1508 | 0.25 / 14.17 / 24.82 | 23.25 / 64.13 | 76.9 | 43 | 91.12 |
| dram+ssd | 1508 / 1508 | 0.24 / 0.65 / 0.94 | 21.40 / 66.05 | 78.3 | 21 | 91.34 |
| dram+gds | 1508 / 1508 | 0.24 / 0.66 / 0.93 | 21.46 / 62.89 | 78.1 | 24 | 91.44 |

## Swaps and restores

| config | wakes | sleeps (busy) | frozen requests | restores with KV | KV GB p50 | restore s p50 / p95 / max | restore with KV s p50 | weights path s p50 | KV path s p50 | swap time s (GPU 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| dram+recompute | 16 | 9 (8) | 584 | 0 / 16 | nan | 0.423 / 0.515 / 0.515 | nan | 0.365 | nan | 10.0 |
| dram+dram | 16 | 9 (8) | 634 | 8 / 16 | 4.03 | 0.431 / 0.728 / 0.730 | 0.584 | 0.374 | 0.202 | 16.3 |
| ssd+ssd | 15 | 8 (7) | 590 | 7 / 15 | 7.21 | 1.870 / 2.890 / 3.319 | 2.402 | 1.796 | 2.286 | 29.5 |
| gds+gds | 13 | 6 (5) | 394 | 5 / 13 | 7.05 | 3.439 / 5.168 / 5.552 | 4.829 | 3.359 | 4.704 | 40.6 |
| dram+ssd | 14 | 7 (6) | 354 | 6 / 14 | 4.20 | 0.437 / 1.185 / 1.306 | 0.784 | 0.370 | 0.710 | 10.9 |
| dram+gds | 14 | 7 (6) | 388 | 6 / 14 | 4.19 | 0.451 / 1.702 / 1.705 | 0.864 | 0.374 | 0.789 | 13.3 |

## Data moved during the trace (GB)

| config | KV offloaded | DRAM: GPU -> host | DRAM: host -> GPU | SSD I/O |
|---|---|---|---|---|
| dram+recompute | 0.00 | 5.33 | 311.25 | 0.00 |
| dram+dram | 41.37 | 46.70 | 353.11 | 0.00 |
| ssd+ssd | 40.85 | 45.60 | 335.89 | 381.49 |
| gds+gds | 28.40 | 0.00 | 0.00 | 313.59 |
| dram+ssd | 30.58 | 34.73 | 300.49 | 61.66 |
| dram+gds | 28.37 | 4.15 | 269.42 | 57.23 |
