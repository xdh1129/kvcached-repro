# wake_storm_smoke_g1

paths `dram+gds` · GPUs 1 · KV 0.5 GiB per model · pool meta-llama/Llama-3.2-1B,meta-llama/Llama-3.2-1B-Instruct · background meta-llama/Llama-3.2-1B · dwell 2 s · 2 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 2 | 0.190 | 0% | 2.86 | 47.2 | 0.56 | 3.5 | 0.199 | 35.8 |
| seq | 1 | 2 | 0.184 | 0% | 2.86 | 35.8 | 0.56 | 3.4 | 0.193 | 17.5 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 2 | 0.190 | 3.42 | 18.4 | 40.3 |
| seq | 2 | 0.184 | 3.42 | 18.5 | 28.7 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 1 | 3 | 0.270 | 52.1 | 3.4 |

## Background decode (TPOT ms, p50)

| GPU | no wake on GPU | during a wake |
|---|---|---|
| 1 | 3.2 (n=38) | 3.5 (n=18) |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
