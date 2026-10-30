# wake_storm_g67_kv5_d0_dram+gds_stage256

paths `dram+gds` · GPUs 6 7 · KV 5 GiB per model · pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.2 · background meta-llama/Llama-3.2-1B · dwell 3 s · 12 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 12 | 0.977 | 100% | 10.48 | 27.0 | 5.44 | 6.2 | 0.998 | 18.9 |
| 0 | 7 | 12 | 0.965 | 100% | 10.48 | 26.8 | 5.44 | 6.2 | 0.993 | 19.9 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 12 | 0.979 | 32.14 | 32.7 | 78.9 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 6 | 11 | 1.468 | 26.8 | 5.6 |
| 7 | 11 | 1.421 | 26.6 | 5.4 |

## Background decode (TPOT ms, p50)

| GPU | no wake on GPU | during a wake |
|---|---|---|
| 6 | 3.0 (n=961) | 3.3 (n=399) |
| 7 | 3.0 (n=914) | 3.4 (n=414) |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
