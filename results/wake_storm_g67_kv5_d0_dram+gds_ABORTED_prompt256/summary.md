# wake_storm_g67_kv5_d0_dram+gds

paths `dram+gds` · GPUs 6 7 · KV 5 GiB per model · pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.2 · background meta-llama/Llama-3.2-1B · dwell 3 s · 12 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 3 | 1.211 | 96% | 7.31 | 25.0 | 5.38 | 5.1 | – | 14.4 |
| 0 | 7 | 3 | 1.208 | 100% | 7.31 | 30.0 | 5.38 | 5.1 | – | 13.2 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 3 | 1.211 | 26.11 | 21.6 | 73.2 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 6 | 3 | 2.103 | 27.0 | 5.2 |
| 7 | 3 | 2.012 | 27.0 | 5.5 |

## Background decode (TPOT ms, p50)

| GPU | no wake on GPU | during a wake |
|---|---|---|
| 6 | 3.0 (n=503) | 3.0 (n=122) |
| 7 | 3.0 (n=512) | 3.0 (n=112) |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
