# wake_storm_g67_kv5_d0_dram+ssd

paths `dram+ssd` · GPUs 6 7 · KV 5 GiB per model · pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.2 · background meta-llama/Llama-3.2-1B · dwell 3 s · 12 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 12 | 1.258 | 100% | 10.48 | 26.8 | 5.44 | 4.8 | 1.286 | 14.4 |
| 0 | 7 | 12 | 1.251 | 100% | 10.48 | 26.8 | 5.44 | 4.8 | 1.279 | 14.2 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 12 | 1.258 | 32.14 | 25.4 | 62.9 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 6 | 11 | 1.478 | 26.8 | 5.6 |
| 7 | 11 | 1.470 | 26.1 | 5.7 |

## Background decode (TPOT ms, p50)

| GPU | no wake on GPU | during a wake |
|---|---|---|
| 6 | 2.9 (n=956) | 3.1 (n=484) |
| 7 | 3.1 (n=910) | 3.2 (n=495) |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
