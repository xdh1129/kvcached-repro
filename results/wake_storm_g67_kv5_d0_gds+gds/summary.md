# wake_storm_g67_kv5_d0_gds+gds

paths `gds+gds` · GPUs 6 7 · KV 5 GiB per model · pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.2 · background meta-llama/Llama-3.2-1B · dwell 3 s · 12 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 12 | 3.095 | 100% | 10.48 | 5.8 | 5.44 | 1.8 | 3.126 | 6.7 |
| 0 | 7 | 12 | 3.107 | 95% | 10.48 | 5.4 | 5.44 | 1.9 | 3.134 | 6.3 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 12 | 3.183 | 32.14 | 10.3 | 19.8 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 6 | 11 | 3.384 | 5.1 | 4.7 |
| 7 | 11 | 3.303 | 4.8 | 4.4 |

## Background decode (TPOT ms, p50)

| GPU | no wake on GPU | during a wake |
|---|---|---|
| 6 | 2.8 (n=1805) | 2.8 (n=984) |
| 7 | 2.8 (n=1781) | 2.9 (n=1001) |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
