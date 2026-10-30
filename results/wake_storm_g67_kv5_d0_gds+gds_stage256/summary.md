# wake_storm_g67_kv5_d0_gds+gds_stage256

paths `gds+gds` · GPUs 6 7 · KV 5 GiB per model · pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.2 · background meta-llama/Llama-3.2-1B · dwell 3 s · 12 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 12 | 2.444 | 96% | 10.48 | 5.2 | 5.44 | 2.4 | 2.471 | 7.2 |
| 0 | 7 | 12 | 2.446 | 100% | 10.48 | 5.7 | 5.44 | 2.4 | 2.473 | 7.8 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 12 | 2.572 | 32.14 | 12.5 | 18.3 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 6 | 11 | 2.600 | 5.5 | 5.7 |
| 7 | 11 | 2.601 | 5.4 | 5.6 |

## Background decode (TPOT ms, p50)

| GPU | no wake on GPU | during a wake |
|---|---|---|
| 6 | 2.9 (n=1583) | 2.9 (n=833) |
| 7 | 2.9 (n=1591) | 3.0 (n=793) |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
