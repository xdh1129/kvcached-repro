# wake_storm_g67_kv5_d0_dram+gds

paths `dram+gds` · GPUs 6 7 · KV 5 GiB per model · pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.2 · background meta-llama/Llama-3.2-1B · dwell 3 s · 12 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 12 | 1.790 | 100% | 10.48 | 26.0 | 5.44 | 3.3 | 1.819 | 10.7 |
| 0 | 7 | 12 | 1.868 | 90% | 10.48 | 26.7 | 5.44 | 3.1 | 1.900 | 10.6 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 12 | 1.879 | 32.14 | 17.2 | 70.8 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 6 | 11 | 1.837 | 27.0 | 4.0 |
| 7 | 11 | 2.197 | 26.9 | 3.5 |

## Background decode (TPOT ms, p50)

| GPU | no wake on GPU | during a wake |
|---|---|---|
| 6 | 2.9 (n=1188) | 3.0 (n=540) |
| 7 | 2.9 (n=1103) | 3.0 (n=641) |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
