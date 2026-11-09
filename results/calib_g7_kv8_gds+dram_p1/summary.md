# calib_g7_kv8_gds+dram_p1

paths `gds+dram` · GPUs 7 · KV 8 GiB per model · pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct · background none · dwell 3 s · 9 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 9 | 2.277 | 0% | 28.16 | 12.9 | 8.91 | 42.1 | 2.306 | 18.7 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 9 | 2.277 | 37.07 | 16.3 | 62.5 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 7 | 8 | 2.888 | 11.0 | 52.6 |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
