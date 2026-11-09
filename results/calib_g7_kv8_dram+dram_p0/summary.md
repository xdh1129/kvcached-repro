# calib_g7_kv8_dram+dram_p0

paths `dram+dram` · GPUs 7 · KV 8 GiB per model · pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct · background none · dwell 3 s · 9 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 9 | 0.787 | 0% | 28.16 | 53.7 | 8.91 | 53.7 | 0.815 | 53.1 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 9 | 0.787 | 37.07 | 47.1 | 62.1 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 7 | 8 | 0.853 | 52.7 | 52.7 |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
