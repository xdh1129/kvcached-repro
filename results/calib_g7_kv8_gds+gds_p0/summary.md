# calib_g7_kv8_gds+gds_p0

paths `gds+gds` · GPUs 7 · KV 8 GiB per model · pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct · background none · dwell 3 s · 9 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 9 | 2.979 | 0% | 28.16 | 13.0 | 8.91 | 12.5 | 3.010 | 14.6 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 9 | 2.979 | 37.07 | 12.5 | 15.5 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 7 | 8 | 3.627 | 10.9 | 10.5 |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
