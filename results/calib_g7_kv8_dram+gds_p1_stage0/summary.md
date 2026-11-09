# calib_g7_kv8_dram+gds_p1_stage0

paths `dram+gds` · GPUs 7 · KV 8 GiB per model · pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct · background none · dwell 3 s · 9 rounds per delta

## Wakes per delta and GPU (medians)

| delta s | GPU | n | wake s | overlap | weights GiB | weights GiB/s | KV GiB | KV GiB/s | resume s | PCIe RX GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 9 | 1.990 | 0% | 28.16 | 51.6 | 8.91 | 4.8 | 2.028 | 20.0 |

## Rounds: all GPUs together (medians)

| delta s | rounds | span s (first start → last end) | GiB moved | aggregate GiB/s | peak summed PCIe RX GB/s |
|---|---|---|---|---|---|
| 0 | 9 | 1.990 | 37.07 | 19.3 | 62.3 |

## Sleeps (all rounds, medians)

| GPU | n | sleep s | weights save GiB/s | KV save GiB/s |
|---|---|---|---|---|
| 7 | 8 | 2.277 | 52.7 | 6.4 |

overlap: share of the wake during which another GPU was also waking. GiB/s per path from the plugin's per-path start/end. resume: wake start to the first token of the frozen requests. PCIe RX from NVML (~20 ms windows).
