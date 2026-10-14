# Tiered swap summary: prism_8m2g_tiered_dram+dram_rep2_kvp

paths `tiered paths=dram+dram weights=dram kv=dram parallel=1` · time scale 0.2 · replication 2 · min resident 5 s

## Swaps

| GPU | wakes | sleeps | busy sleeps (frozen requests) | frozen requests total |
|---|---|---|---|---|
| 1 | 12 | 10 | 8 | 824 |
| 2 | 4 | 0 | 0 | 0 |
| all | 16 | 10 | 8 | 824 |

## Restores

16 restores, 8 of them with > 0.2 GB of KV.

| | n | weights+state GB p50 | KV GB p50 / max | weights path s p50 | KV path s p50 / max | restore s p50 / p95 / max |
|---|---|---|---|---|---|---|
| all restores | 16 | 25.13 | 0.38 / 13.17 | 0.441 | 0.082 / 0.657 | 0.512 / 0.691 / 0.751 |
| restores with KV | 8 | 25.13 | 7.42 / 13.17 | 0.569 | 0.387 / 0.657 | 0.646 / 0.723 / 0.751 |

## Data moved during the trace

| | GB |
|---|---|
| KV offloaded on sleep (any path) | 53.03 |
| through host DRAM, GPU -> host (sleep) | 58.97 |
| through host DRAM, host -> GPU (wake) | 381.56 |
