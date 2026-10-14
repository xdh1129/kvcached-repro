# Tiered swap summary: prism_8m2g_tiered_dram+gds

paths `tiered paths=dram+gds weights=dram kv=gds parallel=1` · time scale 0.2 · replication 1 · min resident 5 s

## Swaps

| GPU | wakes | sleeps | busy sleeps (frozen requests) | frozen requests total |
|---|---|---|---|---|
| 1 | 4 | 1 | 0 | 0 |
| 2 | 4 | 0 | 0 | 0 |
| all | 8 | 1 | 0 | 0 |

## Restores

8 restores, 0 of them with > 0.2 GB of KV.

| | n | weights+state GB p50 | KV GB p50 / max | weights path s p50 | KV path s p50 / max | restore s p50 / p95 / max |
|---|---|---|---|---|---|---|
| all restores | 8 | 15.68 | 0.07 / 0.08 | 0.276 | 0.024 / 0.069 | 0.323 / 0.514 / 0.514 |

## Data moved during the trace

| | GB |
|---|---|
| KV offloaded on sleep (any path) | 0.08 |
| through host DRAM, GPU -> host (sleep) | 0.60 |
| through host DRAM, host -> GPU (wake) | 135.52 |
