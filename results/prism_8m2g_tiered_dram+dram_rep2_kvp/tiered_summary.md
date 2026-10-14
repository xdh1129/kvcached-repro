# Tiered swap summary: prism_8m2g_tiered_dram+dram_rep2_kvp

paths `tiered paths=dram+dram weights=dram kv=dram parallel=1` · time scale 0.2 · replication 2 · min resident 5 s

## Swaps

| GPU | wakes | sleeps | busy sleeps (frozen requests) | frozen requests total |
|---|---|---|---|---|
| 1 | 14 | 11 | 10 | 1033 |
| 2 | 4 | 0 | 0 | 0 |
| all | 18 | 11 | 10 | 1033 |

## Restores

18 restores, 10 of them with > 0.2 GB of KV.

| | n | weights+state GB p50 | KV GB p50 / max | weights path s p50 | KV path s p50 / max | restore s p50 / p95 / max |
|---|---|---|---|---|---|---|
| all restores | 18 | 25.13 | 0.64 / 12.08 | 0.448 | 0.086 / 0.633 | 0.522 / 0.735 / 0.737 |
| restores with KV | 10 | 25.13 | 8.56 / 12.08 | 0.587 | 0.428 / 0.633 | 0.670 / 0.736 / 0.737 |

## Data moved during the trace

| | GB |
|---|---|
| KV offloaded on sleep (any path) | 80.21 |
| through host DRAM, GPU -> host (sleep) | 86.75 |
| through host DRAM, host -> GPU (wake) | 459.07 |
