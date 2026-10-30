# Wake storm, GPUs 6+7, both wake at once (delta 0), 5 GiB KV per model

Per wake: 10.48 GiB weights + 5.44 GiB KV (medians), 12 rounds x 2 GPUs. Medians over both GPUs.
vLLM-default plugin settings, except that the GDS paths use TIERED_GDS_STAGE_MB=256 (256 MiB staging buffers).

| paths | wake s | resume s | weights GiB/s per GPU | KV GiB/s per GPU | both GPUs GiB/s | peak PCIe RX both GB/s | sleep s | run |
|---|---|---|---|---|---|---|---|---|
| dram+dram | 0.69 | 0.71 | 17.2 | 11.5 | 47.0 | 63.1 | 0.60 | `wake_storm_g67_kv5_d0_dram+dram` |
| dram+ssd | 1.25 | 1.28 | 26.8 | 4.8 | 25.4 | 62.9 | 1.47 | `wake_storm_g67_kv5_d0_dram+ssd` |
| dram+gds | 0.97 | 1.00 | 26.9 | 6.2 | 32.7 | 78.9 | 1.44 | `wake_storm_g67_kv5_d0_dram+gds_stage256` |
| ssd+ssd | 2.47 | 2.50 | 6.4 | 2.4 | 12.6 | 25.6 | 2.40 | `wake_storm_g67_kv5_d0_ssd+ssd` |
| gds+gds | 2.44 | 2.47 | 5.5 | 2.4 | 12.5 | 18.3 | 2.60 | `wake_storm_g67_kv5_d0_gds+gds_stage256` |
| vllm | 0.41 | 1.92 | – | – | – | 61.7 | 0.37 | `wake_storm_g67_kv5_d0_vllm` |

vllm: no plugin log, so no per-path bytes; KV is dropped on sleep and re-prefilled on wake (resume includes it).
