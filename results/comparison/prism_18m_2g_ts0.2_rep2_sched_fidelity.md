# Schedule replay fidelity (prism_18m_2g_ts0.2_rep2_sched_*)

Reference: `results/prism_18m_2g_ts0.2_rep2_ref_dram+ssd`, 64 swaps (26 sleeps, 38 wakes).

| config | swaps run / scheduled | sleeps pulled early | extra (policy fallback) | missing | lag s p50 / max | busy sleeps | frozen requests | KV offloaded GB |
|---|---|---|---|---|---|---|---|---|
| vllm | 64 / 64 | 8 | 0 | 0 | 0.00 / 2.60 | 20 | 1419 | 0.0 |
| dram+dram | 64 / 64 | 2 | 0 | 0 | 0.00 / 6.02 | 20 | 1265 | 201.9 |
| ssd+ssd | 64 / 64 | 2 | 0 | 0 | 0.96 / 4.47 | 20 | 1130 | 195.2 |
| gds+gds | 54 / 64 | 2 | 0 | 10 | 10.00 / 44.77 | 12 | 467 | 102.1 |
| dram+ssd | 64 / 64 | 2 | 0 | 0 | 0.00 / 3.17 | 20 | 1304 | 204.4 |
| dram+gds | 64 / 64 | 3 | 0 | 0 | 0.05 / 29.96 | 16 | 1011 | 162.3 |
