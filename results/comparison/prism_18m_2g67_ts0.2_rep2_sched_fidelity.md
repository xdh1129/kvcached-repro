# Schedule replay fidelity (prism_18m_2g67_ts0.2_rep2_sched_*)

Reference: `results/prism_18m_2g67_ts0.2_rep2_ref_dram+ssd`, 62 swaps (25 sleeps, 37 wakes).

| config | swaps run / scheduled | sleeps pulled early | extra (policy fallback) | missing | lag s p50 / max | busy sleeps | frozen requests | KV offloaded GB |
|---|---|---|---|---|---|---|---|---|
| vllm | 62 / 62 | 2 | 0 | 0 | 0.00 / 3.13 | 19 | 1596 | 0.0 |
| dram+dram | 62 / 62 | 3 | 0 | 0 | 0.00 / 7.76 | 19 | 1346 | 221.2 |
| ssd+ssd | 62 / 62 | 1 | 0 | 0 | 0.84 / 5.40 | 17 | 1316 | 215.6 |
| gds+gds | 61 / 62 | 2 | 0 | 1 | 6.39 / 38.81 | 11 | 587 | 109.4 |
| gds+gds_stage256 | 62 / 62 | 1 | 0 | 0 | 1.35 / 7.04 | 17 | 1320 | 218.7 |
| dram+ssd | 62 / 62 | 1 | 0 | 0 | 0.00 / 3.75 | 18 | 1404 | 222.6 |
| dram+gds | 62 / 62 | 4 | 0 | 0 | 1.21 / 27.88 | 15 | 1073 | 189.6 |
| dram+gds_stage256 | 62 / 62 | 3 | 0 | 0 | 0.00 / 3.82 | 19 | 1390 | 223.3 |
