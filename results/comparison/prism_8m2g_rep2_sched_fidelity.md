# Schedule replay fidelity (prism_8m2g_rep2_sched_*)

Reference: `results/schedules/ref_dram+ssd`, 21 swaps (7 sleeps, 14 wakes).

| config | swaps run / scheduled | extra (policy fallback) | missing | lag s p50 / max | busy sleeps | frozen requests | KV offloaded GB |
|---|---|---|---|---|---|---|---|
| dram+recompute | 21 / 21 | 0 | 0 | 0.00 / 1.88 | 6 | 372 | 0.0 |
| dram+dram | 21 / 21 | 0 | 0 | 0.00 / 0.00 | 6 | 354 | 30.6 |
| ssd+ssd | 21 / 21 | 0 | 0 | 0.00 / 3.90 | 6 | 336 | 29.9 |
| gds+gds | 21 / 21 | 0 | 0 | 0.64 / 8.93 | 5 | 272 | 23.8 |
| dram+ssd | 21 / 21 | 0 | 0 | 0.00 / 0.07 | 6 | 354 | 30.5 |
| dram+gds | 21 / 21 | 0 | 0 | 0.00 / 1.46 | 6 | 350 | 30.3 |
