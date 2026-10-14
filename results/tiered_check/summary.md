| config (weights+KV) | checksums after restore | control caught zeroed KV | weights+state GB | KV GB | weights path s | KV path s | restore s (map + loads) | through host DRAM GB |
|---|---|---|---|---|---|---|---|---|
| dram+dram | all match | yes | 16.70 | 2.08 | 0.326 | 0.139 | 0.372 | 18.78 |
| ssd+ssd | all match | yes | 16.70 | 2.08 | 1.339 | 1.300 | 1.386 | 18.78 |
| gds+gds | all match | yes | 16.70 | 2.08 | 1.411 | 1.376 | 1.461 | 0.00 |
| dram+ssd | all match | yes | 16.70 | 2.08 | 0.294 | 0.393 | 0.443 | 18.78 |
| dram+gds | all match | yes | 16.70 | 2.08 | 0.326 | 0.166 | 0.376 | 16.70 |
