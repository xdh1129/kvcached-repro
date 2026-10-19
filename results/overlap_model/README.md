# Overlap model: dram+ssd vs dram+gds on GPUs 6+7

## Validation against the microbenchmarks (s)

| event | KV path | quantity | measured | model |
|---|---|---|---|---|
| t=60.7 s: A model_17 / B model_16 | ssd | A weights with B | 0.414 | 0.414 |
| t=60.7 s: A model_17 / B model_16 | ssd | B wake alone | 0.340 | 0.323 |
| t=60.7 s: A model_17 / B model_16 | ssd | B wake with A | 0.457 | 0.436 |
| t=60.7 s: A model_17 / B model_16 | gds | A weights with B | 0.368 | 0.368 |
| t=60.7 s: A model_17 / B model_16 | gds | B wake alone | 0.283 | 0.283 |
| t=60.7 s: A model_17 / B model_16 | gds | B wake with A | 0.288 | 0.283 |
| t=60.7 s: A model_17 / B model_16 | - | A weights alone | 0.255 | 0.255 |
| typical: A Phi-3 / B Phi-3, 5 GB KV | ssd | A weights with B | 0.270 | 0.267 |
| typical: A Phi-3 / B Phi-3, 5 GB KV | ssd | B wake alone | 0.464 | 0.447 |
| typical: A Phi-3 / B Phi-3, 5 GB KV | ssd | B wake with A | 0.592 | 0.577 |
| typical: A Phi-3 / B Phi-3, 5 GB KV | gds | A weights with B | 0.265 | 0.265 |
| typical: A Phi-3 / B Phi-3, 5 GB KV | gds | B wake alone | 0.389 | 0.396 |
| typical: A Phi-3 / B Phi-3, 5 GB KV | gds | B wake with A | 0.396 | 0.396 |
| typical: A Phi-3 / B Phi-3, 5 GB KV | - | A weights alone | 0.135 | 0.135 |
| large KV: A Llama-3.1-8B / B Phi-3, 20 GB KV | ssd | A weights with B | 0.473 | 0.471 |
| large KV: A Llama-3.1-8B / B Phi-3, 20 GB KV | ssd | B wake alone | 1.534 | 1.514 |
| large KV: A Llama-3.1-8B / B Phi-3, 20 GB KV | ssd | B wake with A | 1.662 | 1.644 |
| large KV: A Llama-3.1-8B / B Phi-3, 20 GB KV | gds | A weights with B | 0.420 | 0.420 |
| large KV: A Llama-3.1-8B / B Phi-3, 20 GB KV | gds | B wake alone | 1.445 | 1.522 |
| large KV: A Llama-3.1-8B / B Phi-3, 20 GB KV | gds | B wake with A | 1.458 | 1.522 |
| large KV: A Llama-3.1-8B / B Phi-3, 20 GB KV | - | A weights alone | 0.290 | 0.290 |

## Replay of the wakes of `results/prism_18m_2g67_balanced_ts0.2_rep3_ref_dram+ssd`

51 wakes over 156 s, 30 with KV > 1 GiB; 4 of those overlap a wake on the other GPU (model, dram+ssd).

| KV path | wake s mean | wake s sum | KV wake s mean |
|---|---|---|---|
| ssd | 0.731 | 37.28 | 1.148 |
| gds | 0.720 | 36.70 | 1.127 |

## What-if sweep (Poisson wakes, sizes drawn from the run)

Base wake rates: GPU 6 0.205/s, GPU 7 0.122/s. rate x: more frequent wakes; RAID x: storage bandwidth (13.1 GiB/s x n, e.g. more NVMe drives); size x: weights and KV scaled. Mean over 20 seeds; wake time includes queueing behind the GPU's previous wake.

| rate x | RAID x | size x | KV wakes overlapping the other GPU | wake s mean ssd / gds | gain | KV wake s mean ssd / gds | gain |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 19% | 0.682 / 0.676 | 1% | 1.033 / 1.025 | 1% |
| 1 | 1 | 2 | 37% | 1.490 / 1.433 | 4% | 2.199 / 2.117 | 4% |
| 1 | 2 | 1 | 13% | 0.402 / 0.387 | 4% | 0.577 / 0.555 | 4% |
| 1 | 2 | 2 | 24% | 0.847 / 0.793 | 6% | 1.180 / 1.109 | 6% |
| 1 | 4 | 1 | 9% | 0.268 / 0.272 | -1% | 0.361 / 0.368 | -2% |
| 1 | 4 | 2 | 18% | 0.569 / 0.555 | 2% | 0.749 / 0.733 | 2% |
| 2 | 1 | 1 | 31% | 0.681 / 0.674 | 1% | 1.064 / 1.055 | 1% |
| 2 | 1 | 2 | 54% | 1.596 / 1.537 | 4% | 2.424 / 2.348 | 3% |
| 2 | 2 | 1 | 19% | 0.397 / 0.381 | 4% | 0.582 / 0.559 | 4% |
| 2 | 2 | 2 | 40% | 0.866 / 0.797 | 8% | 1.225 / 1.137 | 7% |
| 2 | 4 | 1 | 14% | 0.269 / 0.270 | -0% | 0.368 / 0.371 | -1% |
| 2 | 4 | 2 | 27% | 0.582 / 0.555 | 5% | 0.774 / 0.739 | 4% |
| 4 | 1 | 1 | 54% | 0.768 / 0.763 | 1% | 1.214 / 1.213 | 0% |
| 4 | 1 | 2 | 63% | 1.749 / 1.709 | 2% | 2.683 / 2.643 | 1% |
| 4 | 2 | 1 | 39% | 0.431 / 0.407 | 5% | 0.635 / 0.606 | 4% |
| 4 | 2 | 2 | 63% | 0.995 / 0.891 | 10% | 1.407 / 1.288 | 8% |
| 4 | 4 | 1 | 28% | 0.289 / 0.278 | 4% | 0.400 / 0.386 | 3% |
| 4 | 4 | 2 | 50% | 0.677 / 0.588 | 13% | 0.905 / 0.788 | 13% |
| 8 | 1 | 1 | 59% | 0.825 / 0.823 | 0% | 1.310 / 1.319 | -1% |
| 8 | 1 | 2 | 57% | 1.697 / 1.647 | 3% | 2.605 / 2.553 | 2% |
| 8 | 2 | 1 | 65% | 0.480 / 0.440 | 8% | 0.699 / 0.659 | 6% |
| 8 | 2 | 2 | 62% | 1.027 / 0.920 | 10% | 1.440 / 1.336 | 7% |
| 8 | 4 | 1 | 50% | 0.326 / 0.288 | 12% | 0.449 / 0.396 | 12% |
| 8 | 4 | 2 | 66% | 0.797 / 0.634 | 20% | 1.056 / 0.836 | 21% |
