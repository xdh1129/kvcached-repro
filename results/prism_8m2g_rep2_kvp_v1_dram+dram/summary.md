# prism_8m2g_rep2_kvp_dram+dram

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.82 | 3.78 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.5 | 46.09 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 34.41 | 8.67 | 1 | 5 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 42 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 40 | 15.0 | 0.35 | 1.53 | 18.25 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 36.72 | 11.09 | 21 | 23 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 31 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 36 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2139 / 0.5189 / 0.6955 | 68.5347 / 106.2446 / 119.6683 | 15.6735 / 60.5063 |
| model_2 | 44 | 0 | 0.1581 / 0.4287 / 0.6116 | 41.2302 / 251.3274 / 367.914 | 5.8737 / 93.7221 |
| model_3 | 44 | 0 | 0.1161 / 1.7357 / 2.9722 | 38.0389 / 54.1502 / 55.0148 | 4.7148 / 37.8574 |
| model_4 | 524 | 0 | 0.2818 / 0.8625 / 2.7527 | 88.3078 / 135.1416 / 165.9167 | 28.6979 / 53.3648 |
| model_5 | 240 | 0 | 0.0514 / 0.2143 / 2.7349 | 11.077 / 37.9757 / 40.115 | 1.8356 / 16.9119 |
| model_6 | 38 | 0 | 0.1353 / 0.2576 / 0.4192 | 30.4388 / 38.2411 / 39.0999 | 3.8602 / 16.2393 |
| model_7 | 22 | 0 | 0.054 / 0.173 / 0.1775 | 18.4512 / 18.6004 / 18.7304 | 1.5364 / 11.1675 |
| model_8 | 4 | 0 | 0.6127 / 1.0317 / 1.0318 | 12.2778 / 12.6454 / 12.6581 | 4.5727 / 8.7237 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 0.3841 | 3.2989 | 3.4156 |
| wake | 14 | 0.4384 | 0.7093 | 0.7211 |

time spent swapping per GPU (s): GPU 1: 13.104, GPU 2: 0.924

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 3 | 2 |
| model_2 | 2 | 2 |
| model_3 | 3 | 2 |
| model_4 | 2 | 1 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 9, waited > 100 ms for a swap: 30; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.3089 / 3.6628
