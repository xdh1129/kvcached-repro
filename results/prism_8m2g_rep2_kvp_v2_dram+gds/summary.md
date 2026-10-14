# prism_8m2g_rep2_kvp_dram+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.91 | 3.68 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.38 | 46.21 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 34.19 | 8.59 | 37 | 20 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 37 | 15.0 | 0.35 | 1.53 | 18.2 | 0.88 | 1 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.95 | 11.48 | 12 | 23 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 35 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2328 / 1.676 / 2.9869 | 69.8908 / 98.9616 / 110.8748 | 16.1952 / 60.0238 |
| model_2 | 44 | 0 | 0.1796 / 0.4227 / 0.5993 | 41.3056 / 61.603 / 220.7426 | 5.9295 / 75.2019 |
| model_3 | 44 | 0 | 0.1052 / 0.8645 / 1.6822 | 38.505 / 47.1804 / 47.5717 | 5.7581 / 34.0552 |
| model_4 | 524 | 0 | 0.2941 / 0.937 / 4.1331 | 99.4893 / 162.8895 / 245.8526 | 34.543 / 62.1347 |
| model_5 | 240 | 0 | 0.0456 / 0.2056 / 1.3542 | 11.0612 / 38.0161 / 40.1357 | 1.9489 / 16.9645 |
| model_6 | 38 | 0 | 0.1347 / 0.2397 / 0.4172 | 30.4724 / 38.2629 / 39.0951 | 3.8716 / 16.2538 |
| model_7 | 22 | 0 | 0.0566 / 0.1791 / 0.1833 | 18.5356 / 18.8128 / 18.8661 | 1.5491 / 11.166 |
| model_8 | 4 | 0 | 0.1198 / 0.2043 / 0.2054 | 11.4757 / 12.1884 / 12.1918 | 3.6865 / 7.1153 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 8 | 0.2686 | 1.8518 | 1.897 |
| wake | 15 | 0.3974 | 2.04 | 2.1187 |

time spent swapping per GPU (s): GPU 1: 15.263, GPU 2: 0.934

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 2 | 1 |
| model_2 | 2 | 2 |
| model_3 | 4 | 3 |
| model_4 | 3 | 2 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 12, waited > 100 ms for a swap: 48; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 1.138 / 3.9017
