# prism_8m2g_rep2_kvp_dram+dram

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.47 | 4.13 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.19 | 46.4 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 33.97 | 8.67 | 19 | 14 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 38 | 15.0 | 0.35 | 1.53 | 18.19 | 0.88 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.49 | 10.94 | 21 | 29 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.18 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2277 / 1.1513 / 5.4638 | 70.5473 / 103.998 / 119.5575 | 16.4231 / 60.7985 |
| model_2 | 44 | 0 | 0.1712 / 0.4357 / 0.582 | 41.4358 / 61.7612 / 208.0167 | 5.9155 / 70.8882 |
| model_3 | 44 | 0 | 0.12 / 0.6779 / 1.554 | 37.4627 / 47.2094 / 47.2473 | 4.3582 / 35.6145 |
| model_4 | 524 | 0 | 0.2921 / 1.7978 / 6.4143 | 95.084 / 135.949 / 170.2636 | 33.9732 / 60.1234 |
| model_5 | 240 | 0 | 0.0471 / 0.2139 / 4.2661 | 11.0041 / 37.9885 / 40.1698 | 1.8652 / 16.941 |
| model_6 | 38 | 0 | 0.1277 / 0.2441 / 0.422 | 30.4233 / 38.2833 / 39.0883 | 3.8645 / 16.2418 |
| model_7 | 22 | 0 | 0.0613 / 0.18 / 0.1836 | 18.4369 / 18.5834 / 18.7063 | 1.5397 / 11.1649 |
| model_8 | 4 | 0 | 2.8786 / 5.569 / 5.5704 | 10.617 / 12.6201 / 12.6503 | 5.8219 / 11.2228 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 9 | 0.3833 | 3.3444 | 3.5482 |
| wake | 16 | 0.4371 | 0.7383 | 0.744 |

time spent swapping per GPU (s): GPU 1: 16.258, GPU 2: 0.929

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 3 | 2 |
| model_2 | 2 | 2 |
| model_3 | 4 | 3 |
| model_4 | 3 | 2 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 12, waited > 100 ms for a swap: 60; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 4.2916 / 6.4256
