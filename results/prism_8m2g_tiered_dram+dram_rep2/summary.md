# prism_8m2g_tiered_dram+dram_rep2

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.39 | 4.2 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.32 | 46.27 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 33.89 | 8.44 | 53 | 77 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 40 | 15.0 | 0.35 | 1.53 | 17.8 | 0.69 | 9 | 10 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 43 | 22.85 | 0.4 | 1.71 | 36.39 | 10.39 | 117 | 332 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 36 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2638 / 2.1716 / 5.2716 | 79.8868 / 110.2458 / 128.8458 | 17.7575 / 67.0426 |
| model_2 | 44 | 0 | 0.1714 / 0.4387 / 0.6151 | 40.9444 / 59.1895 / 61.5482 | 5.8958 / 27.8932 |
| model_3 | 44 | 0 | 0.1256 / 3.7839 / 4.379 | 38.1647 / 47.442 / 123.7885 | 7.1566 / 35.1139 |
| model_4 | 524 | 0 | 0.5159 / 35.4563 / 38.7174 | 111.929 / 186.2573 / 691.9222 | 41.5578 / 80.6781 |
| model_5 | 240 | 0 | 0.0461 / 0.2023 / 0.3484 | 11.0733 / 38.0349 / 40.1863 | 1.8454 / 16.9283 |
| model_6 | 38 | 0 | 0.1273 / 0.2429 / 0.4203 | 30.4772 / 38.2536 / 39.0654 | 3.8675 / 16.2489 |
| model_7 | 22 | 0 | 0.0564 / 0.179 / 0.1822 | 18.4601 / 18.6169 / 18.6808 | 1.5396 / 11.1827 |
| model_8 | 4 | 0 | 0.1083 / 0.1863 / 0.1864 | 10.7982 / 11.9847 / 11.985 | 3.2973 / 6.3485 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 1 | 0.1067 | 0.1067 | 0.1067 |
| wake | 8 | 0.3292 | 0.5222 | 0.5241 |

time spent swapping per GPU (s): GPU 1: 2.009, GPU 2: 0.929

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 1 | 0 |
| model_2 | 1 | 1 |
| model_3 | 1 | 0 |
| model_4 | 1 | 0 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 8, waited > 100 ms for a swap: 24; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.2059 / 0.5253
