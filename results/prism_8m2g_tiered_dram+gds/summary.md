# prism_8m2g_tiered_dram+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 1

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.57 | 88.03 | 86.49 | 9.11 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.71 | 88.88 | 48.63 | 46.96 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 29.29 | 3.98 | 0 | 0 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 44 | 22.85 | 0.4 | 1.71 | 25.92 | 0.62 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 42 | 15.0 | 0.35 | 1.53 | 17.63 | 0.44 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 42 | 22.85 | 0.4 | 1.71 | 29.96 | 4.53 | 0 | 0 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.09 | 0.75 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 15.69 | 0.38 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 40 | 6.04 | 0.3 | 1.44 | 8.34 | 0.22 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 36 | 6.04 | 0.3 | 1.44 | 8.13 | 0.11 | 0 | 0 | – |

## Requests — 754/754 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 296 | 0 | 0.166 / 0.2762 / 0.3365 | 59.2578 / 73.8148 / 76.5314 | 12.9083 / 49.3684 |
| model_2 | 22 | 0 | 0.1393 / 0.3018 / 0.5207 | 38.9552 / 57.8796 / 58.6092 | 5.7272 / 26.3143 |
| model_3 | 22 | 0 | 0.1127 / 0.2016 / 0.5057 | 38.2251 / 39.2293 / 41.5093 | 5.3984 / 29.4785 |
| model_4 | 262 | 0 | 0.1922 / 0.3548 / 0.436 | 68.2631 / 79.2676 / 82.5957 | 21.791 / 39.1683 |
| model_5 | 120 | 0 | 0.0375 / 0.1311 / 0.3078 | 10.9354 / 36.4367 / 37.4902 | 1.7857 / 16.0896 |
| model_6 | 19 | 0 | 0.091 / 0.172 / 0.3371 | 28.8647 / 36.4684 / 36.71 | 3.6859 / 15.0564 |
| model_7 | 11 | 0 | 0.0515 / 0.1374 / 0.1659 | 18.0243 / 18.2277 / 18.2523 | 1.4986 / 10.3899 |
| model_8 | 2 | 0 | 0.1106 / 0.1834 / 0.1899 | 10.6532 / 11.8707 / 11.9789 | 3.1995 / 6.0941 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 1 | 0.169 | 0.169 | 0.169 |
| wake | 8 | 0.3288 | 0.5199 | 0.5199 |

time spent swapping per GPU (s): GPU 1: 2.074, GPU 2: 0.919

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

requests: 754, triggered a wake: 8, waited > 100 ms for a swap: 12; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.1895 / 0.5257
