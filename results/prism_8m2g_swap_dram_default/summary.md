# prism_8m2g_swap_dram_default

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 1

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 8.82 | 86.78 | 87.56 | 8.03 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 7.72 | 87.88 | 49.21 | 46.39 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.78 | 29.25 | 3.91 | 0 | 0 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.78 | 25.91 | 0.62 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 51 | 15.0 | 0.35 | 2.54 | 18.46 | 0.44 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.78 | 29.86 | 4.53 | 0 | 0 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 45 | 13.5 | 0.35 | 1.68 | 16.08 | 0.75 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.59 | 15.68 | 0.38 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 46 | 6.04 | 0.3 | 2.2 | 8.94 | 0.22 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.49 | 8.13 | 0.11 | 0 | 0 | – |

## Requests — 754/754 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 296 | 0 | 0.1682 / 0.2839 / 0.3437 | 59.1729 / 74.0193 / 76.4929 | 12.9821 / 49.5394 |
| model_2 | 22 | 0 | 0.1416 / 0.2677 / 0.5124 | 38.977 / 57.9579 / 58.7914 | 5.7374 / 26.2638 |
| model_3 | 22 | 0 | 0.1088 / 0.4911 / 0.9671 | 38.2591 / 39.2739 / 41.7921 | 5.3755 / 29.4907 |
| model_4 | 262 | 0 | 0.1923 / 0.3436 / 0.4192 | 68.4944 / 79.787 / 84.4809 | 21.9278 / 39.2471 |
| model_5 | 120 | 0 | 0.0389 / 0.1275 / 0.4799 | 10.9647 / 36.4266 / 37.0169 | 1.8688 / 16.1192 |
| model_6 | 19 | 0 | 0.0946 / 0.1929 / 0.441 | 28.9275 / 36.6092 / 36.6971 | 3.679 / 15.0787 |
| model_7 | 11 | 0 | 0.0561 / 0.2343 / 0.3426 | 18.0281 / 18.1703 / 18.1962 | 1.6954 / 10.4079 |
| model_8 | 2 | 0 | 0.1079 / 0.1765 / 0.1826 | 10.6797 / 11.898 / 12.0063 | 3.2052 / 6.103 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 1 | 0.5199 | 0.5199 | 0.5199 |
| wake | 8 | 0.3275 | 0.5218 | 0.5237 |

time spent swapping per GPU (s): GPU 1: 2.423, GPU 2: 0.923

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

requests: 754, triggered a wake: 8, waited > 100 ms for a swap: 13; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.2284 / 0.8701
