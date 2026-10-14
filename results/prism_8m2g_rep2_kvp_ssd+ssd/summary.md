# prism_8m2g_rep2_kvp_ssd+ssd

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.31 | 88.28 | 91.07 | 4.53 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.6 | 89.0 | 49.13 | 46.46 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 34.13 | 8.67 | 28 | 28 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 38 | 15.0 | 0.35 | 1.53 | 18.19 | 0.88 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.73 | 11.09 | 10 | 24 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 31 | 13.5 | 0.35 | 1.53 | 16.98 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2405 / 2.4289 / 7.8965 | 69.2312 / 102.8092 / 115.3743 | 15.968 / 60.6601 |
| model_2 | 44 | 0 | 0.1805 / 1.4318 / 2.0585 | 41.445 / 62.0004 / 206.6571 | 6.2559 / 70.4384 |
| model_3 | 44 | 0 | 0.1216 / 1.452 / 2.0637 | 37.4065 / 47.7893 / 269.0482 | 4.9468 / 30.0025 |
| model_4 | 524 | 0 | 0.2829 / 1.8788 / 6.2475 | 92.4419 / 125.4982 / 150.4553 | 32.6502 / 58.7107 |
| model_5 | 240 | 0 | 0.0601 / 0.7298 / 2.3896 | 11.3735 / 38.1692 / 40.0707 | 2.1102 / 16.912 |
| model_6 | 38 | 0 | 0.135 / 0.5163 / 2.2415 | 31.2877 / 38.5547 / 39.0728 | 3.8496 / 16.2339 |
| model_7 | 22 | 0 | 0.0604 / 1.3282 / 1.397 | 18.3314 / 18.7814 / 18.9074 | 2.1028 / 11.1491 |
| model_8 | 4 | 0 | 2.9018 / 4.6123 / 4.6133 | 12.1354 / 12.3168 / 12.3301 | 6.8735 / 8.9208 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 8 | 0.6271 | 1.4231 | 1.437 |
| wake | 15 | 1.8787 | 2.8986 | 3.3255 |

time spent swapping per GPU (s): GPU 1: 29.459, GPU 2: 5.445

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 3 | 2 |
| model_2 | 2 | 2 |
| model_3 | 3 | 2 |
| model_4 | 3 | 2 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 12, waited > 100 ms for a swap: 87; swap wait p50/p95/p99/max s: 0.0 / 0.7919 / 3.3269 / 6.263
