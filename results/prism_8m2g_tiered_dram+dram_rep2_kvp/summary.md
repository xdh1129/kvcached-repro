# prism_8m2g_tiered_dram+dram_rep2_kvp

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.89 | 3.7 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.17 | 46.42 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 43 | 22.85 | 0.4 | 1.71 | 34.39 | 8.59 | 25 | 38 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 42 | 15.0 | 0.35 | 1.53 | 18.2 | 0.88 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 42 | 22.85 | 0.4 | 1.71 | 37.11 | 11.33 | 24 | 31 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 39 | 6.04 | 0.3 | 1.44 | 8.18 | 0.16 | 0 | 0 | – |

## Requests — 1507/1508 ok, 1 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2339 / 2.7964 / 6.3813 | 74.3628 / 109.6674 / 131.3261 | 18.1452 / 64.154 |
| model_2 | 44 | 0 | 0.1838 / 0.3975 / 0.6019 | 41.3488 / 61.4979 / 196.7567 | 5.9113 / 67.0815 |
| model_3 | 44 | 0 | 0.1183 / 6.8408 / 7.7752 | 35.9591 / 47.5588 / 60.1757 | 7.1951 / 33.037 |
| model_4 | 524 | 1 | 0.302 / 5.3607 / 6.5835 | 96.978 / 156.8258 / 198.4248 | 35.0278 / 66.6185 |
| model_5 | 240 | 0 | 0.0492 / 0.2176 / 5.1959 | 11.0065 / 38.0457 / 40.0828 | 1.8313 / 16.9354 |
| model_6 | 38 | 0 | 0.134 / 0.2394 / 0.4155 | 30.4931 / 38.264 / 39.0809 | 3.869 / 16.2528 |
| model_7 | 22 | 0 | 0.0601 / 0.1889 / 0.1926 | 18.4991 / 18.8299 / 18.8707 | 1.5706 / 11.1609 |
| model_8 | 4 | 0 | 1.9203 / 3.6674 / 3.6688 | 12.1624 / 12.5421 / 12.5427 | 5.8108 / 11.2181 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 11 | 0.4559 | 3.379 | 3.3932 |
| wake | 18 | 0.5277 | 0.7417 | 0.7474 |

time spent swapping per GPU (s): GPU 1: 22.511, GPU 2: 0.92

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 5 | 4 |
| model_2 | 2 | 2 |
| model_3 | 2 | 1 |
| model_4 | 5 | 4 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 13, waited > 100 ms for a swap: 118; swap wait p50/p95/p99/max s: 0.0 / 1.2097 / 3.9652 / 7.4217
