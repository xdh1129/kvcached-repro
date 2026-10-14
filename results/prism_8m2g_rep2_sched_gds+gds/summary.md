# prism_8m2g_rep2_sched_gds+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.3 | 88.29 | 90.82 | 4.77 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.59 | 89.0 | 49.29 | 46.3 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 33.98 | 8.75 | 0 | 0 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.68 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 38 | 15.0 | 0.35 | 1.53 | 18.19 | 0.88 | 2 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.29 | 10.55 | 7 | 9 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 17.14 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.19 | 0.75 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2149 / 1.0853 / 3.7822 | 64.4943 / 121.4037 / 196.1573 | 16.1576 / 67.7836 |
| model_2 | 44 | 0 | 0.1729 / 2.7665 / 3.5075 | 41.3783 / 255.2782 / 373.9281 | 7.6425 / 86.9711 |
| model_3 | 44 | 0 | 0.1252 / 2.2391 / 8.3047 | 37.7744 / 40.6473 / 47.2238 | 4.6967 / 30.9324 |
| model_4 | 524 | 0 | 0.2865 / 0.8305 / 3.1992 | 81.6291 / 123.8549 / 135.3043 | 26.2718 / 47.4267 |
| model_5 | 240 | 0 | 0.0651 / 1.3785 / 4.4859 | 11.1166 / 38.2571 / 40.3246 | 2.1534 / 17.1059 |
| model_6 | 38 | 0 | 0.1322 / 0.9913 / 5.246 | 31.8676 / 39.2013 / 40.6589 | 4.9491 / 16.5285 |
| model_7 | 22 | 0 | 0.0639 / 5.2489 / 5.4947 | 18.6033 / 18.8204 / 18.8215 | 2.1127 / 12.1351 |
| model_8 | 4 | 0 | 4.4354 / 4.8378 / 4.8385 | 9.1026 / 12.1752 / 12.1802 | 6.489 / 8.7264 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 1.2441 | 1.8197 | 1.8962 |
| wake | 14 | 3.0052 | 4.7072 | 4.7743 |

time spent swapping per GPU (s): GPU 1: 38.932, GPU 2: 12.103

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

requests: 1508, triggered a wake: 0, waited > 100 ms for a swap: 63; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 3.0971 / 5.6323
