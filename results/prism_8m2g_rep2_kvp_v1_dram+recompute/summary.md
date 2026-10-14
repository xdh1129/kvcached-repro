# prism_8m2g_rep2_kvp_dram+recompute

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.83 | 87.76 | 90.78 | 4.81 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.92 | 88.67 | 49.32 | 46.27 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.78 | 35.05 | 8.75 | 100 | 234 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 41 | 22.85 | 0.4 | 1.78 | 26.73 | 1.25 | 0 | 8 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 40 | 15.0 | 0.35 | 1.59 | 18.82 | 0.94 | 0 | 10 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.78 | 37.08 | 10.78 | 179 | 337 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 33 | 13.5 | 0.35 | 1.59 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.59 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.49 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 37 | 6.04 | 0.3 | 1.49 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2151 / 0.7673 / 8.3387 | 69.751 / 130.3437 / 168.5714 | 17.6896 / 75.6929 |
| model_2 | 44 | 0 | 0.1904 / 0.405 / 0.6125 | 41.4605 / 235.6131 / 290.7817 | 5.9214 / 80.242 |
| model_3 | 44 | 0 | 0.1179 / 1.0683 / 1.8086 | 38.2439 / 47.0623 / 47.9239 | 6.4222 / 32.3302 |
| model_4 | 524 | 0 | 0.2833 / 1.8368 / 29.7262 | 94.5223 / 212.9751 / 440.2983 | 40.1581 / 75.3986 |
| model_5 | 240 | 0 | 0.0452 / 0.2008 / 0.3629 | 11.0646 / 37.9783 / 40.3364 | 1.8433 / 16.9448 |
| model_6 | 38 | 0 | 0.1129 / 0.238 / 0.4138 | 30.4777 / 38.3526 / 39.0734 | 3.8728 / 16.2506 |
| model_7 | 22 | 0 | 0.0588 / 0.181 / 0.1843 | 18.4796 / 18.7232 / 18.7526 | 1.5598 / 11.1657 |
| model_8 | 4 | 0 | 0.1177 / 0.2024 / 0.2032 | 10.8309 / 12.0568 / 12.0587 | 3.3087 / 6.3672 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 9 | 0.1933 | 2.0466 | 3.2546 |
| wake | 15 | 0.5076 | 0.5202 | 0.5261 |

time spent swapping per GPU (s): GPU 1: 9.953, GPU 2: 0.925

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 3 | 2 |
| model_2 | 3 | 3 |
| model_3 | 2 | 2 |
| model_4 | 3 | 2 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 11, waited > 100 ms for a swap: 54; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 2.3133 / 5.1798
