# prism_8m2g_rep2_kvp_dram+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.57 | 88.03 | 90.96 | 4.63 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.71 | 88.88 | 49.38 | 46.21 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 50 | 22.85 | 0.4 | 1.71 | 34.88 | 8.91 | 26 | 68 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 50 | 22.85 | 0.4 | 1.71 | 26.74 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 47 | 15.0 | 0.35 | 1.53 | 18.21 | 0.88 | 9 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 48 | 22.85 | 0.4 | 1.71 | 36.67 | 10.78 | 21 | 30 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 38 | 13.5 | 0.35 | 1.53 | 16.95 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 34 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 60 | 6.04 | 0.3 | 1.44 | 8.65 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 60 | 6.04 | 0.3 | 1.44 | 8.19 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2319 / 6.3073 / 10.2133 | 68.1216 / 107.4508 / 139.2451 | 17.4555 / 62.9785 |
| model_2 | 44 | 0 | 0.1783 / 0.4058 / 0.62 | 41.5389 / 61.5272 / 245.4474 | 5.9163 / 83.5902 |
| model_3 | 44 | 0 | 0.1238 / 5.928 / 7.7579 | 30.588 / 40.8162 / 47.4459 | 7.6151 / 24.749 |
| model_4 | 524 | 0 | 0.2921 / 4.7931 / 9.9165 | 91.7077 / 138.4778 / 193.5063 | 33.7617 / 63.7393 |
| model_5 | 240 | 0 | 0.0527 / 0.3799 / 8.9558 | 11.1399 / 38.0762 / 40.1845 | 1.9499 / 16.9411 |
| model_6 | 38 | 0 | 0.1371 / 0.2449 / 0.4299 | 30.4301 / 38.2808 / 39.0836 | 3.8499 / 16.2598 |
| model_7 | 22 | 0 | 0.0623 / 0.1796 / 0.1833 | 18.1474 / 18.7208 / 18.7556 | 1.5423 / 11.1634 |
| model_8 | 4 | 0 | 0.9992 / 1.791 / 1.7923 | 12.0638 / 12.1061 / 12.1062 | 4.9856 / 9.552 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 6 | 0.6462 | 7.5936 | 8.3352 |
| wake | 12 | 0.5142 | 4.6628 | 5.1262 |

time spent swapping per GPU (s): GPU 1: 27.681, GPU 2: 0.922

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 2 | 1 |
| model_2 | 2 | 2 |
| model_3 | 2 | 2 |
| model_4 | 2 | 1 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 10, waited > 100 ms for a swap: 42; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.7315 / 10.4242
