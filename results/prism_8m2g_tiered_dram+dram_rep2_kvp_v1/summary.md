# prism_8m2g_tiered_dram+dram_rep2_kvp

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 92.91 | 2.68 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.32 | 46.27 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 34.05 | 8.59 | 24 | 96 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 41 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 40 | 15.0 | 0.35 | 1.53 | 18.19 | 0.88 | 5 | 1 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 44 | 22.85 | 0.4 | 1.71 | 38.77 | 12.27 | 72 | 182 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.12 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 36 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2568 / 1.4063 / 2.1758 | 78.1511 / 112.754 / 133.3714 | 17.3684 / 64.434 |
| model_2 | 44 | 0 | 0.1711 / 0.4109 / 0.6011 | 41.654 / 59.2571 / 61.6043 | 5.937 / 27.7971 |
| model_3 | 44 | 0 | 0.1277 / 0.9113 / 1.6958 | 37.9657 / 42.0841 / 47.4884 | 5.7103 / 28.8287 |
| model_4 | 524 | 0 | 0.3446 / 3.3119 / 7.0688 | 102.9138 / 141.1878 / 158.3007 | 35.2069 / 68.2211 |
| model_5 | 240 | 0 | 0.0578 / 0.2085 / 0.3426 | 11.0698 / 38.0683 / 40.1909 | 1.9477 / 16.9398 |
| model_6 | 38 | 0 | 0.1262 / 0.251 / 0.4377 | 30.5089 / 38.3255 / 39.0841 | 3.8556 / 16.2616 |
| model_7 | 22 | 0 | 0.0584 / 0.1831 / 0.1864 | 18.5152 / 18.798 / 18.9726 | 1.5478 / 11.1696 |
| model_8 | 4 | 0 | 0.113 / 0.2041 / 0.2049 | 10.8538 / 12.1323 / 12.1346 | 3.2958 / 6.353 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 10 | 1.0417 | 2.9898 | 3.0335 |
| wake | 16 | 0.5172 | 0.6968 | 0.7552 |

time spent swapping per GPU (s): GPU 1: 19.025, GPU 2: 0.928

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 5 | 4 |
| model_2 | 2 | 2 |
| model_3 | 2 | 2 |
| model_4 | 3 | 2 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 12, waited > 100 ms for a swap: 52; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.9576 / 2.0971
