# prism_8m2g_rep2_kvp_dram+ssd

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.13 | 4.46 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.33 | 46.27 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 34.0 | 8.67 | 1 | 5 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 41 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 39 | 15.0 | 0.35 | 1.53 | 18.19 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 36.22 | 10.86 | 24 | 21 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 34 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 40 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2161 / 0.4986 / 0.8478 | 68.4524 / 109.2884 / 122.3052 | 15.6047 / 61.5652 |
| model_2 | 44 | 0 | 0.1893 / 0.403 / 0.6036 | 41.5237 / 249.9491 / 365.5586 | 5.9244 / 93.2002 |
| model_3 | 44 | 0 | 0.1222 / 0.4405 / 0.5094 | 39.7949 / 54.973 / 56.358 | 5.9031 / 39.5804 |
| model_4 | 524 | 0 | 0.2778 / 0.7646 / 1.8385 | 88.2414 / 122.3521 / 137.3532 | 28.0592 / 51.9683 |
| model_5 | 240 | 0 | 0.0446 / 0.2062 / 0.3369 | 11.055 / 38.0177 / 40.1851 | 1.8375 / 16.9284 |
| model_6 | 38 | 0 | 0.1305 / 0.2598 / 0.4219 | 30.4428 / 38.2463 / 39.0604 | 3.863 / 16.2455 |
| model_7 | 22 | 0 | 0.0581 / 0.1905 / 0.1953 | 18.4528 / 18.5988 / 18.6015 | 1.5588 / 11.158 |
| model_8 | 4 | 0 | 0.1122 / 0.1995 / 0.2006 | 10.8375 / 12.0837 / 12.0856 | 3.2992 / 6.3549 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 9 | 0.3074 | 1.2413 | 1.3318 |
| wake | 16 | 0.4406 | 1.1922 | 1.2721 |

time spent swapping per GPU (s): GPU 1: 13.698, GPU 2: 0.931

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 4 | 3 |
| model_2 | 2 | 2 |
| model_3 | 4 | 3 |
| model_4 | 2 | 1 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 11, waited > 100 ms for a swap: 32; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.3486 / 2.4947
