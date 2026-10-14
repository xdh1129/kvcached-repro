# prism_8m2g_rep2_kvp_gds+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.3 | 88.29 | 90.82 | 4.77 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.59 | 89.0 | 49.72 | 45.88 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 33.91 | 8.67 | 1 | 1 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 27.44 | 1.56 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 41 | 15.0 | 0.35 | 1.53 | 18.25 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 42 | 22.85 | 0.4 | 1.71 | 36.27 | 10.31 | 22 | 17 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 17.37 | 1.56 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.73 | 0.94 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 36 | 6.04 | 0.3 | 1.44 | 8.52 | 0.38 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 36 | 6.04 | 0.3 | 1.44 | 8.18 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2391 / 9.2363 / 11.5838 | 64.1983 / 132.7431 / 192.1849 | 19.3274 / 75.7901 |
| model_2 | 44 | 0 | 0.2501 / 11.6835 / 12.0841 | 43.5211 / 285.2102 / 422.0603 | 12.5853 / 104.1842 |
| model_3 | 44 | 0 | 0.1313 / 7.9507 / 9.1419 | 38.1647 / 49.5835 / 52.993 | 4.6913 / 32.9769 |
| model_4 | 524 | 0 | 0.2782 / 1.9285 / 8.3908 | 79.0862 / 124.1835 / 134.1652 | 25.7986 / 46.401 |
| model_5 | 240 | 0 | 0.0737 / 12.1723 / 14.1848 | 11.2217 / 43.5364 / 52.7028 | 3.3723 / 24.9618 |
| model_6 | 38 | 0 | 5.692 / 10.8138 / 16.0161 | 35.7726 / 66.8613 / 155.4632 | 6.8876 / 22.5582 |
| model_7 | 22 | 0 | 4.4642 / 11.0394 / 11.102 | 16.0748 / 19.5272 / 19.5791 | 5.6944 / 18.1379 |
| model_8 | 4 | 0 | 5.7708 / 7.2248 / 7.2262 | 8.747 / 11.9282 / 11.9294 | 9.6445 / 14.8669 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 5 | 0.4996 | 2.1312 | 2.2086 |
| wake | 12 | 4.8933 | 10.3635 | 13.211 |

time spent swapping per GPU (s): GPU 1: 50.117, GPU 2: 22.128

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 3 | 2 |
| model_2 | 2 | 2 |
| model_3 | 2 | 1 |
| model_4 | 1 | 0 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 9, waited > 100 ms for a swap: 148; swap wait p50/p95/p99/max s: 0.0 / 8.5365 / 12.136 / 15.5977
