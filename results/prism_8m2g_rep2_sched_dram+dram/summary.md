# prism_8m2g_rep2_sched_dram+dram

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.78 | 87.81 | 91.75 | 3.84 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.94 | 88.65 | 49.58 | 46.02 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 33.9 | 8.67 | 3 | 1 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 36 | 15.0 | 0.35 | 1.53 | 18.25 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 39 | 22.85 | 0.4 | 1.71 | 36.58 | 10.86 | 15 | 16 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 34 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2149 / 0.4653 / 0.9545 | 69.4019 / 106.5975 / 129.5124 | 16.0048 / 61.5997 |
| model_2 | 44 | 0 | 0.1702 / 0.4444 / 0.6084 | 41.361 / 239.6062 / 349.1655 | 5.9407 / 89.7578 |
| model_3 | 44 | 0 | 0.1112 / 0.2706 / 0.4352 | 38.7331 / 65.0858 / 68.1231 | 7.0218 / 38.6805 |
| model_4 | 524 | 0 | 0.276 / 0.7146 / 0.8715 | 88.5767 / 124.0175 / 137.0576 | 27.2565 / 54.4886 |
| model_5 | 240 | 0 | 0.0518 / 0.2049 / 0.3469 | 11.0712 / 38.0287 / 40.2171 | 1.8386 / 16.9204 |
| model_6 | 38 | 0 | 0.1242 / 0.2411 / 0.42 | 30.467 / 38.3779 / 39.0843 | 3.8711 / 16.2572 |
| model_7 | 22 | 0 | 0.0587 / 0.1978 / 0.2017 | 18.4329 / 18.6194 / 18.6408 | 1.578 / 11.1649 |
| model_8 | 4 | 0 | 0.1188 / 0.2058 / 0.2069 | 10.8168 / 12.0509 / 12.0532 | 3.3034 / 6.3554 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 0.2255 | 0.4695 | 0.4758 |
| wake | 14 | 0.4422 | 0.7063 | 0.7352 |

time spent swapping per GPU (s): GPU 1: 7.157, GPU 2: 0.94

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

requests: 1508, triggered a wake: 0, waited > 100 ms for a swap: 28; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.3033 / 1.3472
