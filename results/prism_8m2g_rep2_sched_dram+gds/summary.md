# prism_8m2g_rep2_sched_dram+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.48 | 4.11 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.41 | 46.18 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 33.9 | 8.67 | 3 | 5 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 38 | 15.0 | 0.35 | 1.53 | 18.25 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 36.57 | 10.86 | 24 | 20 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 17.03 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2223 / 0.4873 / 0.8733 | 68.7 / 111.3398 / 140.3849 | 16.5245 / 61.7462 |
| model_2 | 44 | 0 | 0.1837 / 0.3959 / 0.6094 | 41.4579 / 242.1012 / 352.9705 | 5.9172 / 90.5808 |
| model_3 | 44 | 0 | 0.1278 / 0.5789 / 0.6776 | 38.5212 / 53.3986 / 66.022 | 5.7928 / 36.7027 |
| model_4 | 524 | 0 | 0.2801 / 0.7768 / 0.8914 | 88.7926 / 124.9547 / 137.8365 | 27.127 / 54.8041 |
| model_5 | 240 | 0 | 0.0507 / 0.1997 / 0.3757 | 11.0878 / 37.9569 / 40.1508 | 1.8406 / 16.9186 |
| model_6 | 38 | 0 | 0.1336 / 0.2483 / 0.4168 | 30.4065 / 39.3561 / 41.1656 | 3.8514 / 16.2595 |
| model_7 | 22 | 0 | 0.064 / 0.1836 / 0.1878 | 18.3876 / 18.5928 / 18.5943 | 1.5539 / 11.1652 |
| model_8 | 4 | 0 | 0.117 / 0.2096 / 0.2109 | 10.827 / 12.0779 / 12.08 | 3.2996 / 6.3534 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 0.3347 | 1.7151 | 1.7749 |
| wake | 14 | 0.4817 | 1.7388 | 1.7614 |

time spent swapping per GPU (s): GPU 1: 13.835, GPU 2: 0.95

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

requests: 1508, triggered a wake: 0, waited > 100 ms for a swap: 28; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.3142 / 2.3437
