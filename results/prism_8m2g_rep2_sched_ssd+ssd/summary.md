# prism_8m2g_rep2_sched_ssd+ssd

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.31 | 88.28 | 91.13 | 4.47 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.6 | 89.0 | 49.23 | 46.36 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 33.9 | 8.75 | 0 | 0 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 36 | 15.0 | 0.35 | 1.53 | 18.25 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.44 | 10.7 | 17 | 17 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.95 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2177 / 0.6887 / 2.5809 | 66.6424 / 115.696 / 166.2813 | 16.3117 / 63.4368 |
| model_2 | 44 | 0 | 0.173 / 1.4063 / 2.0785 | 41.1557 / 246.9549 / 360.4969 | 6.2068 / 91.4576 |
| model_3 | 44 | 0 | 0.1328 / 1.5826 / 1.9641 | 38.1101 / 47.9663 / 48.821 | 4.6592 / 34.8799 |
| model_4 | 524 | 0 | 0.2801 / 0.8685 / 2.3839 | 86.9451 / 124.0495 / 136.6035 | 26.849 / 53.7858 |
| model_5 | 240 | 0 | 0.0752 / 0.4641 / 2.7521 | 11.2678 / 38.1699 / 40.388 | 2.1745 / 16.9987 |
| model_6 | 38 | 0 | 0.1389 / 0.5165 / 2.2216 | 31.4347 / 38.3642 / 39.1834 | 3.8771 / 16.3151 |
| model_7 | 22 | 0 | 0.0632 / 2.3665 / 2.4978 | 18.5972 / 19.4718 / 19.5441 | 2.1262 / 11.2399 |
| model_8 | 4 | 0 | 0.8835 / 1.287 / 1.2879 | 12.3411 / 12.7735 / 12.7751 | 4.8187 / 8.9261 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 0.3517 | 1.2553 | 1.2924 |
| wake | 14 | 1.8857 | 3.0522 | 3.3622 |

time spent swapping per GPU (s): GPU 1: 25.519, GPU 2: 5.272

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

requests: 1508, triggered a wake: 0, waited > 100 ms for a swap: 48; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 2.0628 / 3.6178
