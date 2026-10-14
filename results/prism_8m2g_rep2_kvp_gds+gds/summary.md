# prism_8m2g_rep2_kvp_gds+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.3 | 88.29 | 91.12 | 4.48 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.59 | 89.0 | 49.13 | 46.46 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 42 | 22.85 | 0.4 | 1.71 | 33.7 | 8.2 | 17 | 6 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 38 | 15.0 | 0.35 | 1.53 | 18.19 | 0.88 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.69 | 10.62 | 21 | 37 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.98 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 33 | 13.5 | 0.35 | 1.53 | 16.19 | 0.75 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2261 / 10.8731 / 25.4366 | 65.0132 / 111.7217 / 164.3705 | 16.4811 / 65.2256 |
| model_2 | 44 | 0 | 0.1618 / 2.9639 / 3.5265 | 41.2389 / 61.6884 / 189.6748 | 7.8416 / 64.6451 |
| model_3 | 44 | 0 | 0.2398 / 16.3366 / 19.041 | 35.6806 / 63.7004 / 80.6482 | 13.9286 / 32.737 |
| model_4 | 524 | 0 | 0.2855 / 16.7731 / 21.3056 | 88.1556 / 143.0587 / 255.4918 | 33.9459 / 61.1158 |
| model_5 | 240 | 0 | 0.0652 / 3.1986 / 7.3819 | 11.8114 / 38.259 / 40.1328 | 2.1823 / 17.1666 |
| model_6 | 38 | 0 | 0.1219 / 0.9245 / 4.8229 | 31.8716 / 38.393 / 39.0764 | 4.9687 / 16.5502 |
| model_7 | 22 | 0 | 0.0674 / 3.3319 / 3.4617 | 18.4371 / 18.7898 / 18.8191 | 2.1241 / 12.2538 |
| model_8 | 4 | 0 | 4.5748 / 4.9777 / 4.9784 | 11.998 / 12.0094 / 12.0106 | 8.5259 / 12.6707 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 6 | 1.0075 | 2.1558 | 2.233 |
| wake | 13 | 3.4447 | 5.1773 | 5.5614 |

time spent swapping per GPU (s): GPU 1: 40.592, GPU 2: 11.836

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 3 | 2 |
| model_2 | 2 | 2 |
| model_3 | 2 | 1 |
| model_4 | 2 | 1 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 9, waited > 100 ms for a swap: 148; swap wait p50/p95/p99/max s: 0.0 / 9.0329 / 12.4959 / 13.4499
