# prism_8m2g_rep2_kvp_dram+recompute

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.83 | 87.76 | 91.55 | 4.05 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.92 | 88.67 | 49.26 | 46.34 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.78 | 35.05 | 8.75 | 100 | 127 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.78 | 26.73 | 1.25 | 0 | 8 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 38 | 15.0 | 0.35 | 1.59 | 18.92 | 0.94 | 0 | 28 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 39 | 22.85 | 0.4 | 1.78 | 37.23 | 10.94 | 200 | 341 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.59 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.59 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.49 | 8.57 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.49 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2128 / 3.0321 / 7.7493 | 68.558 / 114.4677 / 207.0414 | 17.5456 / 66.7843 |
| model_2 | 44 | 0 | 0.1734 / 0.4507 / 0.6012 | 41.6691 / 236.4841 / 293.0109 | 5.9473 / 80.5717 |
| model_3 | 44 | 0 | 0.1179 / 1.1115 / 1.2178 | 40.836 / 53.8248 / 57.5665 | 6.6274 / 45.5481 |
| model_4 | 524 | 0 | 0.282 / 25.4394 / 30.4993 | 93.288 / 268.3607 / 482.1579 | 40.2546 / 78.6469 |
| model_5 | 240 | 0 | 0.049 / 0.1991 / 0.3362 | 11.0546 / 38.011 / 40.2759 | 1.8399 / 16.9713 |
| model_6 | 38 | 0 | 0.1357 / 0.257 / 0.4222 | 30.4881 / 38.2273 / 39.0349 | 3.8653 / 16.2433 |
| model_7 | 22 | 0 | 0.0586 / 0.1731 / 0.1767 | 18.3763 / 18.7665 / 18.7903 | 1.5283 / 11.1784 |
| model_8 | 4 | 0 | 0.1196 / 0.2125 / 0.2136 | 10.8428 / 12.0547 / 12.0558 | 3.3171 / 6.3879 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 9 | 0.1748 | 1.9573 | 3.0575 |
| wake | 16 | 0.4288 | 0.5188 | 0.52 |

time spent swapping per GPU (s): GPU 1: 9.985, GPU 2: 0.925

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 2 | 1 |
| model_2 | 3 | 3 |
| model_3 | 4 | 3 |
| model_4 | 3 | 2 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 12, waited > 100 ms for a swap: 56; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 1.4836 / 5.7574
