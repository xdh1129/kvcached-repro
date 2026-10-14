# prism_8m2g_rep2_kvp_ssd+ssd

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.31 | 88.28 | 91.84 | 3.76 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.6 | 89.0 | 49.13 | 46.46 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.71 | 34.42 | 8.67 | 12 | 18 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 44 | 15.0 | 0.35 | 1.53 | 18.19 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 42 | 22.85 | 0.4 | 1.71 | 36.77 | 11.09 | 16 | 34 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.98 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 39 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2245 / 3.1493 / 7.7721 | 66.9201 / 99.3425 / 118.8021 | 16.204 / 60.5117 |
| model_2 | 44 | 0 | 0.1737 / 1.4208 / 2.0448 | 41.2005 / 61.5463 / 187.8381 | 6.2428 / 64.0224 |
| model_3 | 44 | 0 | 0.1421 / 6.6203 / 7.361 | 39.8212 / 60.5838 / 72.1921 | 9.8015 / 45.7331 |
| model_4 | 524 | 0 | 0.2904 / 4.8335 / 6.6442 | 95.5155 / 134.0196 / 276.5842 | 32.4028 / 60.4197 |
| model_5 | 240 | 0 | 0.0631 / 0.8698 / 2.734 | 11.1999 / 37.9845 / 40.1726 | 2.1472 / 16.9242 |
| model_6 | 38 | 0 | 0.1324 / 0.5092 / 2.2397 | 31.3168 / 38.2771 / 39.0968 | 3.869 / 16.2474 |
| model_7 | 22 | 0 | 0.0631 / 1.3251 / 1.3967 | 18.4908 / 18.7028 / 18.757 | 2.0948 / 11.1647 |
| model_8 | 4 | 0 | 3.8355 / 4.2391 / 4.2398 | 9.9425 / 11.965 / 11.9657 | 6.4831 / 9.3198 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 9 | 0.2689 | 1.4036 | 1.4575 |
| wake | 16 | 1.8734 | 2.8861 | 3.416 |

time spent swapping per GPU (s): GPU 1: 30.012, GPU 2: 4.875

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 3 | 2 |
| model_2 | 2 | 2 |
| model_3 | 4 | 3 |
| model_4 | 3 | 2 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 11, waited > 100 ms for a swap: 97; swap wait p50/p95/p99/max s: 0.0 / 1.2374 / 5.7591 / 6.4565
