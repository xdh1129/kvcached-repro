# prism_8m2g_rep2_kvp_dram+ssd

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.34 | 4.26 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.39 | 46.2 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 42 | 22.85 | 0.4 | 1.71 | 33.9 | 8.67 | 3 | 1 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 37 | 15.0 | 0.35 | 1.53 | 18.16 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 39 | 22.85 | 0.4 | 1.71 | 36.42 | 10.86 | 24 | 20 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 34 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 36 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2123 / 0.4922 / 0.933 | 68.1561 / 108.8197 / 132.2283 | 16.1613 / 61.6881 |
| model_2 | 44 | 0 | 0.189 / 0.4233 / 0.6049 | 41.8004 / 239.7284 / 349.334 | 5.949 / 89.7753 |
| model_3 | 44 | 0 | 0.112 / 0.4409 / 1.2813 | 38.5116 / 64.0665 / 66.9105 | 5.9436 / 38.3503 |
| model_4 | 524 | 0 | 0.2845 / 0.7239 / 0.9379 | 88.7004 / 125.4081 / 136.1029 | 27.1432 / 54.7429 |
| model_5 | 240 | 0 | 0.0495 / 0.2001 / 0.855 | 11.064 / 38.208 / 40.1653 | 1.8421 / 16.9473 |
| model_6 | 38 | 0 | 0.1367 / 0.2431 / 0.4191 | 30.5027 / 38.2693 / 39.1435 | 3.8744 / 16.2596 |
| model_7 | 22 | 0 | 0.0624 / 0.1801 / 0.1837 | 18.4859 / 18.6106 / 18.6114 | 1.5494 / 11.1844 |
| model_8 | 4 | 0 | 0.1119 / 0.1993 / 0.2007 | 11.221 / 12.0235 / 12.0263 | 3.5654 / 6.8858 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 0.3103 | 1.2326 | 1.297 |
| wake | 14 | 0.4422 | 1.1926 | 1.3147 |

time spent swapping per GPU (s): GPU 1: 10.867, GPU 2: 0.931

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

requests: 1508, triggered a wake: 9, waited > 100 ms for a swap: 26; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.286 / 1.7114
