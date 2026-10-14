# prism_8m2g_rep2_kvp_dram+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.44 | 4.15 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.44 | 46.15 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 33.9 | 8.67 | 2 | 5 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 39 | 15.0 | 0.35 | 1.53 | 18.19 | 0.88 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.59 | 10.86 | 16 | 19 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 31 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 31 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2163 / 0.5004 / 0.7159 | 69.5465 / 118.6567 / 142.9927 | 16.6531 / 62.8072 |
| model_2 | 44 | 0 | 0.185 / 0.4017 / 0.6038 | 41.5133 / 241.0644 / 351.3771 | 5.9204 / 90.193 |
| model_3 | 44 | 0 | 0.1226 / 0.4863 / 1.7223 | 38.5783 / 63.1956 / 67.5107 | 5.8657 / 37.9371 |
| model_4 | 524 | 0 | 0.2789 / 0.7819 / 1.0201 | 85.2068 / 124.4694 / 137.8279 | 26.8761 / 49.4597 |
| model_5 | 240 | 0 | 0.0564 / 0.2101 / 1.2985 | 11.0657 / 37.9809 / 40.1871 | 1.8403 / 16.9184 |
| model_6 | 38 | 0 | 0.1262 / 0.2419 / 0.4212 | 30.3857 / 38.2731 / 39.1048 | 3.8704 / 16.2584 |
| model_7 | 22 | 0 | 0.0556 / 0.1808 / 0.1855 | 18.4748 / 18.7041 / 18.9441 | 1.5509 / 11.1786 |
| model_8 | 4 | 0 | 0.1131 / 0.1908 / 0.1908 | 11.6928 / 12.1324 / 12.1345 | 3.8337 / 7.4133 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 0.3087 | 1.6334 | 1.6338 |
| wake | 14 | 0.4602 | 1.7103 | 1.7151 |

time spent swapping per GPU (s): GPU 1: 13.253, GPU 2: 0.951

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 4 | 3 |
| model_2 | 2 | 2 |
| model_3 | 3 | 2 |
| model_4 | 1 | 0 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 9, waited > 100 ms for a swap: 28; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.3106 / 2.1524
