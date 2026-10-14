# prism_8m2g_rep2_kvp_gds+gds

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.38 | 88.22 | 93.43 | 2.17 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.67 | 88.93 | 50.38 | 45.21 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.72 | 35.02 | 8.44 | 67 | 0 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 43 | 22.85 | 0.4 | 1.72 | 28.0 | 1.8 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 38 | 15.0 | 0.35 | 1.55 | 18.66 | 1.12 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 40 | 22.85 | 0.4 | 1.72 | 35.14 | 8.83 | 83 | 232 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 33 | 13.5 | 0.35 | 1.55 | 18.04 | 2.12 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 31 | 13.5 | 0.35 | 1.55 | 16.74 | 1.0 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 37 | 6.04 | 0.3 | 1.46 | 8.53 | 0.33 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 38 | 6.04 | 0.3 | 1.46 | 8.22 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 1.5725 / 45.9818 / 49.7396 | 62.6337 / 156.8814 / 278.3706 | 28.7889 / 91.0674 |
| model_2 | 44 | 0 | 1.2184 / 14.6947 / 15.5269 | 43.4267 / 162.7068 / 329.0359 | 15.4282 / 48.7152 |
| model_3 | 44 | 0 | 13.8241 / 31.2276 / 31.8473 | 44.9018 / 56.5321 / 80.8358 | 26.2537 / 46.6943 |
| model_4 | 524 | 0 | 10.5408 / 23.0578 / 27.3603 | 72.799 / 125.952 / 208.029 | 33.786 / 60.9425 |
| model_5 | 240 | 0 | 0.187 / 19.5946 / 25.4989 | 14.5619 / 56.8245 / 107.5321 | 7.1942 / 31.0525 |
| model_6 | 38 | 0 | 9.269 / 15.8711 / 20.829 | 38.1185 / 60.4378 / 63.83 | 11.631 / 27.4742 |
| model_7 | 22 | 0 | 8.8258 / 16.0621 / 16.1383 | 18.6712 / 20.5811 / 21.1411 | 8.9226 / 23.3069 |
| model_8 | 4 | 0 | 6.4842 / 6.8899 / 6.8913 | 11.8568 / 12.0845 / 12.0854 | 10.3222 / 14.3421 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 3 | 1.0197 | 4.3961 | 4.7713 |
| wake | 10 | 11.6617 | 19.6285 | 19.6781 |

time spent swapping per GPU (s): GPU 1: 97.7, GPU 2: 25.705

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 2 | 1 |
| model_2 | 2 | 2 |
| model_3 | 1 | 0 |
| model_4 | 1 | 0 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 9, waited > 100 ms for a swap: 378; swap wait p50/p95/p99/max s: 0.0 / 26.6817 / 29.3817 / 31.463
