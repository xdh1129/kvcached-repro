# prism_8m2g_rep2_sched_dram+recompute

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.83 | 87.76 | 91.5 | 4.09 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.92 | 88.67 | 49.26 | 46.34 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.78 | 34.97 | 8.67 | 114 | 272 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.78 | 26.73 | 1.25 | 0 | 6 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 39 | 15.0 | 0.35 | 1.59 | 18.39 | 0.94 | 0 | 22 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.78 | 36.51 | 10.86 | 49 | 92 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.59 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.59 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 35 | 6.04 | 0.3 | 1.49 | 8.57 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.49 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2186 / 0.5779 / 5.3348 | 69.3878 / 139.8266 / 248.2597 | 18.4393 / 71.7836 |
| model_2 | 44 | 0 | 0.1744 / 0.4386 / 0.6166 | 41.8317 / 242.4953 / 353.547 | 5.9457 / 90.6824 |
| model_3 | 44 | 0 | 0.1146 / 0.2775 / 0.4262 | 38.8854 / 66.865 / 70.8153 | 7.0278 / 39.0889 |
| model_4 | 524 | 0 | 0.2799 / 0.7093 / 0.9254 | 92.0805 / 125.8842 / 138.2671 | 27.5964 / 64.3948 |
| model_5 | 240 | 0 | 0.0482 / 0.1977 / 0.3445 | 11.0987 / 37.9435 / 40.0755 | 1.8399 / 16.9243 |
| model_6 | 38 | 0 | 0.1285 / 0.2465 / 0.4309 | 30.4842 / 38.2543 / 39.1267 | 3.8712 / 16.2515 |
| model_7 | 22 | 0 | 0.064 / 0.1796 / 0.1834 | 18.4362 / 19.3973 / 19.5278 | 1.5366 / 11.1992 |
| model_8 | 4 | 0 | 0.121 / 0.2144 / 0.2158 | 10.8005 / 12.0213 / 12.0234 | 3.3042 / 6.3615 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 0.1858 | 0.2858 | 0.3059 |
| wake | 14 | 0.4298 | 0.5204 | 0.5216 |

time spent swapping per GPU (s): GPU 1: 6.017, GPU 2: 0.932

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

requests: 1508, triggered a wake: 0, waited > 100 ms for a swap: 28; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.305 / 1.2116
