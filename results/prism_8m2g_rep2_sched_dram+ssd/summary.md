# prism_8m2g_rep2_sched_dram+ssd

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.53 | 88.06 | 91.49 | 4.1 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.69 | 88.9 | 49.32 | 46.27 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 33.9 | 8.67 | 1 | 4 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 36 | 15.0 | 0.35 | 1.53 | 18.25 | 0.94 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.57 | 10.86 | 19 | 17 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.12 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2164 / 0.4899 / 0.8757 | 69.1344 / 108.4859 / 132.4624 | 16.3424 / 61.8089 |
| model_2 | 44 | 0 | 0.1759 / 0.4369 / 0.6062 | 41.7432 / 239.9139 / 349.566 | 5.9408 / 89.8402 |
| model_3 | 44 | 0 | 0.1197 / 0.2964 / 0.4337 | 38.5508 / 55.8828 / 67.1761 | 5.957 / 38.3506 |
| model_4 | 524 | 0 | 0.2824 / 0.7226 / 0.8726 | 89.0385 / 124.9756 / 136.6639 | 27.1296 / 54.7938 |
| model_5 | 240 | 0 | 0.0487 / 0.1999 / 0.3331 | 11.084 / 37.9382 / 40.3904 | 1.841 / 16.942 |
| model_6 | 38 | 0 | 0.1243 / 0.249 / 0.4301 | 30.4628 / 38.2379 / 39.0887 | 3.8667 / 16.2434 |
| model_7 | 22 | 0 | 0.0589 / 0.1797 / 0.184 | 18.3732 / 18.6297 / 18.6331 | 1.5507 / 11.1809 |
| model_8 | 4 | 0 | 0.1154 / 0.203 / 0.2044 | 10.9472 / 12.2946 / 12.2968 | 3.3075 / 6.3612 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 7 | 0.3031 | 1.3155 | 1.416 |
| wake | 14 | 0.4427 | 1.1926 | 1.3093 |

time spent swapping per GPU (s): GPU 1: 10.995, GPU 2: 0.923

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

requests: 1508, triggered a wake: 0, waited > 100 ms for a swap: 28; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 0.303 / 1.7329
