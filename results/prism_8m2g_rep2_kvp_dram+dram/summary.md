# prism_8m2g_rep2_kvp_dram+dram

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 2

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 7.78 | 87.81 | 91.87 | 3.72 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 6.94 | 88.65 | 49.58 | 46.02 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 33.83 | 8.67 | 3 | 1 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 26.73 | 1.25 | 0 | 0 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 38 | 15.0 | 0.35 | 1.53 | 18.0 | 0.88 | 0 | 0 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 1.71 | 36.84 | 11.09 | 24 | 22 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.94 | 1.44 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 30 | 13.5 | 0.35 | 1.53 | 16.13 | 0.69 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 1.44 | 8.64 | 0.44 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 36 | 6.04 | 0.3 | 1.44 | 8.2 | 0.16 | 0 | 0 | – |

## Requests — 1508/1508 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 592 | 0 | 0.2109 / 0.5396 / 1.0804 | 69.0064 / 110.8174 / 139.1503 | 16.5097 / 60.5772 |
| model_2 | 44 | 0 | 0.1743 / 0.4418 / 0.5805 | 41.3724 / 218.5988 / 289.5319 | 5.9335 / 77.2623 |
| model_3 | 44 | 0 | 0.1171 / 0.5622 / 0.5818 | 38.8362 / 45.7159 / 47.5513 | 5.6866 / 35.583 |
| model_4 | 524 | 0 | 0.2829 / 1.3235 / 5.3958 | 88.9903 / 133.1297 / 185.7491 | 30.4352 / 55.7684 |
| model_5 | 240 | 0 | 0.0506 / 0.208 / 0.3476 | 11.0421 / 38.0494 / 40.2395 | 1.8422 / 16.9507 |
| model_6 | 38 | 0 | 0.1256 / 0.2572 / 0.4378 | 30.5802 / 38.5533 / 39.1567 | 3.8828 / 16.2931 |
| model_7 | 22 | 0 | 0.064 / 0.193 / 0.1975 | 18.6073 / 19.4977 / 19.5845 | 1.556 / 11.2109 |
| model_8 | 4 | 0 | 0.4274 / 0.6408 / 0.6408 | 11.3168 / 12.7177 / 12.7202 | 3.7217 / 6.5786 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 9 | 0.2056 | 0.4822 | 0.5157 |
| wake | 16 | 0.438 | 0.6896 | 0.7309 |

time spent swapping per GPU (s): GPU 1: 8.386, GPU 2: 0.935

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 3 | 2 |
| model_2 | 3 | 3 |
| model_3 | 4 | 3 |
| model_4 | 2 | 1 |
| model_5 | 1 | 0 |
| model_6 | 1 | 0 |
| model_7 | 1 | 0 |
| model_8 | 1 | 0 |

requests: 1508, triggered a wake: 12, waited > 100 ms for a swap: 56; swap wait p50/p95/p99/max s: 0.0 / 0.0001 / 1.778 / 4.6425
