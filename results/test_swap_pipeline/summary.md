# test_swap_pipeline

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 1

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_4 | 95.59 | 4.1 | 91.49 | 21.64 | 73.96 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Qwen3-8B | 1 | yes | 34 | 15.27 | 0.37 | 1.68 | 18.91 | 1.27 | 0 | 0 | – |
| model_4 | Qwen3-8B | 1 | yes | 34 | 15.27 | 0.37 | 1.68 | 18.55 | 0.84 | 0 | 0 | – |

## Requests — 558/558 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 296 | 0 | 264.4797 / 980.626 / 1054.5326 | 11.8765 / 13.2889 / 13.4411 | 268.7254 / 1055.4299 |
| model_4 | 262 | 0 | 404.0761 / 986.7179 / 1038.8556 | 11.7117 / 13.3667 / 14.0041 | 408.8653 / 1044.0398 |

## Weight swaps (DRAM)

| action | count | p50 s | p95 s | max s |
|---|---|---|---|---|
| sleep | 213 | 0.3313 | 0.3334 | 0.3366 |
| wake | 214 | 0.3461 | 0.3483 | 0.3503 |

time spent swapping per GPU (s): GPU 1: 144.661

| instance | wakes | sleeps |
|---|---|---|
| model_1 | 107 | 107 |
| model_4 | 107 | 106 |

requests: 558, triggered a wake: 214, waited > 100 ms for a swap: 458; swap wait p50/p95/p99/max s: 326.0037 / 981.7473 / 1048.656 / 1059.753
