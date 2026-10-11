# test_prism_pipeline

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 1

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1 | 95.59 | 18.23 | 77.36 | 19.89 | 75.7 |
| 2 | model_5 | 95.59 | 18.23 | 77.36 | 19.89 | 75.7 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Qwen3-8B | 1 | yes | 36 | 15.27 | 0.37 | 17.59 | 19.17 | 1.48 | 0 | 0 | – |
| model_5 | Qwen3-8B | 2 | yes | 36 | 15.27 | 0.37 | 17.59 | 18.22 | 0.56 | 0 | 0 | – |

## Requests — 416/416 ok, 0 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 296 | 0 | 0.0376 / 0.0554 / 0.0658 | 12.0 / 12.9579 / 13.2911 | 3.2055 / 9.9283 |
| model_5 | 120 | 0 | 0.0351 / 0.0596 / 0.0721 | 11.8345 / 12.3548 / 13.0034 | 1.3234 / 6.1467 |
