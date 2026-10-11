# prism_8m2g_noswap_default

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 1

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | 92.88 | 2.71 | 94.71 | 0.89 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | 48.08 | 47.51 | 49.48 | 46.11 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 25.18 | 25.63 | 0.47 | 194 | 398 | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 42 | 22.85 | 0.4 | 25.18 | 25.48 | 0.47 | 4 | 4 | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 39 | 15.0 | 0.35 | 17.14 | 17.25 | 0.38 | 2 | 1 | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | 24.71 | 24.71 | 0.0 | 262 | 0 | – |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | yes | 42 | 13.5 | 0.35 | 15.34 | 16.08 | 0.75 | 0 | 0 | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | yes | 32 | 13.5 | 0.35 | 15.24 | 15.68 | 0.38 | 0 | 0 | – |
| model_7 | Llama-3.2-3B | 2 | yes | 44 | 6.04 | 0.3 | 8.77 | 8.94 | 0.22 | 0 | 0 | – |
| model_8 | Llama-3.2-3B | 2 | yes | 34 | 6.04 | 0.3 | 8.07 | 8.13 | 0.11 | 0 | 0 | – |

## Requests — 492/754 ok, 262 failed

| model | n | failed | TTFT p50/p95/p99 s | TPOT p50/p95/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| model_1 | 296 | 0 | 93.8165 / 149.728 / 156.7557 | 26.291 / 99.3784 / 212.898 | 99.4726 / 164.6206 |
| model_2 | 22 | 0 | 0.1497 / 3.3408 / 3.7874 | 39.4286 / 49.4356 / 51.9109 | 8.117 / 22.0893 |
| model_3 | 22 | 0 | 0.0741 / 0.2218 / 0.489 | 25.3559 / 26.2207 / 26.5283 | 3.6427 / 20.1408 |
| model_4 | 262 | 262 | None / None / None | None / None / None | None / None |
| model_5 | 120 | 0 | 0.0372 / 0.1247 / 0.19 | 10.8732 / 36.4258 / 37.007 | 1.7614 / 16.1378 |
| model_6 | 19 | 0 | 0.092 / 0.1407 / 0.1519 | 29.0741 / 36.7839 / 36.9026 | 3.6762 / 15.101 |
| model_7 | 11 | 0 | 0.0556 / 0.171 / 0.2305 | 18.1485 / 19.6575 / 20.4217 | 1.5248 / 10.4875 |
| model_8 | 2 | 0 | 0.0395 / 0.0441 / 0.0445 | 10.6981 / 11.8599 / 11.9632 | 3.1624 / 6.0163 |
- 262× `not completed: no KV capacity; replay stopped 2874s after trace start`
