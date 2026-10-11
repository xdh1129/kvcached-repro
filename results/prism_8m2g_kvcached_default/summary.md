# prism_8m2g_kvcached_default

mode `kvcached` · vLLM 0.30.0 · kvcached v0.1.6 · NVIDIA RTX PRO 6000 Blackwell Server Edition · time scale 0.2 · replication 1

**Launch failed** — not every instance came up; see the instance table and logs.

## GPUs

| GPU | models | total GiB | idle used GiB | idle free GiB | peak used GiB | min free GiB |
|---|---|---|---|---|---|---|
| 1 | model_1, model_2, model_3, model_4 | 95.59 | – | – | 95.42 | 0.17 |
| 2 | model_5, model_6, model_7, model_8 | 95.59 | – | – | 0.63 | 94.96 |

## Instances

| name | model | GPU | started | startup s | weights GiB | CUDA graph GiB | idle mem GiB | peak mem GiB | KV used max GiB | max waiting | preemptions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_1 | Mistral-Nemo-Instruct-2407 | 1 | yes | 58 | 22.85 | 0.4 | – | – | – | – | – | – |
| model_2 | Mistral-Nemo-Instruct-2407 | 1 | yes | 38 | 22.85 | 0.4 | – | – | – | – | – | – |
| model_3 | Meta-Llama-3.1-8B | 1 | yes | 58 | 15.0 | 0.35 | – | – | – | – | – | – |
| model_4 | Mistral-Nemo-Instruct-2407 | 1 | NO | – | 22.85 | – | – | – | – | – | – | OutOfMemoryError; RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}; out of memory |
| model_5 | Mistral-7B-Instruct-v0.2 | 2 | NO | – | – | – | – | – | – | – | – | – |
| model_6 | Mistral-7B-Instruct-v0.2 | 2 | NO | – | – | – | – | – | – | – | – | – |
| model_7 | Llama-3.2-3B | 2 | NO | – | – | – | – | – | – | – | – | – |
| model_8 | Llama-3.2-3B | 2 | NO | – | – | – | – | – | – | – | – | – |
