# agent_rr_g67_dram+dram_off1

paths `dram+dram` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1 · offset 1 s

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 1066.7 | 67.8% | 5.6% | 8.0% | 18.4% | 0.1% |
| 7 | 1066.7 | 67.9% | 5.7% | 8.0% | 18.4% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.266 | 0.000 | 0.261 | 0.094 | 2.82 | 53.3 | 0.06 | 2.4 |
| 2-6 | 80 | 4996 | 4992 | 0.639 | 0.371 | 0.289 | 0.069 | 3.44 | 50.6 | 0.76 | 12.5 |
| 7-12 | 96 | 12278 | 12112 | 0.670 | 0.382 | 0.329 | 0.092 | 3.70 | 47.1 | 1.62 | 17.4 |
| 13-18 | 96 | 20222 | 20064 | 0.808 | 0.409 | 0.370 | 0.119 | 3.92 | 43.5 | 2.54 | 19.1 |
| 19-24 | 96 | 28166 | 28000 | 0.837 | 0.424 | 0.406 | 0.148 | 4.11 | 40.5 | 3.48 | 20.4 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 9.8% | 6.9% | 0.1% | 2.3% | 80.8% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
