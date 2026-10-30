# agent_rr_g67_dram+dram_off0.3

paths `dram+dram` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1 · offset 0.3 s

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 945.6 | 76.4% | 6.7% | 10.0% | 6.8% | 0.1% |
| 7 | 945.6 | 76.6% | 6.7% | 10.1% | 6.6% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.263 | 0.000 | 0.259 | 0.094 | 2.82 | 53.2 | 0.06 | 1.6 |
| 2-6 | 80 | 4996 | 4992 | 0.647 | 0.400 | 0.287 | 0.066 | 3.44 | 49.8 | 0.76 | 12.5 |
| 7-12 | 96 | 12278 | 12112 | 0.685 | 0.413 | 0.330 | 0.090 | 3.70 | 45.6 | 1.62 | 17.2 |
| 13-18 | 96 | 20222 | 20064 | 0.876 | 0.465 | 0.388 | 0.112 | 3.92 | 41.5 | 2.54 | 18.6 |
| 19-24 | 96 | 28166 | 28000 | 0.937 | 0.492 | 0.435 | 0.133 | 4.12 | 38.1 | 3.48 | 20.5 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 10.8% | 7.1% | 0.1% | 2.2% | 79.8% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
