# agent_rr_g67_ssd+ssd

paths `ssd+ssd` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 1652.5 | 43.8% | 24.5% | 29.7% | 1.8% | 0.0% |
| 7 | 1652.5 | 43.9% | 25.0% | 30.3% | 0.8% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 1.826 | 0.000 | 1.825 | 0.092 | 2.81 | 6.5 | 0.06 | 0.0 |
| 2-6 | 80 | 4996 | 4992 | 4.257 | 2.934 | 1.932 | 0.073 | 3.44 | 6.6 | 0.76 | 0.4 |
| 7-12 | 96 | 12278 | 12112 | 4.547 | 3.066 | 2.192 | 0.099 | 3.70 | 6.5 | 1.62 | 0.9 |
| 13-18 | 96 | 20222 | 20064 | 5.954 | 3.285 | 2.444 | 0.129 | 3.91 | 6.5 | 2.54 | 1.3 |
| 19-24 | 96 | 28166 | 28000 | 6.254 | 3.436 | 2.747 | 0.157 | 4.11 | 6.5 | 3.48 | 1.7 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 30.4% | 25.1% | 0.1% | 1.3% | 43.1% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
