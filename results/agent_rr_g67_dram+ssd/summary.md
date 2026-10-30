# agent_rr_g67_dram+ssd

paths `dram+ssd` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 1098.3 | 65.7% | 15.1% | 18.3% | 0.7% | 0.1% |
| 7 | 1098.3 | 65.8% | 15.1% | 18.5% | 0.6% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.493 | 0.000 | 0.487 | 0.089 | 2.81 | 27.0 | 0.06 | 0.2 |
| 2-6 | 80 | 4996 | 4992 | 1.271 | 0.787 | 0.639 | 0.065 | 3.43 | 27.1 | 0.76 | 1.5 |
| 7-12 | 96 | 12278 | 12112 | 1.675 | 0.942 | 0.788 | 0.089 | 3.70 | 27.2 | 1.62 | 3.0 |
| 13-18 | 96 | 20222 | 20064 | 2.090 | 1.150 | 0.942 | 0.115 | 3.92 | 27.1 | 2.54 | 3.8 |
| 19-24 | 96 | 28166 | 28000 | 2.430 | 1.323 | 1.101 | 0.150 | 4.12 | 27.4 | 3.48 | 4.1 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 18.5% | 15.2% | 0.1% | 1.8% | 64.4% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
