# agent_rr_g67_dram+dram

paths `dram+dram` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 984.8 | 73.3% | 11.6% | 14.3% | 0.8% | 0.1% |
| 7 | 984.8 | 73.4% | 11.6% | 14.3% | 0.6% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.496 | 0.000 | 0.481 | 0.095 | 2.82 | 26.8 | 0.06 | 1.0 |
| 2-6 | 80 | 4996 | 4992 | 1.107 | 0.688 | 0.518 | 0.067 | 3.43 | 25.5 | 0.76 | 6.6 |
| 7-12 | 96 | 12278 | 12112 | 1.264 | 0.714 | 0.596 | 0.087 | 3.71 | 23.9 | 1.62 | 9.0 |
| 13-18 | 96 | 20222 | 20064 | 1.512 | 0.765 | 0.692 | 0.112 | 3.91 | 22.5 | 2.54 | 10.1 |
| 19-24 | 96 | 28166 | 28000 | 1.575 | 0.799 | 0.771 | 0.143 | 4.12 | 20.4 | 3.48 | 10.9 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 14.4% | 11.7% | 0.1% | 2.0% | 71.9% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
