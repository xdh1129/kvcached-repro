# agent_rr_g67_gds+gds

paths `gds+gds` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1 · tiered_opt_TIERED_GDS_STAGE_MB=256

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 1675.7 | 43.5% | 25.3% | 30.2% | 0.9% | 0.1% |
| 7 | 1675.7 | 43.5% | 25.2% | 30.5% | 0.8% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 1.818 | 0.000 | 1.797 | 0.090 | 2.84 | 6.4 | 0.06 | 2.4 |
| 2-6 | 80 | 4996 | 4992 | 4.427 | 3.023 | 1.905 | 0.072 | 3.45 | 6.1 | 0.76 | 0.4 |
| 7-12 | 96 | 12278 | 12112 | 4.671 | 3.164 | 2.193 | 0.096 | 3.74 | 5.8 | 1.62 | 0.9 |
| 13-18 | 96 | 20222 | 20064 | 6.149 | 3.378 | 2.469 | 0.125 | 3.95 | 5.7 | 2.54 | 1.3 |
| 19-24 | 96 | 28166 | 28000 | 6.411 | 3.509 | 2.837 | 0.155 | 4.14 | 5.7 | 3.48 | 1.7 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 30.6% | 25.4% | 0.1% | 1.3% | 42.6% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
