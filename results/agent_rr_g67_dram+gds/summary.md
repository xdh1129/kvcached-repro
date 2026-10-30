# agent_rr_g67_dram+gds

paths `dram+gds` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1 · tiered_opt_TIERED_GDS_STAGE_MB=256

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 1077.4 | 67.5% | 12.3% | 19.3% | 0.8% | 0.1% |
| 7 | 1077.4 | 67.7% | 12.4% | 19.3% | 0.6% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.495 | 0.000 | 0.490 | 0.095 | 2.81 | 26.9 | 0.06 | 0.2 |
| 2-6 | 80 | 4996 | 4992 | 1.179 | 0.801 | 0.597 | 0.065 | 3.45 | 27.1 | 0.76 | 1.6 |
| 7-12 | 96 | 12278 | 12112 | 1.605 | 0.976 | 0.650 | 0.090 | 3.75 | 27.2 | 1.62 | 3.7 |
| 13-18 | 96 | 20222 | 20064 | 1.808 | 1.152 | 0.642 | 0.117 | 3.95 | 27.2 | 2.54 | 5.4 |
| 19-24 | 96 | 28166 | 28000 | 1.992 | 1.328 | 0.656 | 0.151 | 4.14 | 27.5 | 3.48 | 6.1 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 19.4% | 12.5% | 0.1% | 1.9% | 66.2% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
