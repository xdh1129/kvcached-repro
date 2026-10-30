# agent_rr_g67_dram+gds_off0.3

paths `dram+gds` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1 · offset 0.3 s · tiered_opt_TIERED_GDS_STAGE_MB=256

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 1036.7 | 70.3% | 8.4% | 13.0% | 8.2% | 0.1% |
| 7 | 1036.7 | 70.3% | 8.0% | 15.6% | 6.0% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.267 | 0.000 | 0.263 | 0.094 | 2.81 | 53.2 | 0.06 | 0.3 |
| 2-6 | 80 | 4996 | 4992 | 0.669 | 0.452 | 0.335 | 0.066 | 3.45 | 50.2 | 0.76 | 3.0 |
| 7-12 | 96 | 12278 | 12112 | 0.935 | 0.575 | 0.378 | 0.092 | 3.74 | 45.7 | 1.62 | 6.3 |
| 13-18 | 96 | 20222 | 20064 | 1.140 | 0.721 | 0.399 | 0.123 | 3.95 | 45.3 | 2.54 | 8.8 |
| 19-24 | 96 | 28166 | 28000 | 1.340 | 0.884 | 0.427 | 0.148 | 4.15 | 44.2 | 3.48 | 9.3 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 15.4% | 8.8% | 0.1% | 2.1% | 73.6% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
