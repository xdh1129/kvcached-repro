# agent_rr_g67_dram+gds_off1

paths `dram+gds` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1 · offset 1 s · tiered_opt_TIERED_GDS_STAGE_MB=256

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 1138.2 | 64.0% | 7.0% | 10.4% | 18.5% | 0.1% |
| 7 | 1138.2 | 64.1% | 6.7% | 11.9% | 17.2% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.268 | 0.000 | 0.264 | 0.096 | 2.81 | 53.2 | 0.06 | 0.3 |
| 2-6 | 80 | 4996 | 4992 | 0.659 | 0.444 | 0.328 | 0.067 | 3.45 | 50.6 | 0.76 | 3.0 |
| 7-12 | 96 | 12278 | 12112 | 0.888 | 0.524 | 0.379 | 0.092 | 3.72 | 47.2 | 1.62 | 6.3 |
| 13-18 | 96 | 20222 | 20064 | 1.040 | 0.635 | 0.400 | 0.119 | 3.95 | 44.3 | 2.54 | 9.1 |
| 19-24 | 96 | 28166 | 28000 | 1.161 | 0.737 | 0.420 | 0.147 | 4.15 | 43.5 | 3.48 | 10.2 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 13.6% | 8.3% | 0.1% | 2.2% | 75.8% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
