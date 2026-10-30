# agent_rr_smoke_g1_dram+gds

paths `dram+gds` · GPUs 1 · 3 x pool meta-llama/Llama-3.2-1B per GPU · 3 turns · input 512 then 256 tokens · output 32 · sync 1 · tiered_opt_TIERED_GDS_STAGE_MB=256

9 requests ok of 9; 9 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 1 | 2.2 | 34.4% | 29.7% | 31.5% | 4.3% | 0.2% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 3 | 0 | 0 | 0.155 | 0.083 | 0.072 | 0.020 | 0.07 | 52.8 | 0.03 | 0.6 |
| 2-6 | 6 | 688 | 672 | 0.159 | 0.087 | 0.070 | 0.010 | 0.07 | 52.8 | 0.03 | 0.6 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 32.7% | 30.8% | 0.6% | 6.1% | 29.8% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
