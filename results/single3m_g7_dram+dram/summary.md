# single3m_g7_dram+dram

paths `dram+dram` · GPUs 7 · 1 x pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct per GPU · 15 turns · input 2048 then 2048 tokens · output 128 · sync 1 · offset 0 s · tiered_opt_TIERED_GDS_STAGE_MB=256, tiered_opt_TIERED_PARALLEL_SAVE=1

45 requests ok of 45; 45 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 7 | 268.4 | 69.7% | 13.1% | 16.8% | 0.4% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 3 | 0 | 0 | 1.350 | 0.000 | 0.613 | 0.190 | 2.64 | 53.5 | 0.08 | 3.3 |
| 2-6 | 15 | 6528 | 6512 | 2.052 | 0.720 | 0.626 | 0.247 | 2.72 | 51.4 | 1.62 | 14.6 |
| 7-12 | 18 | 18496 | 18416 | 2.137 | 1.077 | 0.663 | 0.343 | 2.89 | 48.7 | 4.56 | 16.5 |
| 13-18 | 9 | 28288 | 28272 | 2.201 | 1.182 | 0.693 | 0.429 | 3.08 | 46.5 | 7.00 | 17.3 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 16.9% | 13.1% | 0.0% | 6.8% | 63.2% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
