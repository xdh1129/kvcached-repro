# single3m_g7_dram+gds

paths `dram+gds` · GPUs 7 · 1 x pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct per GPU · 15 turns · input 2048 then 2048 tokens · output 128 · sync 1 · offset 0 s · tiered_opt_TIERED_GDS_STAGE_MB=256, tiered_opt_TIERED_PARALLEL_SAVE=1

45 requests ok of 45; 45 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 7 | 268.8 | 69.8% | 14.5% | 15.4% | 0.3% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 3 | 0 | 0 | 1.353 | 0.000 | 0.615 | 0.194 | 2.63 | 53.5 | 0.08 | 0.1 |
| 2-6 | 15 | 6528 | 6512 | 2.003 | 0.649 | 0.635 | 0.244 | 2.72 | 51.5 | 1.62 | 2.3 |
| 7-12 | 18 | 18496 | 18416 | 2.126 | 1.084 | 0.683 | 0.337 | 2.90 | 48.8 | 4.56 | 4.9 |
| 13-18 | 9 | 28288 | 28272 | 2.219 | 1.519 | 0.928 | 0.411 | 3.13 | 46.6 | 7.00 | 7.2 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 15.4% | 14.6% | 0.0% | 6.7% | 63.3% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
