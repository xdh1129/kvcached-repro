# single3m_g7_dram+ssd

paths `dram+ssd` · GPUs 7 · 1 x pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct per GPU · 15 turns · input 2048 then 2048 tokens · output 128 · sync 1 · offset 0 s · tiered_opt_TIERED_GDS_STAGE_MB=256, tiered_opt_TIERED_PARALLEL_SAVE=1

45 requests ok of 45; 45 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 7 | 282.3 | 66.3% | 15.7% | 17.7% | 0.3% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 3 | 0 | 0 | 1.351 | 0.000 | 0.610 | 0.188 | 2.64 | 53.7 | 0.08 | 0.1 |
| 2-6 | 15 | 6528 | 6512 | 2.126 | 0.718 | 0.667 | 0.240 | 2.73 | 53.7 | 1.62 | 2.1 |
| 7-12 | 18 | 18496 | 18416 | 2.601 | 1.370 | 0.838 | 0.333 | 2.90 | 53.7 | 4.56 | 3.9 |
| 13-18 | 9 | 28288 | 28272 | 2.974 | 2.043 | 1.005 | 0.432 | 3.08 | 53.7 | 7.00 | 5.2 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 17.7% | 15.8% | 0.0% | 6.4% | 60.0% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
