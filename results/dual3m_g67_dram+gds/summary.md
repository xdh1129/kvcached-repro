# dual3m_g67_dram+gds

paths `dram+gds` · GPUs 6 7 · 1 x pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct per GPU · 15 turns · input 2048 then 2048 tokens · output 128 · sync 1 · offset 0 s · tiered_opt_TIERED_PHX_LIB=/home/poc/xdh/phoenix-eval/phoenix/build_t16/libphoenix.so, tiered_opt_TIERED_GDS_STAGE_MB=256, tiered_opt_TIERED_PARALLEL_SAVE=1

90 requests ok of 90; 90 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 333.3 | 55.7% | 21.1% | 22.5% | 0.7% | 0.0% |
| 7 | 333.3 | 56.1% | 21.0% | 22.3% | 0.5% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 6 | 0 | 0 | 2.526 | 0.000 | 1.152 | 0.179 | 2.63 | 27.1 | 0.08 | 0.1 |
| 2-6 | 30 | 6528 | 6512 | 3.682 | 1.184 | 1.155 | 0.234 | 2.72 | 27.0 | 1.62 | 1.2 |
| 7-12 | 36 | 18496 | 18416 | 3.715 | 1.882 | 1.194 | 0.323 | 2.89 | 26.9 | 4.56 | 2.8 |
| 13-18 | 18 | 28288 | 28272 | 3.751 | 2.581 | 1.811 | 0.403 | 3.11 | 27.0 | 7.00 | 4.2 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 22.6% | 21.2% | 0.0% | 5.2% | 51.1% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
