# dual3m_g67_dram+dram

paths `dram+dram` · GPUs 6 7 · 1 x pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct per GPU · 15 turns · input 2048 then 2048 tokens · output 128 · sync 1 · offset 0 s · tiered_opt_TIERED_PHX_LIB=/home/poc/xdh/phoenix-eval/phoenix/build_t16/libphoenix.so, tiered_opt_TIERED_GDS_STAGE_MB=256, tiered_opt_TIERED_PARALLEL_SAVE=1

90 requests ok of 90; 90 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 334.2 | 55.4% | 19.9% | 23.7% | 0.9% | 0.0% |
| 7 | 334.2 | 56.0% | 19.9% | 23.8% | 0.3% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 6 | 0 | 0 | 2.525 | 0.000 | 1.132 | 0.183 | 2.63 | 26.8 | 0.08 | 1.6 |
| 2-6 | 30 | 6528 | 6512 | 3.847 | 1.244 | 1.172 | 0.233 | 2.72 | 25.8 | 1.62 | 6.7 |
| 7-12 | 36 | 18496 | 18416 | 4.040 | 2.000 | 1.259 | 0.325 | 2.89 | 24.5 | 4.56 | 7.8 |
| 13-18 | 18 | 28288 | 28272 | 4.178 | 2.214 | 1.327 | 0.404 | 3.14 | 23.3 | 7.00 | 8.5 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 23.9% | 20.1% | 0.0% | 5.1% | 50.9% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
