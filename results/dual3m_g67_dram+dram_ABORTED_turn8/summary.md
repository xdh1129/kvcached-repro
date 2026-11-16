# dual3m_g67_dram+dram

paths `dram+dram` · GPUs 6 7 · 1 x pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct per GPU · 15 turns · input 2048 then 2048 tokens · output 128 · sync 1 · offset 0 s · tiered_opt_TIERED_PHX_LIB=/home/poc/xdh/phoenix-eval/phoenix/build_t16/libphoenix.so, tiered_opt_TIERED_GDS_STAGE_MB=256, tiered_opt_TIERED_PARALLEL_SAVE=1

66 requests ok of 66; 66 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 235.4 | 54.7% | 19.8% | 24.5% | 0.9% | 0.0% |
| 7 | 235.4 | 55.3% | 19.8% | 24.5% | 0.4% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 6 | 0 | 0 | 2.524 | 0.000 | 1.168 | 0.183 | 2.63 | 26.8 | 0.08 | 2.2 |
| 2-6 | 30 | 6528 | 6512 | 3.850 | 1.241 | 1.167 | 0.228 | 2.72 | 25.8 | 1.62 | 6.7 |
| 7-12 | 30 | 17408 | 17392 | 4.020 | 1.958 | 1.243 | 0.315 | 2.87 | 24.8 | 4.25 | 8.0 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 24.6% | 20.0% | 0.0% | 4.7% | 50.7% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
