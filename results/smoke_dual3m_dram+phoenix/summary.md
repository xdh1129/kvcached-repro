# smoke_dual3m_dram+phoenix

paths `dram+phoenix` · GPUs 6 7 · 1 x pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct per GPU · 2 turns · input 2048 then 2048 tokens · output 128 · sync 1 · offset 0 s · tiered_opt_TIERED_PHX_LIB=/home/poc/xdh/phoenix-eval/phoenix/build_t16/libphoenix.so, tiered_opt_TIERED_GDS_STAGE_MB=256, tiered_opt_TIERED_PARALLEL_SAVE=1

12 requests ok of 12; 12 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 35.6 | 57.9% | 22.5% | 18.7% | 0.9% | 0.0% |
| 7 | 35.6 | 58.4% | 22.5% | 18.8% | 0.3% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 6 | 0 | 0 | 2.515 | 0.000 | 1.135 | 0.181 | 2.63 | 26.9 | 0.08 | 0.1 |
| 2-6 | 6 | 2176 | 2160 | 3.699 | 1.202 | 1.143 | 0.196 | 2.66 | 27.0 | 0.62 | 0.4 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 18.9% | 22.6% | 0.0% | 4.0% | 54.5% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
