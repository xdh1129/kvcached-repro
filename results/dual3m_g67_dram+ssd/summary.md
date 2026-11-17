# dual3m_g67_dram+ssd

paths `dram+ssd` · GPUs 6 7 · 1 x pool Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct per GPU · 15 turns · input 2048 then 2048 tokens · output 128 · sync 1 · offset 0 s · tiered_opt_TIERED_PHX_LIB=/home/poc/xdh/phoenix-eval/phoenix/build_t16/libphoenix.so, tiered_opt_TIERED_GDS_STAGE_MB=256, tiered_opt_TIERED_PARALLEL_SAVE=1

90 requests ok of 90; 90 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 364.0 | 50.9% | 22.6% | 25.2% | 1.2% | 0.0% |
| 7 | 364.0 | 51.3% | 22.7% | 25.6% | 0.4% | 0.0% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 6 | 0 | 0 | 2.528 | 0.000 | 1.144 | 0.180 | 2.63 | 26.9 | 0.08 | 0.1 |
| 2-6 | 30 | 6528 | 6512 | 4.014 | 1.351 | 1.150 | 0.227 | 2.72 | 26.9 | 1.62 | 1.0 |
| 7-12 | 36 | 18496 | 18416 | 4.921 | 2.554 | 1.583 | 0.335 | 2.88 | 27.1 | 4.56 | 2.1 |
| 13-18 | 18 | 28288 | 28272 | 5.635 | 3.920 | 1.920 | 0.403 | 3.10 | 27.3 | 7.00 | 2.7 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 25.6% | 22.8% | 0.0% | 4.8% | 46.7% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
