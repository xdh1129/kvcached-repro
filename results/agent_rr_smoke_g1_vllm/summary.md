# agent_rr_smoke_g1_vllm

paths `vllm` · GPUs 1 · 3 x pool meta-llama/Llama-3.2-1B per GPU · 3 turns · input 512 then 256 tokens · output 32 · sync 1

9 requests ok of 9; 9 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 1 | 2.1 | 37.0% | 29.7% | 29.4% | 3.5% | 0.3% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 3 | 0 | 0 | 0.140 | 0.072 | 0.067 | 0.017 | 0.07 | – | – | – |
| 2-6 | 6 | 688 | 0 | 0.146 | 0.077 | 0.066 | 0.013 | 0.07 | – | – | – |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 30.4% | 30.9% | 0.4% | 6.5% | 31.8% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
