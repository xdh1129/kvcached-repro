# agent_rr_g67_vllm

paths `vllm` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1

384 requests ok of 384; 384 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 1068.5 | 81.3% | 7.7% | 8.3% | 2.5% | 0.1% |
| 7 | 1068.5 | 83.3% | 7.8% | 8.3% | 0.5% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.450 | 0.000 | 0.445 | 0.098 | 2.81 | – | – | – |
| 2-6 | 80 | 4996 | 0 | 0.894 | 0.595 | 0.438 | 0.217 | 3.44 | – | – | – |
| 7-12 | 96 | 12278 | 0 | 1.050 | 0.613 | 0.434 | 0.588 | 3.70 | – | – | – |
| 13-18 | 96 | 20222 | 0 | 1.067 | 0.623 | 0.436 | 1.079 | 3.91 | – | – | – |
| 19-24 | 96 | 28166 | 0 | 0.804 | 0.416 | 0.433 | 1.739 | 4.11 | – | – | – |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 8.5% | 7.9% | 0.1% | 16.8% | 66.8% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
