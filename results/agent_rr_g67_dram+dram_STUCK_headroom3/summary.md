# agent_rr_g67_dram+dram

paths `dram+dram` · GPUs 6 7 · 2 x pool meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct per GPU · 24 turns · input 2048 then 1024 tokens · output 300 · sync 1

19 requests ok of 19; 19 needed a wake.

## GPU time by state

| GPU | window s | compute | load | offload | sync | other |
|---|---|---|---|---|---|---|
| 6 | 33.5 | 80.4% | 11.7% | 5.6% | 2.1% | 0.1% |
| 7 | 33.5 | 73.2% | 11.8% | 4.0% | 10.9% | 0.1% |

## Per request, by turn (medians over both GPUs)

| turns | n | history tokens | hit tokens | wait s | offload s | load s | prefill s | decode s | weights GiB/s | KV GiB | KV GiB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 16 | 0 | 0 | 0.495 | 0.000 | 0.342 | 0.160 | 2.81 | 26.8 | 0.06 | 1.0 |
| 2-6 | 3 | 2348 | 2336 | 0.617 | 0.452 | 0.162 | 0.040 | 1.72 | 51.3 | 0.27 | 9.8 |

## Where request time goes (sum over all requests)

| offload wait | load wait | other wait | prefill | decode |
|---|---|---|---|---|
| 5.2% | 12.4% | 0.1% | 4.1% | 78.3% |

compute = a request between proxy ready and its last token. load / offload = a wake / sleep runs on the GPU. sync = this GPU has no request outstanding (waits for the other GPU). prefill = ready -> first token (new input only on a prefix hit; the whole history under vllm).
