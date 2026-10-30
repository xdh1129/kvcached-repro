# Interleaved multi-turn agents, GPUs 6+7 (8 agents per GPU, 24 turns, every request swaps)

Pool per GPU: 2 x (Llama-3.2-3B-Instruct, Phi-3.5-mini, Llama-3.1-8B, Llama-3.1-8B-Instruct). Turn 1 input 2048 tokens,
later 1024, output 300; no initial history (final ~32.8k tokens). Prefix caching on; LRU eviction of weights + cached
history KV (swap_proxy --keep, headroom 8 GiB). GDS paths with TIERED_GDS_STAGE_MB=256; otherwise vLLM-default plugin settings.
vllm = vLLM's own sleep mode: the history KV is dropped on sleep and re-prefilled on the next turn.

## GPU time (mean of GPUs 6 and 7)

| paths | workload s | compute (prefill+decode) | of which prefill | waiting: load | waiting: offload | waiting total | sync |
|---|---|---|---|---|---|---|---|
| dram+dram | 985 | 73.4% | 2.0% | 11.6% | 14.3% | **25.9%** | 0.7% |
| dram+ssd | 1098 | 65.8% | 1.8% | 15.1% | 18.4% | **33.5%** | 0.7% |
| dram+gds | 1077 | 67.6% | 1.9% | 12.4% | 19.3% | **31.6%** | 0.7% |
| ssd+ssd | 1652 | 43.9% | 1.3% | 24.8% | 30.0% | **54.8%** | 1.3% |
| gds+gds | 1676 | 43.5% | 1.3% | 25.2% | 30.3% | **55.6%** | 0.9% |
| vllm | 1069 | 82.3% | 16.6% | 7.7% | 8.3% | **16.1%** | 1.5% |

## Request arrival -> first token (median, s), by turn

| paths | turns 2-6 | turns 7-12 | turns 13-18 | turns 19-24 | turns 19-24: offload / load / prefill |
|---|---|---|---|---|---|
| dram+dram | 1.16 | 1.36 | 1.63 | 1.72 | 0.80 / 0.77 / 0.14 |
| dram+ssd | 1.33 | 1.77 | 2.21 | 2.58 | 1.32 / 1.10 / 0.15 |
| dram+gds | 1.24 | 1.70 | 1.93 | 2.14 | 1.33 / 0.66 / 0.15 |
| ssd+ssd | 4.33 | 4.64 | 6.08 | 6.42 | 3.44 / 2.75 / 0.16 |
| gds+gds | 4.49 | 4.77 | 6.27 | 6.57 | 3.51 / 2.84 / 0.16 |
| vllm | 1.10 | 1.66 | 2.07 | 2.73 | 0.42 / 0.43 / 1.74 |

## Load bandwidth, turns 19-24 (median per request; ~3.5 GiB history KV)

| paths | weights GiB/s | KV GiB/s | prefix hit tokens / history |
|---|---|---|---|
| dram+dram | 20.4 | 10.9 | 28000 / 28166 |
| dram+ssd | 27.4 | 4.1 | 28000 / 28166 |
| dram+gds | 27.5 | 6.1 | 28000 / 28166 |
| ssd+ssd | 6.5 | 1.7 | 28000 / 28166 |
| gds+gds | 5.7 | 1.7 | 28000 / 28166 |
| vllm | – | – | 0 / 28166 |
