"""Download the HF-format weights of Prism's models.

usage: python scripts/download_prism_models.py [8m|18m]
  8m   the models of swap_8_2gpus.json (default)
  18m  the 18-model set (model_lst_18.yml without its two Llama-2-13B entries)
"""

import sys

from huggingface_hub import snapshot_download

MODELS = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
]
MODELS_18 = [
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-2", "ibm/PowerLM-3b", "microsoft/phi-1_5", "allenai/OLMo-1B-hf", "bigcode/starcoder2-3b",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2b", "google/gemma-2b-it", "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B",
]
ALLOW = ["*.json", "model*.safetensors", "tokenizer*", "*.model", "*.txt"]

for repo in MODELS_18 if sys.argv[1:] == ["18m"] else MODELS:
    print("==", repo, flush=True)
    print(snapshot_download(repo, allow_patterns=ALLOW, ignore_patterns=["original/*"]), flush=True)
