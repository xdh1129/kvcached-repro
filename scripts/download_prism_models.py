"""Download the HF-format weights of the models in Prism's swap_8_2gpus.json."""

from huggingface_hub import snapshot_download

MODELS = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
]
ALLOW = ["*.json", "model*.safetensors", "tokenizer*", "*.model", "*.txt"]

for repo in MODELS:
    print("==", repo, flush=True)
    print(snapshot_download(repo, allow_patterns=ALLOW, ignore_patterns=["original/*"]), flush=True)
