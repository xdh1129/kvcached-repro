"""Placement files for Prism's 18-model experiment on 1 or 2 of our GPUs.

Prism's repo places its 18 models 3 per GPU on 6 GPUs
(benchmark/multi-model/model_configs/8_gpu_18_model_our.json, every model a
Llama-3.2-1B placeholder). Here the models are model_lst_18.yml without its two
Llama-2-13B entries, in file order (model_1 = first entry). The 6 Prism GPUs fold
onto ours: 2 GPUs -> Prism GPUs 0-2 on the first, 3-5 on the second; 1 GPU -> all.

usage: python scripts/make_placement_18m.py   (writes configs/placements/prism_18m_{1,2}gpu.json)
"""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRISM_FILE = os.path.join(ROOT, "third_party/prism-research/benchmark/multi-model/model_configs/"
                          "8_gpu_18_model_our.json")
MODELS = [
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-2", "ibm/PowerLM-3b", "microsoft/phi-1_5", "allenai/OLMo-1B-hf", "bigcode/starcoder2-3b",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2b", "google/gemma-2b-it",
    "google/gemma-2-2b-it", "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B",
]


def main():
    prism = json.load(open(PRISM_FILE))
    out_dir = os.path.join(ROOT, "configs", "placements")
    os.makedirs(out_dir, exist_ok=True)
    for n_gpus in (1, 2):
        placement = []
        for m in sorted(prism, key=lambda m: int(m["model_name"].split("_")[1])):
            prism_gpu = m["init_placements"][0]["gpu_ids"][0]
            path = MODELS[int(m["model_name"].split("_")[1]) - 1]
            placement.append({"model_name": m["model_name"], "model_path": path, "tokenizer_path": path,
                              "tp_size": 1, "init_placements": [{"gpu_ids": [prism_gpu * n_gpus // 6],
                                                                 "on": False}]})
        path = os.path.join(out_dir, f"prism_18m_{n_gpus}gpu.json")
        with open(path, "w") as f:
            json.dump(placement, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
