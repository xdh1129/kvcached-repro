#!/bin/bash
# Create .venv with vLLM 0.30.0 and kvcached (from third_party/kvcached, v0.1.6).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

git submodule update --init third_party/kvcached

uv venv .venv --python 3.12
source .venv/bin/activate

# nvcc must match the CUDA 13.0 runtime headers that torch cu130 pins, or
# FlashInfer's JIT fails with "CUDA compiler and CUDA toolkit headers are incompatible".
uv pip install "vllm==0.30.0" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" \
    "nvidia-nvvm==13.0.*"
uv pip install -r third_party/kvcached/requirements.txt
uv pip install setuptools wheel ninja pandas matplotlib nvidia-ml-py

source scripts/cuda_env.sh
uv pip install -e third_party/kvcached --no-build-isolation --no-cache-dir

python -c "import torch, vllm, kvcached; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__); print('kvcached', kvcached.__file__)"
