# Source after activating .venv. Points nvcc-based builds (kvcached's extension,
# FlashInfer's runtime JIT) at the venv's CUDA 13.0 toolkit wheels, which match
# torch cu130. The system toolkit (/usr/local/cuda, 12.8) cannot target sm_120
# in FlashInfer, which needs >= 12.9.
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
# The wheel ships only libcudart.so.13; give the linker an unversioned name.
mkdir -p "$VIRTUAL_ENV/cuda-link"
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$VIRTUAL_ENV/cuda-link/libcudart.so"
export LIBRARY_PATH="$VIRTUAL_ENV/cuda-link${LIBRARY_PATH:+:$LIBRARY_PATH}"
