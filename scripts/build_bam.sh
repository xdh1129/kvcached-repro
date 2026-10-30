#!/usr/bin/env bash
# Build BaM (github.com/ZaidQureshi/bam) for this machine: sm_120, kernel 6.8,
# NVIDIA open driver 580. Builds libnvm, the libnvm.ko kernel module and the
# benchmarks into third_party/bam/build/b2. Does not load the module or touch
# any NVMe device.
#
# Workarounds:
# - /usr/local/cuda-13.0 holds only GDS, so we use the full system CUDA 12.8
#   (nvcc 12.8 can target sm_120; BaM needs >= 12.3).
# - BaM's bundled libcxx defines __is_convertible, a GCC 13 builtin; rename it
#   with a macro instead of installing an older GCC on this shared host.
# - patches/bam-linux-6.5.patch: class_create() lost its owner argument in
#   Linux 6.4, get_user_pages() lost vmas in 6.5.
# - The module links against nvidia_p2p_* symbols, which needs the driver's
#   Module.symvers. DKMS does not keep it, so build a copy of the driver source.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BAM="$REPO/third_party/bam"
BUILD="$BAM/build"
CUDA=/usr/local/cuda-12.8
DRIVER_VER="$(cat /sys/module/nvidia/version)"

[[ -d "$BAM/.git" ]] || git clone --recursive https://github.com/ZaidQureshi/bam.git "$BAM"
cd "$BAM"
git apply --check "$REPO/patches/bam-linux-6.5.patch" 2>/dev/null \
    && git apply "$REPO/patches/bam-linux-6.5.patch"

mkdir -p "$BUILD"
if [[ ! -x "$BUILD/.tools/bin/cmake" ]]; then
    # FindCUDA still works in CMake 3.x
    uv venv -q "$BUILD/.tools"
    uv pip install -q --python "$BUILD/.tools/bin/python" "cmake>=3.28,<3.32"
fi

if [[ ! -f "$BUILD/nvidia-src/Module.symvers" ]]; then
    cp -r "/usr/src/nvidia-$DRIVER_VER" "$BUILD/nvidia-src"
    make -C "$BUILD/nvidia-src" -j"$(nproc)" modules SYSSRC="/lib/modules/$(uname -r)/build" \
        > "$BUILD/nvidia-build.log" 2>&1
fi

F="-D__is_convertible=__bam_is_convertible"
mkdir -p "$BUILD/b2"
cd "$BUILD/b2"
"$BUILD/.tools/bin/cmake" "$BAM" -Wno-dev -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA" -Dnvidia_archs=120 \
    -DNVIDIA="$BUILD/nvidia-src/nvidia" \
    "-DCMAKE_C_FLAGS=$F" "-DCMAKE_CXX_FLAGS=$F" "-DCUDA_NVCC_FLAGS=$F"
make -j"$(nproc)" libnvm
make -C module
make -j"$(nproc)" benchmarks
ls -l "$BUILD/b2/module/libnvm.ko" "$BUILD/b2/lib/libnvm.so" "$BUILD/b2/bin/"
