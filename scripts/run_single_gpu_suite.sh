#!/bin/bash
# Single-GPU closed-loop agent experiment: three models (large, medium, small) in round robin,
# one request at a time, every path with parallel wake and parallel sleep.
#   usage: run_single_gpu_suite.sh [paths ...]    (default: dram+dram dram+ssd dram+gds split)
# Env: GPU (7), SPLIT_FRAC (0.20), NAME_PREFIX (single3m_g<GPU>), TURNS (15), FIRST_LEN (2048),
#      TURN_LEN (2048), OUT_LEN (128). A failed run is logged and the suite goes on.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PATHS_LIST=("$@")
((${#PATHS_LIST[@]})) || PATHS_LIST=(dram+dram dram+ssd dram+gds split)
GPU=${GPU:-7}
SPLIT_FRAC=${SPLIT_FRAC:-0.20}
PREFIX=${NAME_PREFIX:-single3m_g$GPU}
export GPUS=$GPU COPIES=1 SYNC=1
export POOL=${POOL:-Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct}
export TURNS=${TURNS:-15} FIRST_LEN=${FIRST_LEN:-2048} TURN_LEN=${TURN_LEN:-2048} OUT_LEN=${OUT_LEN:-128}
export TIERED_PARALLEL=1 TIERED_PARALLEL_SAVE=1 TIERED_GDS_STAGE_MB=${TIERED_GDS_STAGE_MB:-256}

for p in "${PATHS_LIST[@]}"; do
    name=${PREFIX}_$p
    if [[ "$p" == split ]]; then
        name=${PREFIX}_split_f$SPLIT_FRAC
        export TIERED_SPLIT_FRAC=$SPLIT_FRAC
    fi
    echo "=== $(date -Is) $name"
    PATHS=$p bash "$ROOT/scripts/run_agent_rr.sh" "$name" || echo "=== $name failed (exit $?)"
    unset TIERED_SPLIT_FRAC
    sleep 10  # let the GPU release memory before the next run's idle check
done
echo "=== $(date -Is) suite done"
