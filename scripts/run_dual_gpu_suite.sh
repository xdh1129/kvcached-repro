#!/bin/bash
# Two-GPU version of run_single_gpu_suite.sh: GPUs 6 and 7 each run the same three models
# (large, medium, small) round robin, closed loop, every request starting on both GPUs at
# the same time (agent_rr.py sync mode, offset 0: step k starts once both GPUs finished k-1).
# Parallel wake and parallel sleep on every path, fast monitor (10 ms; NVML ~24 ms).
#   usage: run_dual_gpu_suite.sh [paths ...]
#          (default: dram+dram dram+gds dram+phoenix split split+phoenix)
# phoenix paths: GDS writes through md0, Phoenix P2P reads from the RAID0 member NVMes
# (tiered_restore/phoenix.py). They need phoenixfs.ko loaded (STAGING mode) and read access
# for this user to /dev/phxfs_dev, /dev/nvme1n1 and /dev/nvme2n1; checked before each run.
# Env: GPUS ("6 7"), SPLIT_FRAC (0.20), NAME_PREFIX (dual3m_g67), TURNS (15), FIRST_LEN (2048),
#      TURN_LEN (2048), OUT_LEN (128), TIERED_PHX_LIB (16-worker libphoenix build).
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PATHS_LIST=("$@")
((${#PATHS_LIST[@]})) || PATHS_LIST=(dram+dram dram+gds dram+phoenix split split+phoenix)
export GPUS=${GPUS:-6 7}
SPLIT_FRAC=${SPLIT_FRAC:-0.20}
PREFIX=${NAME_PREFIX:-dual3m_g67}
export COPIES=1 SYNC=1 OFFSET_S=0 MONITOR_FAST=1
export POOL=${POOL:-Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct}
export TURNS=${TURNS:-15} FIRST_LEN=${FIRST_LEN:-2048} TURN_LEN=${TURN_LEN:-2048} OUT_LEN=${OUT_LEN:-128}
export TIERED_PARALLEL=1 TIERED_PARALLEL_SAVE=1 TIERED_GDS_STAGE_MB=${TIERED_GDS_STAGE_MB:-256}
export TIERED_PHX_LIB=${TIERED_PHX_LIB:-/home/poc/xdh/phoenix-eval/phoenix/build_t16/libphoenix.so}

phoenix_ready() {
    grep -q '^phoenixfs ' /proc/modules || { echo "phoenixfs.ko is not loaded" >&2; return 1; }
    for f in /dev/phxfs_dev /dev/nvme1n1 /dev/nvme2n1; do
        [[ -r $f ]] || { echo "no read access to $f" >&2; return 1; }
    done
}

for p in "${PATHS_LIST[@]}"; do
    name=${PREFIX}_$p
    if [[ "$p" == split || "$p" == split+phoenix ]]; then
        name=${PREFIX}_${p}_f$SPLIT_FRAC
        export TIERED_SPLIT_FRAC=$SPLIT_FRAC
    fi
    echo "=== $(date -Is) $name"
    if [[ "$p" == *phoenix* ]] && ! phoenix_ready; then
        echo "=== $name skipped: Phoenix not ready"
        continue
    fi
    PATHS=$p bash "$ROOT/scripts/run_agent_rr.sh" "$name" || echo "=== $name failed (exit $?)"
    unset TIERED_SPLIT_FRAC
    sleep 10  # let the GPUs release memory before the next run's idle check
done
echo "=== $(date -Is) suite done"
