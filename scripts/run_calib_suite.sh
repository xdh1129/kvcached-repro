#!/bin/bash
# Calibration for the split path: per-tier read/write rates of weights and KV on one GPU.
# Uses the wake-storm driver on a single GPU with the three models of the single-GPU
# experiment and no background model. Each round sleeps the awake model and wakes the
# next one, so every sleep and wake moves one model's weights + its primed KV.
#   usage: run_calib_suite.sh [spec ...]     spec = <weights>+<kv>:<parallel>
# Default specs:
#   dram+dram:0  DRAM alone (weights, then KV, one at a time)
#   gds+gds:0    GDS alone
#   dram+gds:1   DRAM weights || GDS KV at once (do they slow each other?)
#   gds+dram:1   GDS weights || DRAM KV at once
# Sleep (suspend) always saves weights, then KV, so its rates are per tier alone.
# Env: GPU (default 7), POOL, KV_GIB (8), PROMPT_LEN (8192), MAX_MODEL_LEN (40960),
#      REPS (9 rounds: 3 wakes per model), TIERED_GDS_STAGE_MB (256)
#      NAME_SUFFIX appended to run names; other TIERED_* options pass through
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPECS=("$@")
((${#SPECS[@]})) || SPECS=(dram+dram:0 gds+gds:0 dram+gds:1 gds+dram:1)
GPU=${GPU:-7}
export GPUS=$GPU DELTAS=0 REPS=${REPS:-9} BG_MODEL=
export POOL=${POOL:-Qwen/Qwen3-32B,Qwen/Qwen3-14B,microsoft/Phi-3.5-mini-instruct}
export KV_GIB=${KV_GIB:-8} PROMPT_LEN=${PROMPT_LEN:-8192} MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
export TIERED_GDS_STAGE_MB=${TIERED_GDS_STAGE_MB:-256}

for spec in "${SPECS[@]}"; do
    p=${spec%:*} par=${spec#*:}
    name=calib_g${GPU}_kv${KV_GIB}_${p}_p${par}${NAME_SUFFIX:-}
    echo "=== $(date -Is) $name"
    PATHS=$p TIERED_PARALLEL=$par bash "$ROOT/scripts/run_wake_storm.sh" "$name" \
        || echo "=== $name failed (exit $?)"
    sleep 10  # let the GPU release memory before the next run's idle check
done
echo "=== $(date -Is) suite done"
