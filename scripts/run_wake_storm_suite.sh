#!/bin/bash
# Wake-storm suite on GPUs 6+7: both GPUs wake at the same instant (delta 0), one run
# per offload path. Other settings: run_wake_storm.sh defaults (KV_GIB etc. can be set).
#   usage: run_wake_storm_suite.sh [paths ...]
# A failed run is logged and the suite goes on with the next path.
# NAME_SUFFIX is appended to each run name (e.g. _stage256 for a TIERED_GDS_STAGE_MB ablation).
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PATHS_LIST=("$@")
((${#PATHS_LIST[@]})) || PATHS_LIST=(dram+gds dram+ssd dram+dram ssd+ssd gds+gds vllm)
export GPUS=${GPUS:-6 7} DELTAS=${DELTAS:-0} REPS=${REPS:-12}
KV_GIB=${KV_GIB:-5}

for p in "${PATHS_LIST[@]}"; do
    name=wake_storm_g67_kv${KV_GIB}_d${DELTAS//,/_}_$p${NAME_SUFFIX:-}
    echo "=== $(date -Is) $name"
    PATHS=$p KV_GIB=$KV_GIB bash "$ROOT/scripts/run_wake_storm.sh" "$name" \
        || echo "=== $name failed (exit $?)"
    sleep 10  # let the GPUs release memory before the next run's idle check
done
echo "=== $(date -Is) suite done"
