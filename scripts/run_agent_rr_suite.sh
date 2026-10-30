#!/bin/bash
# Interleaved-agents suite on GPUs 6+7, one run per offload path (run_agent_rr.sh defaults
# otherwise). Paths with GDS get TIERED_GDS_STAGE_MB=256 (STAGE_MB to change, 0 for none).
#   usage: run_agent_rr_suite.sh [paths ...]
# A failed run is logged and the suite goes on with the next path.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PATHS_LIST=("$@")
((${#PATHS_LIST[@]})) || PATHS_LIST=(dram+dram dram+ssd dram+gds ssd+ssd gds+gds vllm)
export GPUS=${GPUS:-6 7}
STAGE_MB=${STAGE_MB:-256}

for p in "${PATHS_LIST[@]}"; do
    name=agent_rr_g67_$p${NAME_SUFFIX:-}
    stage=()
    [[ "$p" == *gds* && "$STAGE_MB" != 0 ]] && stage=(TIERED_GDS_STAGE_MB="$STAGE_MB")
    echo "=== $(date -Is) $name ${stage[*]}"
    env "${stage[@]}" PATHS="$p" bash "$ROOT/scripts/run_agent_rr.sh" "$name" || echo "=== $name failed (exit $?)"
    sleep 10  # let the GPUs release memory before the next run's idle check
done
echo "=== $(date -Is) suite done"
