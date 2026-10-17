#!/bin/bash
# Prism 18-model experiment: for each (GPU count, time scale), one dram+ssd policy run
# (LRU + KV pressure) as the reference schedule, then the six configs replaying it.
#   usage: run_18m_suite.sh [gpus:time_scale ...]     default: 1:0.2 2:0.2 1:1 2:1
# Runs: results/prism_18m_<n>g_ts<scale>_ref_dram+ssd, results/prism_18m_<n>g_ts<scale>_sched_<config>.
# A run whose trace replay finished (workload_end in config.txt) is skipped, so the suite can be restarted.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SETS=("$@")
((${#SETS[@]})) || SETS=(1:0.2 2:0.2 1:1 2:1)
CONFIGS=(vllm dram+dram ssd+ssd gds+gds dram+ssd dram+gds)  # vllm: vLLM's own sleep mode, no plugin
declare -A GPUS=([1]="1" [2]="1 2")

run() {  # run <name> <env...>
    local name=$1; shift
    # finished = the trace replay completed (an interrupted run still writes summary.json)
    if grep -q '^workload_end=' "$ROOT/results/$name/config.txt" 2> /dev/null; then
        echo "$(date -Is) skip $name (done)"
        return 0
    fi
    rm -rf "$ROOT/results/$name"
    echo "$(date -Is) start $name"
    env "$@" bash "$ROOT/scripts/run_prism_trace.sh" kvcached "$name" > "$ROOT/logs/$name.log" 2>&1
    echo "$(date -Is) end $name (exit $?)"
}

for set in "${SETS[@]}"; do
    n=${set%:*} ts=${set#*:}
    common=(PRISM_MODELS=18m PLACEMENT="$ROOT/configs/placements/prism_18m_${n}gpu.json" PRISM_GPUS="${GPUS[$n]}"
            TIME_SCALE="$ts" REPLICATION=1 MAX_MODEL_LEN=2048 CONTINUE_ON_FAIL=1)
    ref=prism_18m_${n}g_ts${ts}_ref_dram+ssd
    run "$ref" "${common[@]}" PATHS=dram+ssd KV_PRESSURE=1
    [[ -f "$ROOT/results/$ref/swaps.jsonl" ]] || { echo "no reference schedule for $set; skipping its replays"; continue; }
    for cfg in "${CONFIGS[@]}"; do
        run "prism_18m_${n}g_ts${ts}_sched_$cfg" "${common[@]}" PATHS="$cfg" SCHEDULE="$ROOT/results/$ref"
    done
done
echo "$(date -Is) suite done"
