#!/bin/bash
# Bursty multi-model workload from upstream benchmarks/bench_latency_benefit/run_benchmark.sh:
# each instance gets a 0 -> PEAK_RPS -> 1 ramp, and instance i+1 starts MODEL_DELAY
# seconds after instance i, so the peaks are staggered across models.
#   usage: run_workload.sh <run_dir>
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/configs/experiment.env"
source "$ROOT/.venv/bin/activate"

RUN_DIR=${1:?usage: run_workload.sh <run_dir>}
BENCH="$ROOT/third_party/kvcached/benchmarks/bench_latency_benefit/bench_kvcached_vllm.py"
mkdir -p "$RUN_DIR/metrics"

RAMP_START_RPS=0
RAMP_END_RPS=1
RAMP_INCREMENT=1
RAMP_UP_DURATION=$(((PEAK_RPS - RAMP_START_RPS) / RAMP_INCREMENT))
MODEL_DELAY=$((RAMP_UP_DURATION / 4 + RAMP_UP_DURATION * 2))
NUM_PROMPTS=$((PEAK_RPS * PEAK_RPS))
NUM_MODELS=${#PORTS[@]}
UNIFIED_START_TIME=$(date +%s.%N)
eos_args=()
[[ "$IGNORE_EOS" == 1 ]] && eos_args=(--ignore-eos)

trap 'kill $(jobs -p) 2> /dev/null' INT TERM

pids=()
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    name="inst$((i + 1))"
    if ((i > 0)); then
        echo "waiting ${MODEL_DELAY}s before $name"
        sleep "$MODEL_DELAY"
    fi
    model_num_prompts=$((NUM_PROMPTS + (NUM_MODELS - i) * MODEL_DELAY * RAMP_END_RPS))
    echo "starting client for $name (port $port, $model_num_prompts prompts)"
    python "$BENCH" \
        --backend vllm \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$PROMPT_LEN" \
        --random-output-len "$COMPLETION_LEN" \
        "${eos_args[@]}" \
        --num-prompts "$model_num_prompts" \
        --host localhost \
        --port "$port" \
        --endpoint /v1/completions \
        --save-result \
        --result-filename "$RUN_DIR/metrics/${name}.json" \
        --metadata "unified_start_time=$UNIFIED_START_TIME" \
        --ramp-up-strategy ramp-up-down \
        --ramp-start-rps "$RAMP_START_RPS" \
        --ramp-peak-rps "$PEAK_RPS" \
        --ramp-end-rps "$RAMP_END_RPS" \
        --ramp-increment "$RAMP_INCREMENT" \
        > "$RUN_DIR/client_${name}.log" 2>&1 &
    pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "client inst$((i + 1)) failed; see client_inst$((i + 1)).log" >&2; rc=1; }
done
exit $rc
