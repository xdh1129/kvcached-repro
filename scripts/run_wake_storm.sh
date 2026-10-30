#!/bin/bash
# Wake-storm run: the same pool of models on each GPU, woken on all GPUs at controlled
# offsets (scripts/wake_storm.py), to measure DRAM / SSD / GDS data movement when
# several GPUs wake at once.
#   usage: run_wake_storm.sh [run_name]
# Env: GPUS        GPUs to use (default "6 7"; they must be idle)
#      POOL        comma-separated HF repos, the pool on every GPU (default: four 3B-8B models)
#      BG_MODEL    background model per GPU, always awake and decoding (default
#                  meta-llama/Llama-3.2-1B; empty: none). BG_CONCURRENCY streams (16).
#      PATHS       <weights>+<kv> for the tiered_restore plugin (default dram+gds), or
#                  per GPU: "6=dram+dram,7=dram+gds", or vllm (vLLM's sleep mode: KV
#                  dropped and recomputed on wake)
#      KV_GIB      KV each pool model holds while asleep (default 5)
#      PROMPT_LEN  tokens per priming request (default 7168); MAX_MODEL_LEN (8192)
#      DELTAS      wake start offsets between GPUs (default "0,0.1,0.3,1,seq"), REPS each (4)
#      DWELL_S     time between wake and next round (3); SETTLE_S sleep -> wake gap (1)
#      STAGGER=1   GPU i wakes a different pool model than GPU 0 in each round
#      BASE_PORT   first port (default 32001, apart from the Prism runs' 31001)
#      TIERED_*    plugin options pass through (e.g. TIERED_GDS_STAGE_MB=256); off by default
# Output: results/<run_name>/ (report: analysis/wake_storm_report.py)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PROMPT_LEN=${PROMPT_LEN:-7168}  # before experiment.env, whose PROMPT_LEN (256) is for run_workload.sh
source "$ROOT/configs/experiment.env"
PY="$ROOT/.venv/bin/python"

GPUS=(${GPUS:-6 7})
POOL=${POOL:-meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,mistralai/Mistral-7B-Instruct-v0.2}
BG_MODEL=${BG_MODEL-meta-llama/Llama-3.2-1B}
BG_CONCURRENCY=${BG_CONCURRENCY:-16}
PATHS=${PATHS:-dram+gds}
KV_GIB=${KV_GIB:-5}
DELTAS=${DELTAS:-0,0.1,0.3,1,seq}
REPS=${REPS:-4}
DWELL_S=${DWELL_S:-3}
SETTLE_S=${SETTLE_S:-1}
BASE_PORT=${BASE_PORT:-32001}
RUN_NAME=${1:-wake_storm_$(date +%Y%m%d_%H%M%S)}
RUN_DIR="$ROOT/results/$RUN_NAME"

if [[ -e "$RUN_DIR" ]]; then
    echo "$RUN_DIR exists; pick another run name" >&2
    exit 1
fi
for g in "${GPUS[@]}"; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    if ((used > 1024)); then
        echo "GPU $g already has ${used} MiB in use; pick idle GPUs via GPUS" >&2
        exit 1
    fi
done
mkdir -p "$RUN_DIR"

IFS=, read -ra pool <<< "$POOL"
port=$BASE_PORT
: > "$RUN_DIR/instances.tsv"
for g in "${GPUS[@]}"; do
    for i in "${!pool[@]}"; do
        printf 'g%s_p%s\t%s\t%s\t%s\n' "$g" "$i" "${pool[$i]}" "$g" "$port" >> "$RUN_DIR/instances.tsv"
        port=$((port + 1))
    done
    if [[ -n "$BG_MODEL" ]]; then
        printf 'g%s_bg\t%s\t%s\t%s\n' "$g" "$BG_MODEL" "$g" "$port" >> "$RUN_DIR/instances.tsv"
        port=$((port + 1))
    fi
done

if [[ "$PATHS" == vllm ]]; then
    unset TIERED_WEIGHTS TIERED_KV TIERED_ROOT  # cleanup removes TIERED_ROOT: never one from the caller
elif [[ "$PATHS" == *=* ]]; then
    # per-GPU paths: the plugin is on for all; instance_paths.tsv picks each instance's paths
    IFS=, read -ra per_gpu <<< "$PATHS"
    : > "$RUN_DIR/instance_paths.tsv"
    for spec in "${per_gpu[@]}"; do
        g=${spec%%=*} p=${spec#*=}
        awk -F'\t' -v g="$g" -v p="$p" '$3 == g {print $1 "\t" p}' "$RUN_DIR/instances.tsv" \
            >> "$RUN_DIR/instance_paths.tsv"
    done
    first=${per_gpu[0]#*=}
    export TIERED_WEIGHTS=${first%+*} TIERED_KV=${first#*+}
else
    export TIERED_WEIGHTS=${PATHS%+*} TIERED_KV=${PATHS#*+}
fi
if [[ "$PATHS" != vllm ]]; then
    export TIERED_PARALLEL=${TIERED_PARALLEL:-1}
    export TIERED_ROOT=/mnt/raid0/xdh/tiered/$RUN_NAME
fi

{
    echo "experiment=wake_storm"
    echo "date=$(date -Is)"
    echo "repo_git=$(git -C "$ROOT" rev-parse --short HEAD 2> /dev/null || echo none)"
    echo "kvcached_git=$(git -C "$ROOT/third_party/kvcached" describe --tags --always)"
    "$PY" -c "import vllm, torch; print(f'vllm={vllm.__version__}'); print(f'torch={torch.__version__}')"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "${GPUS[0]}")"
    echo "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPUS[0]}")"
    echo "gpus=${GPUS[*]}"
    echo "pool=$POOL"
    echo "bg_model=$BG_MODEL"
    echo "bg_concurrency=$BG_CONCURRENCY"
    echo "paths=$PATHS"
    [[ "$PATHS" != vllm ]] && echo "tiered_parallel=$TIERED_PARALLEL" && echo "tiered_root=$TIERED_ROOT"
    env | grep -E '^TIERED_(WEIGHTS_SAVE|KV_MERGE|WEIGHTS_PACE|DRAM_KV_RESERVE_GB|GDS_GATE|GDS_STAGE_MB|GDS_STAGE_BUFS)=' \
        | sed 's/^/tiered_opt_/' || true
    echo "kv_gib=$KV_GIB"
    echo "prompt_len=$PROMPT_LEN"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "deltas=$DELTAS"
    echo "reps=$REPS"
    echo "dwell_s=$DWELL_S"
    echo "settle_s=$SETTLE_S"
    echo "stagger=${STAGGER:-0}"
    echo "kvcached_gpu_utilization=$KVCACHED_GPU_UTILIZATION"
} > "$RUN_DIR/config.txt"

VRAM_PID= PCIE_PID=
cleanup() {
    set +e  # under set -e, a child's non-zero wait status would abort the cleanup
    for p in $VRAM_PID $PCIE_PID; do kill "$p" 2> /dev/null; wait "$p" 2> /dev/null; done
    bash "$ROOT/scripts/stop_servers.sh" "$RUN_DIR" || true
    [[ -n "${TIERED_ROOT:-}" ]] && rm -rf "$TIERED_ROOT"
    "$PY" "$ROOT/analysis/wake_storm_report.py" "$RUN_DIR" || true
}
trap cleanup EXIT

"$PY" "$ROOT/scripts/vram_monitor.py" "$RUN_DIR" --interval 0.25 &
VRAM_PID=$!
"$PY" "$ROOT/scripts/pcie_monitor.py" "$RUN_DIR" --gpus "${GPUS[@]}" --interval 0.1 &
PCIE_PID=$!

echo "launch_start=$(date +%s.%N)" >> "$RUN_DIR/config.txt"
if ! SLEEP_AFTER_START=1 bash "$ROOT/scripts/launch_servers.sh" kvcached "$RUN_DIR"; then
    echo "launch_failed=1" >> "$RUN_DIR/config.txt"
    echo "launch failed; keeping data in $RUN_DIR" >&2
    exit 1
fi
echo "launch_end=$(date +%s.%N)" >> "$RUN_DIR/config.txt"

storm_args=(--kv-gib "$KV_GIB" --prompt-len "$PROMPT_LEN" --max-model-len "$MAX_MODEL_LEN"
            --deltas "$DELTAS" --reps "$REPS" --dwell-s "$DWELL_S" --settle-s "$SETTLE_S"
            --bg-concurrency "$BG_CONCURRENCY")
[[ "${STAGGER:-0}" == 1 ]] && storm_args+=(--stagger)
"$PY" "$ROOT/scripts/wake_storm.py" "$RUN_DIR" "${storm_args[@]}" | tee "$RUN_DIR/storm.log"
sleep 3
echo "done: $RUN_DIR"
