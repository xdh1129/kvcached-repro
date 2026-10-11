#!/bin/bash
# Measured run of Prism's "8 models on 2 GPUs" experiment at its default setting
# (swap_8_2gpus.json placement, real_trace.pkl, time scale 0.2, replication 1).
#   usage: run_prism_trace.sh [kvcached|static] [run_name]
# Env: PRISM_GPUS  our GPU ids for Prism's GPU0 and GPU1 (default "1 2")
#      BASE_PORT   first instance port (default 31001)
#      PLACEMENT   Prism-format placement JSON (default: swap_8_2gpus.json)
#      SWAP=1      DRAM weight swapping: instances start asleep and scripts/swap_proxy.py
#                  wakes them on demand, evicting idle LRU instances for room
#      HEADROOM_GIB  swap proxy free-memory margin beyond the wake size (default 3)
#      CONTINUE_ON_FAIL=1  an instance that fails to start is skipped (failed.tsv) and the
#                  trace still runs; requests for it fail and are recorded as failures
# Output: results/<run_name>/ (see analysis/prism_run_report.py for the report)
# A launch failure (e.g. out of memory) is a result too: the run stops, keeps its
# data and still writes the report.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}  # trace max prompt+output < 2k tokens
source "$ROOT/configs/experiment.env"
PY="$ROOT/.venv/bin/python"

MODE=${1:-kvcached}
RUN_NAME=${2:-prism_8m2g_${MODE}_$(date +%Y%m%d_%H%M%S)}
RUN_DIR="$ROOT/results/$RUN_NAME"
PRISM_GPUS=(${PRISM_GPUS:-1 2})
BASE_PORT=${BASE_PORT:-31001}
TIME_SCALE=${TIME_SCALE:-0.2}
REPLICATION=${REPLICATION:-1}
SWAP=${SWAP:-0}
HEADROOM_GIB=${HEADROOM_GIB:-3}
PROXY_PORT=${PROXY_PORT:-30999}
PLACEMENT=${PLACEMENT:-"$ROOT/third_party/prism-research/experiments/experiments_1/model_configs/swap_8_2gpus.json"}
mkdir -p "$RUN_DIR"

"$PY" - "$PLACEMENT" "${PRISM_GPUS[*]}" "$BASE_PORT" > "$RUN_DIR/instances.tsv" <<'EOF'
import json, sys
placement, gpus, port = json.load(open(sys.argv[1])), sys.argv[2].split(), int(sys.argv[3])
for i, m in enumerate(placement):
    prism_gpu = m["init_placements"][0]["gpu_ids"][0]
    print(f"{m['model_name']}\t{m['model_path']}\t{gpus[prism_gpu]}\t{port + i}")
EOF

for g in "${PRISM_GPUS[@]}"; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    if ((used > 1024)); then
        echo "GPU $g already has ${used} MiB in use; pick idle GPUs via PRISM_GPUS" >&2
        exit 1
    fi
done

{
    echo "mode=$MODE"
    echo "date=$(date -Is)"
    echo "repo_git=$(git -C "$ROOT" rev-parse --short HEAD 2> /dev/null || echo none)"
    echo "kvcached_git=$(git -C "$ROOT/third_party/kvcached" describe --tags --always)"
    echo "prism_git=$(git -C "$ROOT/third_party/prism-research" rev-parse --short HEAD)"
    "$PY" -c "import vllm, torch; print(f'vllm={vllm.__version__}'); print(f'torch={torch.__version__}')"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "${PRISM_GPUS[0]}")"
    echo "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${PRISM_GPUS[0]}")"
    echo "prism_gpus=${PRISM_GPUS[*]}"
    echo "placement=$PLACEMENT"
    echo "time_scale=$TIME_SCALE"
    echo "replication=$REPLICATION"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "kvcached_gpu_utilization=$KVCACHED_GPU_UTILIZATION"
    echo "sample_interval=0.25"
    echo "swap=$([[ "$SWAP" == 1 ]] && echo dram || echo none)"
    [[ "$SWAP" == 1 ]] && echo "swap_headroom_gib=$HEADROOM_GIB"
} > "$RUN_DIR/config.txt"

MONITOR_PID= PROXY_PID=
cleanup() {
    [[ -n "$PROXY_PID" ]] && kill "$PROXY_PID" 2> /dev/null
    [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2> /dev/null && wait "$MONITOR_PID" 2> /dev/null
    bash "$ROOT/scripts/stop_servers.sh" "$RUN_DIR"
    "$PY" "$ROOT/analysis/prism_run_report.py" "$RUN_DIR" || true
}
trap cleanup EXIT

"$PY" "$ROOT/scripts/vram_monitor.py" "$RUN_DIR" --interval 0.25 &
MONITOR_PID=$!

echo "launch_start=$(date +%s.%N)" >> "$RUN_DIR/config.txt"
if ! SLEEP_AFTER_START=$SWAP CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL:-0} \
    bash "$ROOT/scripts/launch_servers.sh" "$MODE" "$RUN_DIR"; then
    echo "launch_failed=1" >> "$RUN_DIR/config.txt"
    echo "launch failed; keeping data in $RUN_DIR" >&2
    exit 1
fi
echo "launch_end=$(date +%s.%N)" >> "$RUN_DIR/config.txt"
if [[ -s "$RUN_DIR/failed.tsv" ]]; then
    echo "failed_instances=$(cut -f1 "$RUN_DIR/failed.tsv" | paste -sd,)" >> "$RUN_DIR/config.txt"
fi

replay_args=()
if [[ "$SWAP" == 1 ]]; then
    "$PY" "$ROOT/scripts/swap_proxy.py" "$RUN_DIR" --port "$PROXY_PORT" --headroom-gib "$HEADROOM_GIB" \
        > "$RUN_DIR/proxy.log" 2>&1 &
    PROXY_PID=$!
    until curl -sf "http://localhost:$PROXY_PORT/health" > /dev/null; do
        kill -0 "$PROXY_PID" 2> /dev/null || { echo "swap proxy exited; see proxy.log" >&2; exit 1; }
        sleep 1
    done
    replay_args=(--proxy "http://localhost:$PROXY_PORT")
fi

sleep 10  # idle baseline with every model loaded
"$PY" "$ROOT/scripts/replay_prism_trace.py" "$RUN_DIR" \
    --time-scale "$TIME_SCALE" --replication "$REPLICATION" "${replay_args[@]}" | tee "$RUN_DIR/replay.log"
echo "workload_end=$(date +%s.%N)" >> "$RUN_DIR/config.txt"
sleep 10  # observe memory after traffic
echo "done: $RUN_DIR"
