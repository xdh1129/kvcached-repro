#!/bin/bash
# Launch the vLLM instances listed in <run_dir>/instances.tsv, one after another.
#   usage: launch_servers.sh <kvcached|static> <run_dir>
# instances.tsv: "name<TAB>model<TAB>gpu<TAB>port" per line.
# Each instance runs in its own process group. Written to <run_dir>:
#   servers.tsv   "name<TAB>port<TAB>pgid", for the VRAM monitor and stop_servers.sh
#   startup.tsv   per instance: seconds to healthy and GPU memory once healthy
# SLEEP_AFTER_START=1 enables vLLM sleep mode and puts each instance to sleep
# (level 1: weights backed up to CPU DRAM) right after it is healthy, so the next
# one has room; startup.tsv then also records memory while asleep.
# CONTINUE_ON_FAIL=1 records an instance that fails to start in failed.tsv
# ("name<TAB>gpu<TAB>port<TAB>seconds") and goes on with the rest; otherwise the
# first failure stops the launch.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/configs/experiment.env"
source "$ROOT/.venv/bin/activate"
source "$ROOT/scripts/cuda_env.sh"

MODE=${1:?usage: launch_servers.sh <kvcached|static> <run_dir>}
RUN_DIR=${2:?usage: launch_servers.sh <kvcached|static> <run_dir>}
: > "$RUN_DIR/servers.tsv"
printf 'name\tgpu\tport\tstartup_s\tgpu_used_mib\tinstance_mib\tfirst_sleep_s\tinstance_asleep_mib\n' \
    > "$RUN_DIR/startup.tsv"

export VLLM_USE_V1=1
SLEEP_AFTER_START=${SLEEP_AFTER_START:-0}
CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL:-0}
: > "$RUN_DIR/failed.tsv"

common_args=(--no-enable-prefix-caching --max-model-len "$MAX_MODEL_LEN")
if [[ "$SLEEP_AFTER_START" == 1 ]]; then
    export VLLM_SERVER_DEV_MODE=1  # exposes /sleep and /wake_up
    common_args+=(--enable-sleep-mode)
fi

case "$MODE" in
    kvcached)
        export ENABLE_KVCACHED=true
        export KVCACHED_AUTOPATCH=1
        export KVCACHED_GPU_UTILIZATION KVCACHED_MIN_RESERVED_PAGES KVCACHED_MAX_RESERVED_PAGES
        export KVCACHED_PAGE_PREALLOC_ENABLED=true
        export KVCACHED_SANITY_CHECK=false
        mode_args=()
        ;;
    static)
        unset ENABLE_KVCACHED KVCACHED_AUTOPATCH
        mode_args=(--gpu-memory-utilization "$STATIC_GPU_MEM_UTIL")
        ;;
    *)
        echo "unknown mode: $MODE (expected kvcached|static)" >&2
        exit 1
        ;;
esac

wait_healthy() {
    local port=$1 pid=$2 deadline=$((SECONDS + 900))
    until curl -sf "http://localhost:$port/health" > /dev/null; do
        if ! kill -0 "$pid" 2> /dev/null; then
            echo "instance on port $port exited; see its log" >&2
            return 1
        fi
        if ((SECONDS > deadline)); then
            echo "instance on port $port not healthy after 900s" >&2
            return 1
        fi
        sleep 2
    done
}

# GPU memory held by the processes in one process group, in MiB.
pgid_mib() {
    local pgid=$1 total=0 pid mib
    while IFS=', ' read -r pid mib; do
        [[ "$(ps -o pgid= -p "$pid" 2> /dev/null | tr -d ' ')" == "$pgid" ]] && total=$((total + mib))
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    echo "$total"
}

while IFS=$'\t' read -r name model gpu port; do
    log="$RUN_DIR/server_${name}.log"
    echo "[$MODE] starting $name ($model) on GPU $gpu port $port -> $log"
    start=$SECONDS
    CUDA_VISIBLE_DEVICES=$gpu KVCACHED_IPC_NAME="kvc_${name}_${port}" setsid vllm serve "$model" \
        --host localhost --port "$port" \
        "${common_args[@]}" "${mode_args[@]}" > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\t%s\t%s\n' "$name" "$port" "$pid" >> "$RUN_DIR/servers.tsv"
    if ! wait_healthy "$port" "$pid"; then
        [[ "$CONTINUE_ON_FAIL" == 1 ]] || exit 1
        kill -TERM -- "-$pid" 2> /dev/null || true
        printf '%s\t%s\t%s\t%s\n' "$name" "$gpu" "$port" "$((SECONDS - start))" >> "$RUN_DIR/failed.tsv"
        echo "[$MODE] $name failed to start; continuing (CONTINUE_ON_FAIL=1)"
        sleep 5  # let the failed process group release its memory
        continue
    fi
    startup_s=$((SECONDS - start))
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    awake_mib=$(pgid_mib "$pid")
    echo "[$MODE] $name healthy after ${startup_s}s; GPU $gpu used ${used} MiB"
    sleep_s="" asleep_mib=""
    if [[ "$SLEEP_AFTER_START" == 1 ]]; then
        t0=$(date +%s.%N)
        curl -sf -X POST "http://localhost:$port/sleep?level=1" > /dev/null
        sleep_s=$(echo "$(date +%s.%N) - $t0" | bc)
        asleep_mib=$(pgid_mib "$pid")
        echo "[$MODE] $name asleep after ${sleep_s}s; holds ${asleep_mib} MiB (awake ${awake_mib} MiB)"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$gpu" "$port" "$startup_s" "$used" \
        "$awake_mib" "$sleep_s" "$asleep_mib" >> "$RUN_DIR/startup.tsv"
done < "$RUN_DIR/instances.tsv"
