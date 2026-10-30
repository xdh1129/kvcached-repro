#!/bin/bash
# Interleaved multi-turn agents on several GPUs (scripts/agent_rr.py): one conversation per
# model, requests round-robin across the models of a GPU, LRU swapping of weights + cached
# history KV through scripts/swap_proxy.py --keep. Prefix caching on, so a restored agent
# re-uses its history instead of re-prefilling it.
#   usage: run_agent_rr.sh [run_name]
# Env: GPUS        GPUs to use (default "6 7"; they must be idle)
#      POOL        comma-separated HF repos; COPIES instances of each per GPU (default 2)
#                  (default: four 3B-8B models with >= 128k context)
#      PATHS       <weights>+<kv> for the tiered_restore plugin (default dram+gds), or vllm
#                  (vLLM's sleep mode: the cached history is dropped and re-prefilled)
#      TURNS (24), FIRST_LEN (2048), TURN_LEN (1024), OUT_LEN (300)   see agent_rr.py
#      SYNC=0      let the GPUs run independently (default 1: steps in lockstep)
#      HEADROOM_GIB  swap proxy free-memory margin after a wake (default 8). kvcached maps KV
#                  pages only out of NVML free memory minus (1 - KVCACHED_GPU_UTILIZATION) of
#                  the GPU (~4.8 GiB at 0.95), so the margin must cover that plus the turn's
#                  KV growth; with 3 a woken agent's request waited on "capacity" forever.
#      MAX_MODEL_LEN (default 40960: covers 24 turns of the default lengths)
#      BASE_PORT   first port (default 33001)
#      TIERED_*    plugin options pass through (e.g. TIERED_GDS_STAGE_MB=256)
# Output: results/<run_name>/ (report: analysis/agent_rr_report.py)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
source "$ROOT/configs/experiment.env"
PY="$ROOT/.venv/bin/python"

GPUS=(${GPUS:-6 7})
POOL=${POOL:-meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct,meta-llama/Meta-Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct}
COPIES=${COPIES:-2}
PATHS=${PATHS:-dram+gds}
TURNS=${TURNS:-24}
FIRST_LEN=${FIRST_LEN:-2048}
TURN_LEN=${TURN_LEN:-1024}
OUT_LEN=${OUT_LEN:-300}
SYNC=${SYNC:-1}
HEADROOM_GIB=${HEADROOM_GIB:-8}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-600}
BASE_PORT=${BASE_PORT:-33001}
PROXY_PORT=${PROXY_PORT:-33999}
RUN_NAME=${1:-agent_rr_$(date +%Y%m%d_%H%M%S)}
RUN_DIR="$ROOT/results/$RUN_NAME"
export PREFIX_CACHING=1
export KVCACHED_MAX_CACHED_TOKENS=-1  # kvcached keeps every cached history block (default: 16k tokens)

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

# agents a1..aN per GPU, the pool repeated COPIES times (a1=pool[0], a2=pool[1], ...)
IFS=, read -ra pool <<< "$POOL"
port=$BASE_PORT
: > "$RUN_DIR/instances.tsv"
for g in "${GPUS[@]}"; do
    i=1
    for _ in $(seq "$COPIES"); do
        for repo in "${pool[@]}"; do
            printf 'g%s_a%s\t%s\t%s\t%s\n' "$g" "$i" "$repo" "$g" "$port" >> "$RUN_DIR/instances.tsv"
            port=$((port + 1)) i=$((i + 1))
        done
    done
done

if [[ "$PATHS" == vllm ]]; then
    unset TIERED_WEIGHTS TIERED_KV TIERED_ROOT  # cleanup removes TIERED_ROOT: never one from the caller
else
    export TIERED_WEIGHTS=${PATHS%+*} TIERED_KV=${PATHS#*+}
    export TIERED_PARALLEL=${TIERED_PARALLEL:-1}
    export TIERED_ROOT=/mnt/raid0/xdh/tiered/$RUN_NAME
fi

{
    echo "experiment=agent_rr"
    echo "date=$(date -Is)"
    echo "repo_git=$(git -C "$ROOT" rev-parse --short HEAD 2> /dev/null || echo none)"
    echo "kvcached_git=$(git -C "$ROOT/third_party/kvcached" describe --tags --always)"
    "$PY" -c "import vllm, torch; print(f'vllm={vllm.__version__}'); print(f'torch={torch.__version__}')"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "${GPUS[0]}")"
    echo "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPUS[0]}")"
    echo "gpus=${GPUS[*]}"
    echo "pool=$POOL"
    echo "copies=$COPIES"
    echo "paths=$PATHS"
    [[ "$PATHS" != vllm ]] && echo "tiered_parallel=$TIERED_PARALLEL" && echo "tiered_root=$TIERED_ROOT"
    env | grep -E '^TIERED_(WEIGHTS_SAVE|KV_MERGE|WEIGHTS_PACE|DRAM_KV_RESERVE_GB|GDS_GATE|GDS_STAGE_MB|GDS_STAGE_BUFS)=' \
        | sed 's/^/tiered_opt_/' || true
    echo "turns=$TURNS"
    echo "first_len=$FIRST_LEN"
    echo "turn_len=$TURN_LEN"
    echo "out_len=$OUT_LEN"
    echo "sync=$SYNC"
    echo "swap_headroom_gib=$HEADROOM_GIB"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "prefix_caching=1"
    echo "kvcached_max_cached_tokens=$KVCACHED_MAX_CACHED_TOKENS"
    echo "kvcached_gpu_utilization=$KVCACHED_GPU_UTILIZATION"
} > "$RUN_DIR/config.txt"

VRAM_PID= PCIE_PID= PROXY_PID=
cleanup() {
    set +e  # under set -e, a SIGTERM'd child's wait (status 143) would abort the cleanup
    for p in $PROXY_PID $VRAM_PID $PCIE_PID; do kill "$p" 2> /dev/null; wait "$p" 2> /dev/null; done
    bash "$ROOT/scripts/stop_servers.sh" "$RUN_DIR" || true
    [[ -n "${TIERED_ROOT:-}" ]] && rm -rf "$TIERED_ROOT"
    "$PY" "$ROOT/analysis/agent_rr_report.py" "$RUN_DIR" || true
}
trap cleanup EXIT

"$PY" "$ROOT/scripts/vram_monitor.py" "$RUN_DIR" --interval 0.5 &
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

"$PY" "$ROOT/scripts/swap_proxy.py" "$RUN_DIR" --port "$PROXY_PORT" --headroom-gib "$HEADROOM_GIB" --keep \
    > "$RUN_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
until curl -sf "http://localhost:$PROXY_PORT/health" > /dev/null; do
    kill -0 "$PROXY_PID" 2> /dev/null || { echo "swap proxy exited; see proxy.log" >&2; exit 1; }
    sleep 1
done

client_args=(--proxy "http://localhost:$PROXY_PORT" --turns "$TURNS" --first-len "$FIRST_LEN"
             --turn-len "$TURN_LEN" --out-len "$OUT_LEN" --request-timeout "$REQUEST_TIMEOUT_S")
[[ "$SYNC" == 1 ]] || client_args+=(--no-sync)
"$PY" "$ROOT/scripts/agent_rr.py" "$RUN_DIR" "${client_args[@]}" | tee "$RUN_DIR/agent_rr.log"
sleep 3
echo "done: $RUN_DIR"
