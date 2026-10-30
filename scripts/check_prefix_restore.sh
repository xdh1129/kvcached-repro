#!/bin/bash
# Prefix-cache survival check for each offload path: one instance per path on one GPU,
# prefix caching on, scripts/check_prefix_restore.py against it.
#   usage: check_prefix_restore.sh [paths ...]   (default: dram+dram dram+gds vllm)
#   <paths>:skipkv is a negative control: the plugin's DEBUG_SKIP_KV_LOAD marker makes the
#   wake skip the KV load, so the "restored" step must show wrong logprobs.
# Env: GPU (default 1), MODEL (default meta-llama/Llama-3.2-1B), HISTORY (6000),
#      TIERED_GDS_STAGE_MB etc. pass through. KVCACHED_MAX_CACHED_TOKENS is set to -1
#      (unlimited) so kvcached keeps the whole history cached.
# Output: results/check_prefix_restore/<paths>/ (check.json + server log)
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/.venv/bin/python"
GPU=${GPU:-1}
MODEL=${MODEL:-meta-llama/Llama-3.2-1B}
HISTORY=${HISTORY:-6000}
PORT=${PORT:-31901}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export KVCACHED_MAX_CACHED_TOKENS=${KVCACHED_MAX_CACHED_TOKENS:--1}
export PREFIX_CACHING=1
PATHS_LIST=("$@")
((${#PATHS_LIST[@]})) || PATHS_LIST=(dram+dram dram+gds vllm)

used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if ((used > 1024)); then
    echo "GPU $GPU already has ${used} MiB in use" >&2
    exit 1
fi

for p in "${PATHS_LIST[@]}"; do
    dir="$ROOT/results/check_prefix_restore/$p"
    mkdir -p "$dir"
    printf 'agent\t%s\t%s\t%s\n' "$MODEL" "$GPU" "$PORT" > "$dir/instances.tsv"
    (
        paths=${p%:skipkv}
        if [[ "$paths" == vllm ]]; then
            unset TIERED_WEIGHTS TIERED_KV TIERED_ROOT
        else
            export TIERED_WEIGHTS=${paths%+*} TIERED_KV=${paths#*+} TIERED_PARALLEL=1
            export TIERED_ROOT=/mnt/raid0/xdh/tiered/check_prefix_restore_${p//:/_}
        fi
        trap 'bash "$ROOT/scripts/stop_servers.sh" "$dir"; [[ -n "${TIERED_ROOT:-}" ]] && rm -rf "$TIERED_ROOT"' EXIT
        SLEEP_AFTER_START=1 bash "$ROOT/scripts/launch_servers.sh" kvcached "$dir" || exit 1
        [[ "$p" == *:skipkv ]] && touch "$TIERED_ROOT/agent/DEBUG_SKIP_KV_LOAD"
        curl -sf -X POST "http://localhost:$PORT/wake_up" > /dev/null  # launch leaves it asleep
        "$PY" "$ROOT/scripts/check_prefix_restore.py" --port "$PORT" --model "$MODEL" --history "$HISTORY" \
            | tee "$dir/check.json"
    )
    echo "=== $p exit $?"
    sleep 5
done
