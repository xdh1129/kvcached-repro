#!/bin/bash
# Correctness check of the tiered sleep backend (tiered_restore plugin) for each path config.
#   usage: check_tiered_restore.sh [config ...]     config = <weights>+<kv>[:seq], e.g. dram+gds
# Default: the five configs of the planned experiment, all with parallel restore.
# Per config: starts one kvcached vLLM instance with sleep mode, runs
# check_tiered_restore.py (reference / restore / control), stops it.
# Env: GPU, PORT, CHECK_MODEL, CHECK_MAX_LEN (8192), CHECK_PARAGRAPHS (110),
#      CHECK_SLEEP_AT (2.0 s; earlier for small models that finish before it), CHECK_TAG (results under tiered_check/<tag>/)
# Results: results/tiered_check/[<tag>/]<config>/{check.json,tiered.jsonl,server.log}
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/configs/experiment.env"
source "$ROOT/.venv/bin/activate"
source "$ROOT/scripts/cuda_env.sh"

GPU=${GPU:-3}
PORT=${PORT:-31101}
MODEL=${CHECK_MODEL:-meta-llama/Meta-Llama-3.1-8B}
DATA_ROOT=${DATA_ROOT:-/mnt/raid0/xdh/tiered}
CONFIGS=("$@")
((${#CONFIGS[@]})) || CONFIGS=(dram+dram ssd+ssd gds+gds dram+ssd dram+gds)

export VLLM_USE_V1=1 VLLM_SERVER_DEV_MODE=1
export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_PAGE_PREALLOC_ENABLED=true KVCACHED_SANITY_CHECK=false
export KVCACHED_GPU_UTILIZATION KVCACHED_MIN_RESERVED_PAGES KVCACHED_MAX_RESERVED_PAGES

for cfg in "${CONFIGS[@]}"; do
    paths=${cfg%%:*}
    out="$ROOT/results/tiered_check/${CHECK_TAG:+$CHECK_TAG/}$cfg"
    mkdir -p "$out"
    rm -f "$out/tiered.jsonl"
    dir="$DATA_ROOT/check_${CHECK_TAG:+${CHECK_TAG//\//_}_}${paths/+/_}"
    rm -rf "$dir" && mkdir -p "$dir"
    while curl -sf "http://localhost:$PORT/health" > /dev/null; do sleep 2; done  # previous server gone
    echo "== $cfg: $MODEL on GPU $GPU, files in $dir"
    TIERED_WEIGHTS=${paths%+*} TIERED_KV=${paths#*+} TIERED_DIR=$dir TIERED_LOG="$out/tiered.jsonl" \
    TIERED_VERIFY=1 TIERED_PARALLEL=$([[ "$cfg" == *:seq ]] && echo 0 || echo 1) \
    CUDA_VISIBLE_DEVICES=$GPU KVCACHED_IPC_NAME="kvc_check_$PORT" setsid vllm serve "$MODEL" \
        --host localhost --port "$PORT" --no-enable-prefix-caching --max-model-len "${CHECK_MAX_LEN:-8192}" \
        --enable-sleep-mode > "$out/server.log" 2>&1 < /dev/null &
    pid=$!
    trap 'kill -TERM -- -$pid 2>/dev/null || true' EXIT
    until curl -sf "http://localhost:$PORT/health" > /dev/null; do
        kill -0 "$pid" 2> /dev/null || { echo "server exited; see $out/server.log" >&2; exit 1; }
        sleep 2
    done
    python "$ROOT/scripts/check_tiered_restore.py" --port "$PORT" --gpu "$GPU" --tiered-dir "$dir" \
        --out "$out/check.json" --control --sleep-at "${CHECK_SLEEP_AT:-2.0}"
    python - "$out/tiered.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["event"] == "resume":
        print(f"  resume {r['n']}: skip_kv={r['debug_skip_kv_load']} verify={r.get('verify')}")
PY
    kill -TERM -- "-$pid" 2> /dev/null || true
    wait "$pid" 2> /dev/null || true
    trap - EXIT
    rm -rf "$dir"
    sleep 5
done
