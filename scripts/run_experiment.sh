#!/bin/bash
# One end-to-end run: launch instances, sample VRAM, drive the workload, tear down.
#   usage: run_experiment.sh <kvcached|static> [run_name]
# Output: results/<run_name>/{instances.tsv,servers.tsv,startup.tsv,vram.csv,metrics/*.json,*.log}
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/configs/experiment.env"

MODE=${1:?usage: run_experiment.sh <kvcached|static> [run_name]}
RUN_NAME=${2:-${MODE}_$(date +%Y%m%d_%H%M%S)}
RUN_DIR="$ROOT/results/$RUN_NAME"
mkdir -p "$RUN_DIR"

{
    echo "mode=$MODE"
    echo "git=$(git -C "$ROOT" rev-parse --short HEAD 2> /dev/null || echo none)"
    echo "kvcached=$(git -C "$ROOT/third_party/kvcached" describe --tags --always)"
    grep -E '^[A-Z_]+=' "$ROOT/configs/experiment.env" | cut -d= -f1 | while read -r v; do
        declare -n ref=$v
        echo "$v=${ref[*]}"
    done
} > "$RUN_DIR/config.txt"

MONITOR_PID=
cleanup() {
    [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2> /dev/null
    bash "$ROOT/scripts/stop_servers.sh" "$RUN_DIR"
}
trap cleanup EXIT

for i in "${!PORTS[@]}"; do
    printf 'inst%s\t%s\t%s\t%s\n' "$((i + 1))" "$MODEL" "$GPU_ID" "${PORTS[$i]}"
done > "$RUN_DIR/instances.tsv"

bash "$ROOT/scripts/launch_servers.sh" "$MODE" "$RUN_DIR"

"$ROOT/.venv/bin/python" "$ROOT/scripts/vram_monitor.py" "$RUN_DIR" --interval "$SAMPLE_INTERVAL" &
MONITOR_PID=$!
sleep 5  # idle baseline before traffic

echo "workload_start=$(date +%s.%N)" >> "$RUN_DIR/config.txt"
bash "$ROOT/scripts/run_workload.sh" "$RUN_DIR"
echo "workload_end=$(date +%s.%N)" >> "$RUN_DIR/config.txt"

sleep 10  # observe memory release after traffic
echo "done: $RUN_DIR"
