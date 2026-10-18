#!/bin/bash
# Stop every instance recorded in <run_dir>/servers.tsv.
#   usage: stop_servers.sh <run_dir>
set -uo pipefail

RUN_DIR=${1:?usage: stop_servers.sh <run_dir>}
[[ -f "$RUN_DIR/servers.tsv" ]] || exit 0

while IFS=$'\t' read -r name port pgid; do
    kill -TERM -- "-$pgid" 2> /dev/null && echo "stopping $name (pgid $pgid)"
done < "$RUN_DIR/servers.tsv"

for _ in $(seq 30); do
    alive=0
    while IFS=$'\t' read -r _ _ pgid; do
        kill -0 -- "-$pgid" 2> /dev/null && alive=1
    done < "$RUN_DIR/servers.tsv"
    ((alive)) || exit 0
    sleep 1
done

while IFS=$'\t' read -r name _ pgid; do
    kill -KILL -- "-$pgid" 2> /dev/null && echo "killed $name (pgid $pgid)"
done < "$RUN_DIR/servers.tsv"
exit 0
