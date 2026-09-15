#!/bin/bash
# Execute registered phases once. The launcher applies the four-hour
# timeout to this process group; dependent phases stop on failure.
set -u
. "$(dirname "$0")/beast_env.sh"
failed=""
for phase in 0 1 2; do
    bash "$REPO/scripts/run_phase${phase}.sh"
    rc=$?
    if [ "$rc" != "0" ]; then
        failed="phase $phase exited $rc"
        break
    fi
done
clocks_stop
if [ -n "$failed" ]; then
    "$V/python" "$REPO/scoping/finalize.py" \
        --phase0-dir "$RUNS/phase0" --phase1-dir "$RUNS/phase1" \
        --phase2-dir "$RUNS/phase2" --failed-step "$failed" \
        --out "$RUNS/summary.json" || exit 1
    echo "SESSION-STOPPED: $failed"
    exit 1
fi
echo SESSION-COMPLETE
