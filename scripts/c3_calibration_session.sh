#!/bin/bash
# Supplemental Phase-1 class calibration only; original C3 deadline applies.
set -euo pipefail
AXIAL_C3_BASE="$HOME/axial/extensions/c3_20260915_dc"
AXIAL_C3_REPO=/mnt/Muhit/Home/ms/axial-layout-scoping
AXIAL_C3_FIRST=$(cat "$HOME/axial/extensions/c3_20260915/started_unix.txt")
test "$(date +%s)" -lt "$((AXIAL_C3_FIRST+3600))"
exec 9> "$HOME/axial/study_session.lock"
flock -n 9
export CUDA_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1
export AXIAL_MANIFEST="$AXIAL_C3_BASE/manifest.json"
export TORCH_LOGS=graph_breaks,recompiles TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
mkdir -p "$AXIAL_C3_BASE/matched_calibration"
date +%s > "$AXIAL_C3_BASE/matched_calibration/started_unix.txt"
nvidia-smi -i 4 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw --format=csv,noheader -l 1 > "$AXIAL_C3_BASE/matched_calibration/clocks.csv" &
AXIAL_C3_CLOCK_PID=$!
trap 'kill "$AXIAL_C3_CLOCK_PID" 2>/dev/null || true; date +%s > "$AXIAL_C3_BASE/matched_calibration/finished_unix.txt"' EXIT
for AXIAL_C3_CASE in norm:time:single_half norm:time:single_float norm:freq:single_float norm:time:double_half norm:time:double_float norm:freq:double_float rotary:time:standard rotary:freq:standard gate:time:standard gate:freq:standard; do
    test "$(date +%s)" -lt "$((AXIAL_C3_FIRST+3600))"
    AXIAL_C3_OUT="$AXIAL_C3_BASE/matched_calibration/${AXIAL_C3_CASE//:/_}"
    mkdir -p "$AXIAL_C3_OUT/dump"
    export TORCH_COMPILE_DEBUG_DIR="$AXIAL_C3_OUT/dump"
    if "$HOME/axial/venv/bin/python" -u "$AXIAL_C3_REPO/scripts/c3_matched_microbench.py" --case "$AXIAL_C3_CASE" --out "$AXIAL_C3_OUT" > "$AXIAL_C3_OUT/driver.log" 2>&1; then
        printf '0\n' > "$AXIAL_C3_OUT/exit.txt"
    else
        printf '%s\n' "$?" > "$AXIAL_C3_OUT/exit.txt"
    fi
done
