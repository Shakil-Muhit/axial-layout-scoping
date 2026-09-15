#!/bin/bash
# Phase 0 ON ml-beast (handoff section 2): eager refs -> C1 baseline
# (settle, LOOSE validation, 5a five-repeats) -> env.json.
# Invoke:  ssh ml-beast 'bash /mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_phase0.sh'
set -u
. "$(dirname "$0")/beast_env.sh"
banner "PHASE 0 preflight"
gpu4_must_be_free
write_manifest || exit 1
OUT="$RUNS/phase0"
mkdir -p "$OUT" "$RUNS/logs"
LOG="$RUNS/logs/phase0_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner "PHASE 0"
clocks_snapshot "$OUT/clocks_pre.txt"
clocks_start "$OUT/clocks_sampled_phase0.csv"

echo "--- [P0.1] eager reference outputs (3 seeds) ---"
"$V/python" "$REPO/scoping/baseline_run.py" --mode eager-ref \
    --out-dir "$OUT" || exit 1

echo "--- [P0.2] C1 baseline: settle + loose validation + 5a timing + env ---"
TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$OUT/dump_baseline" \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 \
TORCH_LOGS=recompiles "$V/python" "$REPO/scoping/baseline_run.py" \
    --mode baseline --out-dir "$OUT" || exit 1

clocks_stop
clocks_snapshot "$OUT/clocks_post.txt"
for f in baseline_timing.json validation_baseline.json env.json; do
    [ -s "$OUT/$f" ] || { echo "FATAL: missing $OUT/$f"; exit 1; }
done
date -Is > "$RUNS/phase0.done"
echo "PHASE0-DONE"
