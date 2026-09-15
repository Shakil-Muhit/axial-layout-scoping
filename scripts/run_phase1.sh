#!/bin/bash
# Phase 1 ON ml-beast (handoff section 4): node-level nsys trace of the 5a
# latency block + output_code dump (SAME process, so the wrapper<->trace
# cross-reference is exact) -> stats exports -> classifier sample ->
# P-L2 microbench -> kernel map / materialization inventory / R_layout.
# Invoke:  ssh ml-beast 'bash /mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_phase1.sh'
set -u
. "$(dirname "$0")/beast_env.sh"
P0="$RUNS/phase0"
OUT="$RUNS/phase1"
check_manifest
[ -s "$RUNS/phase0.done" ] || { echo "FATAL: phase 0 did not pass"; exit 1; }
fresh_phase "$OUT"
rm -f "$RUNS/phase1.done"
LOG="$RUNS/logs/phase1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner "PHASE 1"
[ -s "$P0/baseline_timing.json" ] || { echo "FATAL: run phase 0 first"; exit 1; }
check_manifest
gpu4_must_be_free
clocks_snapshot "$OUT/clocks_pre.txt"
clocks_start "$OUT/clocks_sampled_phase1.csv"

echo "--- [P1a] traced 5a block + output_code dump (one process) ---"
rm -rf "$OUT/dump_baseline"
mkdir -p "$OUT/dump_baseline"
TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$OUT/dump_baseline" \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 \
TORCH_LOGS=recompiles \
"$NSYS" profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node -s none --cpuctxsw=none --force-overwrite=true \
    -o "$OUT/trace_baseline" \
    "$V/python" "$REPO/scoping/baseline_run.py" --mode trace-iters \
    --out-dir "$OUT" || exit 1
[ -s "$OUT/trace_baseline.nsys-rep" ] || { echo "FATAL: no .nsys-rep"; exit 1; }
DUMPS=$(find "$OUT/dump_baseline" -name output_code.py | wc -l)
echo "output_code.py files dumped: $DUMPS"
[ "$DUMPS" -ge 1 ] || { echo "FATAL: no output_code dumped"; exit 1; }

echo "--- [P1a] nsys stats exports ---"
nsys_export "$OUT/trace_baseline.nsys-rep" "$OUT/trace_baseline" || exit 1
TRACE_CSV="$OUT/trace_baseline_cuda_gpu_trace.csv"
KSUM_CSV="$OUT/trace_baseline_cuda_gpu_kern_sum.csv"
[ -s "$TRACE_CSV" ] && [ -s "$KSUM_CSV" ] || {
    echo "FATAL: expected export CSVs missing; dir listing:"; ls -la "$OUT"; exit 1; }
NROWS=$(wc -l < "$TRACE_CSV")
echo "gpu-trace rows: $NROWS"
# kernels-per-pass floor (sm120 lesson): 50 passes of this model must show
# thousands of launches; ~50 rows means graph replays were not expanded.
[ "$NROWS" -ge 1000 ] || { echo "FATAL: $NROWS rows — cuda-graph-trace=node broken?"; exit 1; }

echo "--- [P1b] classifier sample (owner spot-check) ---"
"$V/python" "$REPO/scoping/classify.py" "$KSUM_CSV" \
    "$OUT/classifier_sample.csv" || exit 1

echo "--- [P1e/B] P-L2 microbench: BW_ref + permuted penalty ---"
"$V/python" "$REPO/scoping/microbench_p_l2.py" \
    --out "$OUT/microbench_bwref.json" || exit 1

echo "--- [P1c/1d/1e] kernel map -> materialization inventory -> R_layout ---"
NVTX_CSV="$OUT/trace_baseline_nvtx_gpu_proj_trace.csv"
NVTX_ARG=""
[ -s "$NVTX_CSV" ] && NVTX_ARG="--nvtx-csv $NVTX_CSV"
"$V/python" "$REPO/scoping/kernel_map.py" \
    --output-code-dir "$OUT/dump_baseline" \
    --gpu-trace-csv "$TRACE_CSV" \
    --kern-sum-csv "$KSUM_CSV" \
    --export-status "$OUT/trace_baseline_exports.json" \
    $NVTX_ARG \
    --trace-meta "$OUT/trace_run_meta.json" \
    --baseline-timing "$P0/baseline_timing.json" \
    --bwref "$OUT/microbench_bwref.json" \
    --out-dir "$OUT" --tag baseline || exit 1

"$V/python" "$REPO/scoping/inventory.py" "$OUT/dump_baseline" \
    "$OUT/inventory_baseline.json" || exit 1

clocks_stop
clocks_snapshot "$OUT/clocks_post.txt"
date -Is > "$RUNS/phase1.done"
echo "PHASE1-DONE"
