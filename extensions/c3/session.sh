#!/bin/bash
# Bounded C3 extension on the existing ml-beast environment. Source the
# existing clock/export helpers; override RUNS before any evidence writer.
set -euo pipefail
. "$(dirname "$0")/../../scripts/beast_env.sh"
RUNS="$1"
export AXIAL_MANIFEST="$RUNS/manifest.json"
export PYTHONPATH="$REPO/scoping:$REPO/extensions/c3"
export TORCH_LOGS=graph_breaks,recompiles
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCH_COMPILE_DEBUG=1
mkdir -p "$RUNS"
exec 9> "$AX/study_session.lock"
flock -n 9 || { echo 'Study lock is already held'; exit 1; }
finish_extension() {
    local rc=$?
    trap - EXIT
    clocks_stop || true
    clocks_snapshot "$RUNS/clocks_post.txt"
    date -Is > "$RUNS/finished.txt"
    date +%s > "$RUNS/finished_unix.txt"
    echo "$rc" > "$RUNS/exit.txt"
    git -C "$MSST_DIR" status --porcelain > "$RUNS/public_clone_status_post.txt"
    exit "$rc"
}
trap finish_extension EXIT
date -Is > "$RUNS/started.txt"
date +%s > "$RUNS/started_unix.txt"
printf '%s\n' "${C3_FIRST_STARTED_UNIX:-$(cat "$RUNS/started_unix.txt")}" \
    > "$RUNS/extension_first_started_unix.txt"
gpu4_must_be_free
clocks_snapshot "$RUNS/clocks_pre.txt"
"$V/python" "$REPO/extensions/c3/manifest.py" create \
    --repo "$REPO" --out "$RUNS" --c1 "$AX/runs"
clocks_start "$RUNS/clocks.csv"
mkdir -p "$RUNS/qualification/dump" "$RUNS/phase1/dump"
"$V/python" -u "$REPO/extensions/c3/driver.py" --mode eager-check \
    --out "$RUNS/eager_check" --refs "$RUNS/c1_reference/phase0" \
    > "$RUNS/eager_check.log" 2>&1
export TORCH_COMPILE_DEBUG_DIR="$RUNS/qualification/dump"
"$V/python" -u "$REPO/extensions/c3/driver.py" --mode qualify \
    --out "$RUNS/qualification" --refs "$RUNS/c1_reference/phase0" \
    > "$RUNS/qualification/driver.log" 2>&1
echo 'C3 qualification and unprofiled timing completed'
"$V/python" "$REPO/extensions/c3/manifest.py" check \
    --repo "$REPO" --out "$RUNS" --c1 "$AX/runs"
export TORCH_COMPILE_DEBUG_DIR="$RUNS/phase1/dump"
"$NSYS" profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node -s none --cpuctxsw=none --force-overwrite=true \
    -o "$RUNS/phase1/trace_c3" \
    "$V/python" -u "$REPO/extensions/c3/driver.py" --mode trace \
    --out "$RUNS/phase1" --refs "$RUNS/c1_reference/phase0" \
    > "$RUNS/phase1/driver.log" 2>&1
nsys_export "$RUNS/phase1/trace_c3.nsys-rep" "$RUNS/phase1/trace_c3"
echo 'C3 node-level trace and required exports completed'
TORCH_COMPILE_DEBUG_DIR="$RUNS/phase1/microbench_dump" \
    "$V/python" -u "$REPO/scoping/microbench_p_l2.py" \
    --out "$RUNS/phase1/microbench_bwref.json" \
    > "$RUNS/phase1/microbench.log" 2>&1
"$V/python" "$REPO/scoping/classify.py" \
    "$RUNS/phase1/trace_c3_cuda_gpu_kern_sum.csv" "$RUNS/phase1/classifier_sample.csv"
"$V/python" "$REPO/scoping/inventory.py" "$RUNS/phase1/dump" \
    "$RUNS/phase1/inventory_c3.json"
"$V/python" "$REPO/scoping/kernel_map.py" --output-code-dir "$RUNS/phase1/dump" \
    --gpu-trace-csv "$RUNS/phase1/trace_c3_cuda_gpu_trace.csv" \
    --kern-sum-csv "$RUNS/phase1/trace_c3_cuda_gpu_kern_sum.csv" \
    --export-status "$RUNS/phase1/trace_c3_exports.json" \
    --nvtx-csv "$RUNS/phase1/trace_c3_nvtx_gpu_proj_trace.csv" \
    --trace-meta "$RUNS/phase1/trace_run_meta.json" \
    --baseline-timing "$RUNS/qualification/timing.json" \
    --bwref "$RUNS/phase1/microbench_bwref.json" \
    --out-dir "$RUNS/phase1" --tag c3 \
    > "$RUNS/phase1/kernel_map.log" 2>&1
"$V/python" "$REPO/extensions/c3/manifest.py" check \
    --repo "$REPO" --out "$RUNS" --c1 "$AX/runs"
echo 'C3 Phase 1 collection completed; independent attribution audit still required'
