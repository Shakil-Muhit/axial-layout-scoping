#!/bin/bash
# Phase 2 ON ml-beast (handoff section 5) — gated on R_layout >= 0.03:
#   [P2.K]  P-L4 knob sweep on the UNPATCHED baseline (each its own process,
#           own output_code dump, TIGHT validation vs baseline outputs,
#           inventory diff; 25-min timeout -> UNTESTED). A knob whose
#           inventory CHANGES gets a follow-up nsys trace + kernel_map so its
#           (A)+(B) reduction is measured before any flag_recovers claim.
#   [P2.1a] mechanical transcription check of the patched forward
#   [P2.1]  CPU fp32 parity of the patched forward (fails -> stop)
#   [P2.2]  patched C1 run: settle (fail-closed), TIGHT validation vs Phase-0
#           baseline outputs, 5a five-repeats
#   [P2.3]  patched trace + output_code dump; re-run of 1a-1d (tag patched)
#   [P2.4]  finalize: gated R_recovered + flag_recovers -> summary.json
# The threshold-skip path still runs finalize so the summary/RESULT inputs
# exist (gate-02 finding 15).
# Invoke:  ssh ml-beast 'bash /mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_phase2.sh'
set -u
. "$(dirname "$0")/beast_env.sh"
P0="$RUNS/phase0"
P1="$RUNS/phase1"
OUT="$RUNS/phase2"
check_manifest
fresh_phase "$OUT"
rm -f "$RUNS/phase2.done" "$RUNS/phase2.skipped" "$RUNS/phase2.failed"
LOG="$RUNS/logs/phase2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner "PHASE 2"
[ -s "$P1/r_layout_baseline.json" ] || { echo "FATAL: run phase 1 first"; exit 1; }
check_manifest
gpu4_must_be_free
clocks_snapshot "$OUT/clocks_pre.txt"

RL=$("$V/python" -c "import json;print(json.load(open('$P1/r_layout_baseline.json'))['R_layout_vs_clean'])")
echo "R_layout (clean) = $RL"
MAP_OK=$("$V/python" -c "import json;d=json.load(open('$P1/r_layout_baseline.json'));print(int(d['status']=='ok' and d['B_status']=='ok'))")
if [ "$MAP_OK" != "1" ] && [ "$("$V/python" -c "print(int(float('$RL') < 0.03))")" = "1" ]; then
    "$V/python" "$REPO/scoping/finalize.py" --phase0-dir "$P0" --phase1-dir "$P1" \
        --failed-step "attribution incomplete: lower bound does not establish Phase-2 threshold" \
        --out "$RUNS/summary.json"
    date -Is > "$RUNS/phase2.failed"
    exit 1
fi
if [ "$("$V/python" -c "print(1 if float('$RL') >= 0.03 else 0)")" != "1" ]; then
    echo "R_layout < 0.03 -> NO-GO by threshold (handoff section 4); Phase 2 skipped."
    "$V/python" "$REPO/scoping/finalize.py" --phase0-dir "$P0" \
        --phase1-dir "$P1" --out "$RUNS/summary.json" || exit 1
    date -Is > "$RUNS/phase2.skipped"
    echo "PHASE2-SKIPPED"
    exit 0
fi

clocks_start "$OUT/clocks_sampled_phase2.csv"

knob_status() {   # $1 = knob_result.json -> status string or MISSING
    if [ -s "$1" ]; then
        "$V/python" -c "import json;print(json.load(open('$1')).get('status','MISSING'))" 2>/dev/null || echo MISSING
    else
        echo MISSING
    fi
}

finalize_and_fail() {   # $1 = failed step — failures still produce summary.json
    echo "PHASE2-FAILED at: $1"
    clocks_stop
    "$V/python" "$REPO/scoping/finalize.py" --phase0-dir "$P0" --phase1-dir "$P1" \
        --phase2-dir "$OUT" --failed-step "$1" \
        --clocks-baseline "$P0/clocks_sampled_phase0.csv" \
        --clocks-patched "$OUT/clocks_sampled_phase2.csv" \
        --out "$RUNS/summary.json" || true
    date -Is > "$RUNS/phase2.failed"
    echo "PHASE2-FAILED"
    exit 1
}

KNOBS="layout_opt_true layout_opt_false keep_output_stride_false \
permute_fusion_true persistent_reductions_false max_autotune \
max_autotune_no_cudagraphs"

echo "--- [P2.K] P-L4 knob sweep (unpatched baseline) ---"
for K in $KNOBS; do
    KD="$OUT/knob_$K"
    mkdir -p "$KD"
    ST=$(knob_status "$KD/knob_result.json")
    if [ "$ST" = "MISSING" ]; then
        rm -rf "$KD/dump"
        echo "[knob $K] running (timeout 25 min)..."
        TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$KD/dump" \
        TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 \
        timeout 1500 "$V/python" "$REPO/scoping/knob_run.py" \
            --knob "$K" --out-dir "$KD" --refs-dir "$P0"
        RC=$?
        if [ "$RC" = "124" ]; then
            printf '{"knob":"%s","status":"UNTESTED: compile/settle exceeded the 25-min timeout"}\n' "$K" > "$KD/knob_result.json"
            echo "[knob $K] TIMEOUT -> UNTESTED"
        elif [ "$RC" != "0" ]; then
            printf '{"knob":"%s","status":"FAILED: exit %s (see phase2 log)"}\n' "$K" "$RC" > "$KD/knob_result.json"
            echo "[knob $K] FAILED (exit $RC)"
        fi
        ST=$(knob_status "$KD/knob_result.json")
    else
        echo "[knob $K] recorded ($ST) — not rerunning"
    fi
    if [ "$ST" = "ok" ] && [ ! -s "$KD/inventory.json" ] && [ -d "$KD/dump" ] \
       && [ "$(find "$KD/dump" -name output_code.py | wc -l)" -ge 1 ]; then
        "$V/python" "$REPO/scoping/inventory.py" "$KD/dump" "$KD/inventory.json" || true
    fi
done

echo "--- [P2.K+] follow-up traces for validation-eligible knobs ---"
for K in $KNOBS; do
    KD="$OUT/knob_$K"
    [ "$(knob_status "$KD/knob_result.json")" = "ok" ] || continue
    [ -s "$KD/inventory.json" ] || continue
    TIGHT=$("$V/python" -c "import json; d=json.load(open('$KD/knob_result.json')); print(d.get('validation',{}).get('tight_all_pass'))")
    if [ "$TIGHT" = "False" ]; then
        echo "[knob $K] explicitly failed tight validation; disqualified"
        continue
    fi
    if [ -s "$KD/r_layout_knob_$K.json" ]; then
        echo "[knob $K] follow-up map already present"
        continue
    fi
    echo "[knob $K] inventory CHANGED -> follow-up trace + kernel map"
    # fresh dump from the TRACE process itself (gate-03 f27: the screening
    # dump belongs to a different compilation), stale artifacts cleared
    rm -rf "$KD/dump_trace" "$KD"/trace_knob* "$KD/trace_run_meta_knob.json" \
           "$KD/knob_trace_result.json"
    mkdir -p "$KD/dump_trace"
    TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$KD/dump_trace" \
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 \
    "$NSYS" profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node -s none --cpuctxsw=none --force-overwrite=true \
        -o "$KD/trace_knob" \
        timeout 2400 "$V/python" "$REPO/scoping/knob_run.py" \
        --knob "$K" --mode trace --out-dir "$KD" --refs-dir "$P0" || {
            echo "[knob $K] follow-up trace FAILED (flag verdict stays UNTESTED)"; continue; }
    TST=$("$V/python" -c "import json;print(json.load(open('$KD/knob_trace_result.json')).get('status','MISSING'))" 2>/dev/null || echo MISSING)
    if [ "$TST" != "ok" ]; then
        echo "[knob $K] trace result status=$TST (flag verdict stays UNTESTED)"
        continue
    fi
    nsys_export "$KD/trace_knob.nsys-rep" "$KD/trace_knob" || continue
    KNVTX="$KD/trace_knob_nvtx_gpu_proj_trace.csv"
    KNARG=""
    [ -s "$KNVTX" ] && KNARG="--nvtx-csv $KNVTX"
    "$V/python" "$REPO/scoping/kernel_map.py" \
        --output-code-dir "$KD/dump_trace" \
        --gpu-trace-csv "$KD/trace_knob_cuda_gpu_trace.csv" \
        --kern-sum-csv "$KD/trace_knob_cuda_gpu_kern_sum.csv" \
        --export-status "$KD/trace_knob_exports.json" \
        $KNARG \
        --trace-meta "$KD/trace_run_meta_knob.json" \
        --baseline-timing "$P0/baseline_timing.json" \
        --bwref "$P1/microbench_bwref.json" \
        --out-dir "$KD" --tag "knob_$K" || \
        echo "[knob $K] kernel map FAILED (flag verdict stays UNTESTED)"
    # the traced compile's own inventory replaces the screening one for the diff
    "$V/python" "$REPO/scoping/inventory.py" "$KD/dump_trace" "$KD/inventory.json" || true
done

echo "--- [P2.1a] transcription check (patched forward vs pinned source) ---"
"$V/python" "$REPO/scoping/verify_transcription.py" \
    --out "$OUT/transcription_check.json" || finalize_and_fail "transcription check"

echo "--- [P2.1] CPU fp32 parity (patched vs stock eager) ---"
"$V/python" "$REPO/scoping/patched_run.py" --mode cpu-parity \
    --out-dir "$OUT" || finalize_and_fail "cpu parity"

echo "--- [P2.2] patched C1: settle + TIGHT validation + 5a timing ---"
TORCH_LOGS=recompiles "$V/python" "$REPO/scoping/patched_run.py" --mode gpu \
    --out-dir "$OUT" --baseline-dir "$P0" || finalize_and_fail "patched gpu run"

echo "--- [P2.3] patched trace + output_code dump ---"
rm -rf "$OUT/dump_patched"
mkdir -p "$OUT/dump_patched"
TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$OUT/dump_patched" \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 \
TORCH_LOGS=recompiles \
"$NSYS" profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node -s none --cpuctxsw=none --force-overwrite=true \
    -o "$OUT/trace_patched" \
    "$V/python" "$REPO/scoping/patched_run.py" --mode trace-iters \
    --out-dir "$OUT" || finalize_and_fail "patched trace"
nsys_export "$OUT/trace_patched.nsys-rep" "$OUT/trace_patched" || finalize_and_fail "patched exports"
PTRACE_CSV="$OUT/trace_patched_cuda_gpu_trace.csv"
PKSUM_CSV="$OUT/trace_patched_cuda_gpu_kern_sum.csv"
[ -s "$PTRACE_CSV" ] && [ -s "$PKSUM_CSV" ] || finalize_and_fail "patched export csvs missing"
NROWS=$(wc -l < "$PTRACE_CSV")
echo "patched gpu-trace rows: $NROWS"
[ "$NROWS" -ge 1000 ] || finalize_and_fail "patched trace rows $NROWS < 1000 (graph tracing broken?)"

PNVTX="$OUT/trace_patched_nvtx_gpu_proj_trace.csv"
PNARG=""
[ -s "$PNVTX" ] && PNARG="--nvtx-csv $PNVTX"
"$V/python" "$REPO/scoping/kernel_map.py" \
    --output-code-dir "$OUT/dump_patched" \
    --gpu-trace-csv "$PTRACE_CSV" \
    --kern-sum-csv "$PKSUM_CSV" \
    --export-status "$OUT/trace_patched_exports.json" \
    $PNARG \
    --trace-meta "$OUT/trace_run_meta_patched.json" \
    --baseline-timing "$OUT/patched_timing.json" \
    --bwref "$P1/microbench_bwref.json" \
    --out-dir "$OUT" --tag patched || finalize_and_fail "patched kernel map"
"$V/python" "$REPO/scoping/inventory.py" "$OUT/dump_patched" \
    "$OUT/inventory_patched.json" || finalize_and_fail "patched inventory"

echo "--- [P2.4] finalize: gated R_recovered + flag_recovers -> summary.json ---"
"$V/python" "$REPO/scoping/finalize.py" --phase0-dir "$P0" --phase1-dir "$P1" \
    --phase2-dir "$OUT" \
    --clocks-baseline "$P0/clocks_sampled_phase0.csv" \
    --clocks-patched "$OUT/clocks_sampled_phase2.csv" \
    --out "$RUNS/summary.json" || exit 1

clocks_stop
clocks_snapshot "$OUT/clocks_post.txt"
date -Is > "$RUNS/phase2.done"
echo "PHASE2-DONE"
