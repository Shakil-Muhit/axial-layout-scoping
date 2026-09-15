# Shared execution environment for study runs ON ml-beast.
# Sourced by scripts/run_phase{0,1,2}.sh (invoked as `bash <mount path>` —
# the repo mount is sshfs/noexec). Everything heavy lives on LOCAL disk
# (~/axial, set up by scripts/beast_bootstrap.sh).
AX="$HOME/axial"
V="$AX/venv/bin"
REPO="/mnt/Muhit/Home/ms/axial-layout-scoping"
RUNS="$AX/runs"
NSYS="$(cat "$AX/nsys_path.txt")"

export CUDA_VISIBLE_DEVICES=4        # study GPU; GPU 0 carries an unrelated
                                     # external workload (docs/design_notes.md)
export PYTHONDONTWRITEBYTECODE=1     # no .pyc litter on the sshfs mount
export MSST_DIR="$AX/msst"
export CKPT_FILE="$AX/ckpt/MelBandRoformer.ckpt"
export CONFIG_FILE="$AX/msst/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"
export NSYS_PATH="$NSYS"

mkdir -p "$RUNS/logs"

banner() {
    echo "=== $1 | $(date -Is) | repo @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'n/a') | GPU$CUDA_VISIBLE_DEVICES ==="
}

gpu4_must_be_free() {
    local n
    n=$(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "$n" = "0" ] || { echo "FATAL: GPU 4 has $n compute app(s) running"; exit 1; }
}

clocks_snapshot() {   # $1 = out file (point-in-time; the SAMPLER is the audit)
    {
        date -Is
        nvidia-smi -i 4 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv
        echo "-- all GPUs (external load on GPU 0 documented) --"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
    } > "$1" 2>&1 || true
}

# Sustained background clock log (handoff rule 9, verbatim query set;
# gate-02 finding 13). finalize.py cross-references the rows with the timed
# blocks' unix_start/unix_end.
CLK_PID=""
clocks_start() {   # $1 = out csv
    nvidia-smi -i 4 \
        --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
        --format=csv,noheader -l 1 > "$1" 2>/dev/null &
    CLK_PID=$!
    echo "clock sampler -> $1 (pid $CLK_PID)"
}
clocks_stop() {
    [ -n "$CLK_PID" ] && kill "$CLK_PID" 2>/dev/null
    CLK_PID=""
}
trap clocks_stop EXIT

# Every baseline attempt starts a fresh generation. All downstream JSON
# writers/readers bind to this manifest; config CONTENTS are hashed too.
export AXIAL_MANIFEST="$RUNS/manifest.json"
write_manifest() {
    "$V/python" "$REPO/scoping/run_manifest.py" create --repo "$REPO" --runs "$RUNS"
}
check_manifest() {
    "$V/python" "$REPO/scoping/run_manifest.py" check --repo "$REPO" --runs "$RUNS" || exit 1
}

# Preserve attempt artifacts before retrying; no stale map may survive.
fresh_phase() {
    if [ -d "$1" ]; then
        mv "$1" "${1}_archive_$(date +%Y%m%d_%H%M%S)_$$" || exit 1
    fi
    mkdir -p "$1" "$RUNS/logs"
}

nsys_export() {   # $1 = report path, $2 = export prefix
    local report rc=0
    for report in cuda_gpu_trace cuda_gpu_kern_sum nvtx_gpu_proj_trace nvtx_gpu_proj_sum; do
        rm -f "${2}_${report}.csv"
        "$NSYS" stats --force-export=true --format csv --force-overwrite=true \
            --report "$report" --output "$2" "$1" || rc=1
        [ -s "${2}_${report}.csv" ] || rc=1
    done
    PYTHONPATH="$REPO/scoping" "$V/python" - "$2" "$rc" <<'PYEXPORT'
import csv, sys
from evidence_io import write_json
prefix, rc = sys.argv[1:]
reports = {}
for name in ('cuda_gpu_trace', 'cuda_gpu_kern_sum', 'nvtx_gpu_proj_trace', 'nvtx_gpu_proj_sum'):
    try:
        with open(f'{prefix}_{name}.csv', newline='') as f:
            rows = list(csv.reader(f))
        reports[name] = len(rows) > 1 and bool(rows[0]) and any(rows[1:])
    except (OSError, csv.Error):
        reports[name] = False
ok = rc == '0' and all(reports.values())
write_json(prefix + '_exports.json', {'status': 'ok' if ok else 'FAILED', 'reports': reports})
if not ok:
    raise SystemExit('FATAL: required nsys exports are missing or empty')
PYEXPORT
}
