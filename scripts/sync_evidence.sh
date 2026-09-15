#!/bin/bash
# Run on the local git host. Preserve partial/failure artifacts and full
# generated-code/trace evidence. No successful phase marker is required.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EV="$REPO/evidence"
mkdir -p "$EV/raw"
rsync -a --exclude='*_archive_*' ml-beast:axial/runs/ "$EV/raw/"
# Handoff names are relative symlinks to the complete raw collection.
for f in baseline_timing.json validation_baseline.json env.json; do
    if [ -e "$EV/raw/phase0/$f" ]; then ln -sfn "raw/phase0/$f" "$EV/$f"; fi
done
if [ -d "$EV/raw/phase1/dump_baseline" ]; then
    ln -sfn raw/phase1/dump_baseline "$EV/output_code_baseline"
elif [ -d "$EV/raw/phase0/dump_baseline" ]; then
    ln -sfn raw/phase0/dump_baseline "$EV/output_code_baseline"
fi
if [ -f "$EV/raw/phase1/inductor_kernel_map_baseline.csv" ]; then
    ln -sfn raw/phase1/inductor_kernel_map_baseline.csv "$EV/inductor_kernel_map.csv"
fi
if [ -d "$EV/raw/phase2/dump_patched" ]; then
    ln -sfn raw/phase2/dump_patched "$EV/output_code_patched"
fi
echo SYNC-DONE
