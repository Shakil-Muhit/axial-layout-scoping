# Evidence — axial-layout scoping

The C3 extension and original C1 study have separate lossless packages in
this private repository. This keeps the handoff's main-branch rule that
only `evidence/*.md` is versioned. Restore both packages to resolve all raw
artifact links in [RESULT.md](../RESULT.md).

## C3 extension

Branch: [`evidence/2026-09-15-c3-b12c7049`](https://github.com/Shakil-Muhit/axial-layout-scoping/tree/evidence/2026-09-15-c3-b12c7049).
Evidence commit: `78a3b3bbe37fd87eea719d7b2cc519bb523d6bfa`.
[Immutable artifact manifest](https://github.com/Shakil-Muhit/axial-layout-scoping/blob/78a3b3bbe37fd87eea719d7b2cc519bb523d6bfa/ARTIFACTS.json).

Generation: `b12c7049-f5a5-4f06-881b-59c87fdd9fa4`. The archive contains
**836 files**, including all failed attempts, saved validation tensors,
generated code, the Nsight trace and exports, timing and clock samples,
copy attribution, ten calibration cases, and independent audits. Compressed
size: **118257418 bytes**, split into three parts.

Archive SHA256: `6244bbb467918d75521d4038295fb320c25a37d10c9b1a03abc94890b16c60fe`.
Per-part and per-file hashes are in `ARTIFACTS.json`.

### Restore C3 after cloning main

Run from the repository root:

```bash
git fetch origin evidence/2026-09-15-c3-b12c7049
AXIAL_C3_RESTORE=$(mktemp -d /tmp/axial-c3.XXXXXX)
git archive 78a3b3bbe37fd87eea719d7b2cc519bb523d6bfa | tar -x -C "$AXIAL_C3_RESTORE"
(cd "$AXIAL_C3_RESTORE" && sha256sum -c SHA256SUMS)
cat "$AXIAL_C3_RESTORE"/artifacts.tar.gz.part* > "$AXIAL_C3_RESTORE/artifacts.tar.gz"
printf '%s  %s\n' '6244bbb467918d75521d4038295fb320c25a37d10c9b1a03abc94890b16c60fe' "$AXIAL_C3_RESTORE/artifacts.tar.gz" | sha256sum -c -
mkdir -p evidence/extensions/c3
tar -xzf "$AXIAL_C3_RESTORE/artifacts.tar.gz" -C evidence/extensions/c3
```

The archive restores `evidence/extensions/c3/raw/` and preserves artifact
mtimes. Packaging verified every archived file against its source bytes.
A fresh clone of the published evidence commit was then independently
restored: all 836 file hashes, all three part hashes, and the combined archive
hash passed on 2026-09-15.

### C3 audit entry points

- [Baseline and trace audit](extensions/c3/raw/independent_audit.json): saved-output
  comparisons, zero graph breaks, all five timing blocks, clocks, and trace totals.
- [Active graph selection](extensions/c3/raw/active_graph_selection.json): exact
  ordered Triton inventory matching across all captured passes.
- [Copy attribution](extensions/c3/raw/phase1/layout_attribution_audited.json):
  source-proven head-merge copies and why A alone cannot establish NO-GO.
- [Reference refinement](extensions/c3/raw/matched_calibration/audit.json): ten
  cases, their tight failures, and incomplete B coverage.
- [Provenance audit](extensions/c3/raw/provenance/audit_record.json): code hashes,
  original-evidence integrity, checks, and the complete GPU time window.

## Original C1 study

The original package is published on
[`evidence/2026-09-15-b23aca5d`](https://github.com/Shakil-Muhit/axial-layout-scoping/tree/evidence/2026-09-15-b23aca5d).

Evidence commit: `21faad925f5e9bd351e225d82006f6144ee87a27`.
[Immutable artifact manifest](https://github.com/Shakil-Muhit/axial-layout-scoping/blob/21faad925f5e9bd351e225d82006f6144ee87a27/ARTIFACTS.json).

Generation: `b23aca5d-3315-42c4-832f-8668c8b36ceb`. The archive contains **948 files**,
including both Nsight traces, SQLite/CSV exports, generated code, saved validation
tensors, clock logs, runtime manifests, and independent audits. Its compressed
size is **25763899 bytes**.

Archive SHA256: `478cc56b5ee5ff4fcc962f92db0cc33fdc07d2765702848e952ee799b4dd054c`.
Per-part and per-file hashes are in `ARTIFACTS.json` on the evidence branch.

### Restore C1 after cloning main

Run from the repository root:

```bash
git fetch origin evidence/2026-09-15-b23aca5d
AXIAL_EVIDENCE_RESTORE=$(mktemp -d /tmp/axial-evidence.XXXXXX)
mkdir -p evidence
git archive FETCH_HEAD | tar -x -C "$AXIAL_EVIDENCE_RESTORE"
(cd "$AXIAL_EVIDENCE_RESTORE" && sha256sum -c SHA256SUMS)
cat "$AXIAL_EVIDENCE_RESTORE"/artifacts.tar.gz.part* > "$AXIAL_EVIDENCE_RESTORE/artifacts.tar.gz"
printf '%s  %s\n' '478cc56b5ee5ff4fcc962f92db0cc33fdc07d2765702848e952ee799b4dd054c' "$AXIAL_EVIDENCE_RESTORE/artifacts.tar.gz" | sha256sum -c -
tar -xzf "$AXIAL_EVIDENCE_RESTORE/artifacts.tar.gz" -C evidence
```

The archive restores `evidence/raw/` and the relative handoff-name symlinks.
It preserves original artifact mtimes. `ARTIFACTS.json` inventories every file;
packaging verified each archived file's bytes against its source hash.
A fresh fetch of the published evidence commit was then extracted independently:
all 948 file hashes, the archive hash, and the handoff-name symlinks passed.

### C1 audit entry points

- [RESULT.md](../RESULT.md): fixed-format decision inputs and limitations.
- [Independent baseline audit](raw/independent_audit.json): saved-output validation,
  all five timing medians, spread, throughput and clocks recomputed from samples.
- [Attribution audit](raw/phase1/attribution_audit.json): event/pass counts,
  aggregate cross-check, and why the partial map cannot yield R_layout.
- [NVTX-hook equivalence audit](raw/hook_diagnostic/equivalence_audit.json):
  the supplemental annotation trace fails both required equivalence checks.
- [Provenance audit](raw/provenance/audit_record.json): pre-registration ordering,
  code hashes, source integrity, checks performed and study time window.

No Phase-2 result is claimed. The final report records the unresolved outcome.
