# Evidence — axial-layout scoping

The full evidence package is published in the same private repository on
[`evidence/2026-09-15-b23aca5d`](https://github.com/Shakil-Muhit/axial-layout-scoping/tree/evidence/2026-09-15-b23aca5d). This keeps the handoff's main-branch rule that only
`evidence/*.md` is versioned, while making all raw artifacts available remotely.

Evidence commit: `21faad925f5e9bd351e225d82006f6144ee87a27`.
[Immutable artifact manifest](https://github.com/Shakil-Muhit/axial-layout-scoping/blob/21faad925f5e9bd351e225d82006f6144ee87a27/ARTIFACTS.json).

Generation: `b23aca5d-3315-42c4-832f-8668c8b36ceb`. The archive contains **948 files**,
including both Nsight traces, SQLite/CSV exports, generated code, saved validation
tensors, clock logs, runtime manifests, and independent audits. Its compressed
size is **25763899 bytes**.

Archive SHA256: `478cc56b5ee5ff4fcc962f92db0cc33fdc07d2765702848e952ee799b4dd054c`.
Per-part and per-file hashes are in `ARTIFACTS.json` on the evidence branch.

## Restore after cloning main

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

## Audit entry points

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
