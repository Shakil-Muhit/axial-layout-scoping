# Bounded C3 Phase 1 evidence

Generation: `b12c7049-f5a5-4f06-881b-59c87fdd9fa4`, 2026-09-15.

The archive preserves all 836 files from the C3 extension: failed attempts,
saved validation outputs, generated code, timing samples, clock logs,
Nsight trace and exports, source-based copy attribution, calibration cases,
and independent audits. GPU execution used 2622 seconds of the 3600-second
extension window.

C3 reaches 30.3055275 ms with zero graph breaks. Its required loose
comparison against eager passes; tight comparison against C1 fails on all
three seeds. Verified layout copies contribute 0.2688275 ms/pass, an A-only
share of 0.887058%. Full R_layout remains unresolved because validated
hidden-cost reference coverage is incomplete. Phase 2 was not run.

Measuring code: `700c2c6e0d6351d032d9003972b34092127dfc2e`.
Calibration code: `a4ab60751bb07f987a0f0643763d505c2395cfb8`.
Prediction registration: `9b1fefdefb4bc9962f2be4f867e77e1905b627df`.
See the [report on main](https://github.com/Shakil-Muhit/axial-layout-scoping/blob/main/RESULT.md)
for the complete scope, failures, and limitations.

Archive size: **118257418 bytes**.
Archive SHA256:
`6244bbb467918d75521d4038295fb320c25a37d10c9b1a03abc94890b16c60fe`.
`ARTIFACTS.json` records every file hash and the ordered archive parts.
Packaging checked every archived file against its source bytes.

## Restore

After fetching this branch into a clone of main, extract its tracked files
into a separate temporary directory. Verify `SHA256SUMS`, concatenate
`artifacts.tar.gz.part*` in name order, and verify the archive hash above.
Extract the archive into **`evidence/extensions/c3/`**; it contains `raw/`.
Full commands are in [evidence/README.md on main](https://github.com/Shakil-Muhit/axial-layout-scoping/blob/main/evidence/README.md).

The original C1 package remains on `evidence/2026-09-15-b23aca5d`.
Its 948 file hashes were rechecked unchanged during this extension.
