# Codex continuation — pre-execution corrections

Date: 2026-09-15. Continued directly by Codex at the owner's request. No Claude
process or delegated reviewer is used for this continuation.

The inherited tree stopped during gate-04 finding 32. No GPU measurement had
started. The earlier review's seven blockers were addressed before execution:

| Finding | Correction | Verification |
|---|---|---|
| 32: incomplete memory-access analysis | Parse complete kernel pointer expressions with Python AST; multiple distinct accesses and ambiguous stores remain unresolved. Store-only `in_out_ptr` is not charged as a reader. Inventory shares these rules. | Parser fixture and access regression cases. |
| 33: invalid reference bandwidth | Price actual input/output dtypes and auxiliary vectors; exclude failed CUDA-graph timing and unmatched boundary dtypes. | Source inspection; actual dtype/timing records remain required on GPU. |
| 34: premature head-merge verdict | Follow the SDPA output through gating to projection with reaching-writer checks; intervening clones reject the view verdict. Unknown paths remain unresolved. | Gate → clone → projection regression. |
| 35: unsupported compiler-flag verdict | Neither equal inventory counts nor missing validation can establish `False`. Validation-eligible flags require complete follow-up attribution and valid baseline evidence. | Finalizer regressions for unchanged counts, flagged baseline, and missing validation. |
| 36: evidence rebinding | Every baseline creates a fresh generation, archives older runs, and records configuration contents plus code/model/checkpoint hashes. JSON readers reject other generations. No drift override remains. | Generation mismatch regression; on-host provenance preflight. |
| 37: stale patched map after failure | Archive each phase attempt before execution; pass the failed step into an explicit finalizer gate. | Stale-map failure regression. |
| 38: incomplete required exports | Clear old exports, require nonempty GPU trace, kernel summary, NVTX trace and summary; save export status and require relevant kernel-summary coverage. | Shell/source inspection; actual export checks run with each trace. |

Additional corrections:

- Initialize Phase-0 output/log paths after archival (finding 39).
- Save environment and generated code during Phase 0 so a validation failure
  still has inspectable provenance. Debug generation occurs during compilation;
  only settled execution is eligible for timing. The compile call stays C1.
- Unknown B contributions are not a valid lower bound: an approximate reference
  can overestimate them. Any flagged map uses A only for its reported lower
  bound. A bound below 3% cannot establish the Phase-2 skip condition.
- Enforce exact validation seeds and tolerances when accepting results.
- Preserve failure evidence during synchronization and produce a summary when
  a dependent phase cannot run.

## Limits that remain explicit

Per-sublayer ownership uses wrapper order and SDPA anchors, not inner NVTX
hooks, because such hooks can change Dynamo's graphs. This is a declared
deviation from handoff 1a. The CSV states that ownership is ordinal. Parsing and
reference-chain coverage must be checked against real generated artifacts;
offline fixtures do not prove complete attribution for an unseen graph.

The representative bandwidth microbenchmarks are class-based. A missing or
incompatible reference leaves B unresolved; it must never be silently priced
from a different class or promoted to a full R_layout estimate.

The final study outcome must follow actual validation and evidence gates. This
document is a pre-execution review record and makes no performance claim.
