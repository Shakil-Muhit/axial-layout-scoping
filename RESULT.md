# Axial-layout scoping — RESULT

**C3 meets the latency prediction: 30.305527 ms with zero graph breaks. Full layout attribution remains UNRESOLVED.** The verified head-merge copies account for an A-only share of **0.887058%**. Missing B cannot establish NO-GO. [Timing and independent audit](evidence/extensions/c3/raw/independent_audit.json), [attribution](evidence/extensions/c3/raw/phase1/layout_attribution_audited.json)

This report includes the owner-authorized bounded C3 extension. The original
C1 study and all its evidence are preserved. Raw links below resolve after
[restoring the evidence packages](evidence/README.md).

## Arm, provenance, and scope

- Measuring code: `700c2c6e0d6351d032d9003972b34092127dfc2e`; calibration code:
  `a4ab60751bb07f987a0f0643763d505c2395cfb8`. Generation:
  `b12c7049-f5a5-4f06-881b-59c87fdd9fa4`. [Manifest](evidence/extensions/c3/raw/manifest.json)
- sm120-nulltest tooling: `a997e27d843badb3e742f4e57f3896ac61681431`;
  public model: `ea7eb9c20ea0e3f94368a30fc1654b51cdd55789`.
  Checkpoint SHA256:
  `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`.
  [Manifest](evidence/extensions/c3/raw/manifest.json)
- Torch 2.8.0+cu128, Triton 3.4.0, driver 595.84, RTX 5090 GPU 4;
  `use_amp=true`. Timed SM clock median **2640 MHz**, range **2617–2745 MHz**;
  memory **13801 MHz**. Other GPUs were busy. [Environment](evidence/extensions/c3/raw/qualification/env.json),
  [clock audit](evidence/extensions/c3/raw/independent_audit.json),
  [initial load](evidence/extensions/c3/raw/clocks_pre.txt)

C3 uses the original full model, weights, inputs, precision, and
`torch.compile(model, mode='reduce-overhead', dynamic=False)` call. Two external
compatibility changes are required in this pinned environment:

1. Replace `_asdict` and the unsupported legacy SDPA context with precomputed
   backend lists and `torch.nn.attention.sdpa_kernel`. Preserve every backend
   flag, including the legacy default enabling cuDNN. [Source/backend checks](evidence/extensions/c3/raw/qualification/source_verification.json)
2. Keep the original complex DC-filter `Tensor.index_fill` call opaque to the
   compiler. An isolated reproducer establishes an AOT functional-graph
   assertion in its decomposition. The adapter invokes that native operation;
   it adds no GPU kernel or arithmetic rewrite. The public forward differs
   only at that callee, and iSTFT remains the public call.
   [Reproducer](evidence/extensions/c3/raw/provenance/dc_fill_probe.log),
   [mechanical comparison](evidence/extensions/c3/raw/qualification/native_tail_verification.json),
   [operator/schema/fake-tensor/CUDA-graph checks](evidence/extensions/c3/raw/eager_check/native_tail_opcheck.json)

The attention-only repair was insufficient. This result is for **attention
repair plus native DC-filter boundary**. Earlier failed attempts, including
the rejected iSTFT-boundary diagnosis, are retained and explained in
[execution notes](extensions/c3/ATTEMPT_NOTES.md).

All GPU work, including reference refinement, finished **2622 seconds** after
the extension began, within its **3600-second** limit. The public clone and
sm120-nulltest remain clean. [Time audit](evidence/extensions/c3/raw/matched_calibration/audit.json),
[final state](evidence/extensions/c3/raw/provenance/post_extension.txt)

## Decision inputs (mechanical)

| Input | C3 result | Evidence |
|---|---|---|
| Full R_layout | UNRESOLVED | [Attribution](evidence/extensions/c3/raw/phase1/layout_attribution_audited.json) |
| Source-proven A contribution | 0.2688275 ms/pass; A-only share 0.0088705765 | [Copy ledger](evidence/extensions/c3/raw/phase1/inductor_kernel_map_c3_audited.csv) |
| B | UNRESOLVED: incomplete validated reference coverage | [Generic-reference coverage](evidence/extensions/c3/raw/phase1/bandwidth_reference_coverage.csv), [refinement audit](evidence/extensions/c3/raw/matched_calibration/audit.json) |
| R_recovered | UNTESTED: Phase-1-only extension | [Registered scope](PREDICTIONS_EXTENSION_C3.md) |
| flag_recovers | UNTESTED: no Phase-2 sweep | [Registered scope](PREDICTIONS_EXTENSION_C3.md) |
| Head split | No separate pure split copy; q/k are RoPE outputs, v is a QKV view | [Generated code](evidence/extensions/c3/raw/phase1/active_dump/output_code.py), [call metadata](evidence/extensions/c3/raw/phase1/call_metadata_c3.json) |
| Head merge | Materialized: one pure copy per attention call | [Per-attention copy inventory](evidence/extensions/c3/raw/phase1/layout_attribution_audited.json) |

A is time spent in verified layout-copy kernels. B is the extra time caused
by permuted reads inside kernels that also compute, measured against
validated contiguous references. Full R_layout requires both.

The original thresholds remain unchanged: NO-GO below 0.03; TICKET in
[0.03, 0.05), or on the specified Phase-2 findings; GO requires R_layout ≥0.05,
R_recovered ≥0.60, and flag_recovers=False. **No threshold decision is
established.** An A-only value below 0.03 does not bound the missing B from
above. Phase 2 was not run. [Protocol](docs/handoff.md)

## Timing and validation

Five 5a blocks, each with 50 synchronized latency passes and 50 free-running
passes, use the original protocol. [Complete timing](evidence/extensions/c3/raw/qualification/timing.json)

| Quantity | Result |
|---|---|
| Latency medians, ms | 30.3055275, 30.3727295, 30.4944545, 30.4826515, 30.6935820 |
| Minimum median | 30.3055275 ms |
| Median spread | 0.3880545 ms |
| Best throughput | 29.6987211 ms/pass |
| Historical C1 minimum median | 35.4032750 ms |
| Historical latency reduction | 5.0977475 ms; 14.399085% |

The arithmetic and clock windows were recomputed independently.
The comparison is historical; load and clocks differ, and the extra DC-filter
boundary is part of C3. It does not isolate how much of the speedup comes
from staging copies. [Independent audit](evidence/extensions/c3/raw/independent_audit.json)

The external eager model matches stock **bitwise on all three seeds**.
C3 passes the required loose tier against stock eager. As authorized by the
owner, tight disagreement with C1 is reported separately and does not stop
Phase 1. [Eager check](evidence/extensions/c3/raw/eager_check/validation_external_eager.json),
[loose validation](evidence/extensions/c3/raw/qualification/validation_vs_eager.json),
[tight comparison](evidence/extensions/c3/raw/qualification/validation_vs_c1.json)

| Seed | C3 vs eager, loose: max_abs_diff | C3 vs C1, tight: max_abs_diff |
|---|---|---|
| 4242 | PASS: 7.32939224690e-06 | FAIL: 1.90411228687e-05 |
| 1337 | PASS: 2.79112718999e-05 | FAIL: 2.14488245547e-05 |
| 90210 | PASS: 5.93266449869e-05 | FAIL: 3.66256572306e-05 |

An additional comparison of traced C3 against timed C3 fails tight tolerance
on seeds 4242 and 90210, while both runs pass the registered loose baseline
gate. Their generated call bodies and all 232 Triton function bodies match;
runtime-selected tuning or library algorithms are not proved identical.
No tight equivalence is claimed. [Independent comparisons](evidence/extensions/c3/raw/independent_audit.json),
[program comparison](evidence/extensions/c3/raw/program_comparison.json)

## Attribution and materializations

Both timed and traced processes record **zero graph breaks**, no unimplemented
operations, no suppressed compiler errors, and stable measurement counters.
A rotary-cache guard causes one warmup recompile, leaving two dumps. Only
one graph's complete ordered Triton inventory matches every captured pass.
[Graph/counter audit](evidence/extensions/c3/raw/independent_audit.json),
[graph selection](evidence/extensions/c3/raw/active_graph_selection.json)

The trace contains **50 passes, 1181 GPU events/pass, 791 Triton calls/pass,
and 12 Flash Attention calls/pass**. All required exports succeeded and
kernel-summary totals match the per-event trace. The selected wrapper has
all 12 SDPA anchors. [Audit](evidence/extensions/c3/raw/independent_audit.json),
[all kernel classes](evidence/extensions/c3/raw/phase1/all_kernel_classes.csv),
[Nsight report](evidence/extensions/c3/raw/phase1/trace_c3.nsys-rep)

The head-merge copies are named `triton_poi_fused_mm_86` and
`triton_poi_fused_mm_93`. Their bodies only load, widen fp16 values exactly,
and store them unchanged in a different order; they perform no matrix
multiplication. Each moves **49,213,440 bytes in and 49,213,440 bytes out**.
Gating first writes the SDPA result in head-major order; the copy prepares
token-major input for the output projection. [Source at the copy call](evidence/extensions/c3/raw/phase1/active_dump/output_code.py),
[temporal argument metadata](evidence/extensions/c3/raw/phase1/call_metadata_c3.json),
[copy inventory](evidence/extensions/c3/raw/phase1/layout_attribution_audited.json)

| Layer | Source-proven A, ms/pass | B |
|---|---|---|
| L0 | 0.04482604 | UNRESOLVED |
| L1 | 0.04482536 | UNRESOLVED |
| L2 | 0.04493094 | UNRESOLVED |
| L3 | 0.04492326 | UNRESOLVED |
| L4 | 0.04459056 | UNRESOLVED |
| L5 | 0.04473134 | UNRESOLVED |

Per-layer values sum from the ordered event ledger; shared buffer names are
resolved at each call before reuse. [Ledger and per-layer totals](evidence/extensions/c3/raw/phase1/layout_attribution_audited.json)

The initial bandwidth references mismatch 36 candidate sites' dtypes and
12 gating sites' operation chain. The refinement measures ten cases for
actual normalization, rotary, and gating families; each layout variant
generates one fused kernel. Ratios span **0.732969×–1.071438×**, but the three
single-normalization cases fail tight layout comparison. The final axial
normalization also fuses into mask-head consumers, beyond a simple
attention-window attribution. Thus the refinement does not close B coverage.
[Reference coverage](evidence/extensions/c3/raw/phase1/bandwidth_reference_coverage.csv),
[all calibration results and source checks](evidence/extensions/c3/raw/matched_calibration/audit.json)

## Predictions versus measurements

The additional prediction was committed and pushed as
`9b1fefdefb4bc9962f2be4f867e77e1905b627df` before extension execution.
[Registration](PREDICTIONS_EXTENSION_C3.md), [timestamp evidence](evidence/extensions/c3/raw/manifest.json),
[provenance audit](evidence/extensions/c3/raw/provenance/audit_record.json)

| Claim | Outcome |
|---|---|
| C3 latency ≤31 ms | SUPPORTED for the implemented C3 arm: 30.3055275 ms |
| Staging copies and boundaries were ≥12% of wall | UNRESOLVED: causal contribution not isolated |
| R_layout in P-L1's 3–5% branch | UNRESOLVED: B incomplete |
| P-L2: hidden penalty ≥1.5× | Not observed in standalone calibration; full model prediction UNRESOLVED |
| P-L3/P-L4 recoverability and existing flags | UNTESTED in this Phase-1-only extension |

Sources: [timing audit](evidence/extensions/c3/raw/independent_audit.json),
[attribution](evidence/extensions/c3/raw/phase1/layout_attribution_audited.json),
[calibration audit](evidence/extensions/c3/raw/matched_calibration/audit.json).

## Original C1 study

C1 remains **35.403275 ms**, validated against eager on every registered
seed. Its original trace lacks compiled SDPA anchors, so its layout share
remains unresolved. Its Phase 2 was not run. The original report and all 948
evidence files are preserved without changing their hashes.
[Original report](https://github.com/Shakil-Muhit/axial-layout-scoping/blob/06e3fb22210f9d0c8fc8ef5f3051a5d27270da81/RESULT.md),
[original evidence](evidence/README.md), [original independent audit](evidence/raw/independent_audit.json)
