# Axial-layout scoping — RESULT

**The registered baseline passes validation; layout attribution is UNRESOLVED.**

repo (measuring code): `9120bf31194674b76503c14348582cc27b7dd546` · sm120-nulltest tooling @ `a997e27d843badb3e742f4e57f3896ac61681431` · msst @ `ea7eb9c20ea0e3f94368a30fc1654b51cdd55789` · checkpoint SHA256 `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`. [Provenance](evidence/raw/manifest.json)

torch 2.8.0+cu128 · triton 3.4.0 · driver 595.84 · NVIDIA GeForce RTX 5090, GPU 4 · `use_amp=true`. Sustained baseline clocks: SM **2752 MHz**, memory **13801 MHz**; SM range 2745–2812 MHz within timed blocks. [Environment](evidence/env.json), [independent clock audit](evidence/raw/independent_audit.json), [clock samples](evidence/raw/phase0/clocks_sampled_phase0.csv)

## Decision inputs (mechanical)

`R_layout = UNRESOLVED; (A) = UNRESOLVED; (B) = UNRESOLVED.` The trace has attention calls, but the generated Inductor wrappers contain no SDPA anchors; generic copy kernels cannot establish layout-specific cost. No nonzero A-only lower bound is supported. [Attribution audit](evidence/raw/phase1/attribution_audit.json)

| Input | Result | Evidence |
|---|---|---|
| R_recovered | FAILED: attribution prerequisite unavailable; wrapper unrun | [Session summary](evidence/raw/summary.json) |
| flag_recovers | UNTESTED: Phase-2 threshold could not be established | [Session summary](evidence/raw/summary.json) |
| Head reshapes materialized | UNRESOLVED | [Partial kernel catalog](evidence/inductor_kernel_map.csv), [inventory](evidence/materialization_inventory.md) |
| Baseline compiled latency, 5a min-of-five medians | **35.403275 ms** | [Timing](evidence/baseline_timing.json), [independent recomputation](evidence/raw/independent_audit.json) |
| Median spread | 0.321688 ms | [Timing](evidence/baseline_timing.json) |
| Baseline throughput, best block | 35.583140 ms/pass | [Timing](evidence/baseline_timing.json) |
| Patched latency and delta | NOT RUN | [Session summary](evidence/raw/summary.json) |

Baseline vs eager, loose tier (`rtol=1e-2, atol=1e-3`), exact shapes and finite outputs: **PASS for every seed**. The saved tensors independently reproduce the recorded maxima. [Validation](evidence/validation_baseline.json), [CPU audit](evidence/raw/independent_audit.json)

| Seed | Result | max_abs_diff |
|---|---|---|
| 4242 | PASS | 1.17117306218e-05 |
| 1337 | PASS | 4.86569479108e-05 |
| 90210 | PASS | 9.43732447922e-05 |

Patched vs baseline, tight tier: **NOT RUN**. Tolerances were not changed.

## Threshold application (do not editorialize; just apply)

Registered rules: GO if `R_layout >= 0.05 AND R_recovered >= 0.60 AND flag_recovers == False`; TICKET if `0.03 <= R_layout < 0.05`, or `flag_recovers == True`, or `R_layout >= 0.05 AND R_recovered < 0.60`; NO-GO if `R_layout < 0.03`. [Registered protocol](evidence/raw/provenance/handoff.md)

**Result: UNRESOLVED.** Missing attribution cannot establish any threshold. The Phase-2 skip condition is not satisfied by an unknown value. No GO / TICKET / NO-GO decision is made.

## Predictions vs measurements

Predictions were committed and confirmed on the remote before execution. [Predictions](evidence/raw/provenance/PREDICTIONS.md), [timestamp audit](evidence/raw/provenance/audit_record.json)

| Prediction | Measured | Status |
|---|---|---|
| P-L1: layout share 3–5%; higher with materialized heads | Full share and head classification unavailable | UNRESOLVED |
| P-L2: hidden permuted-read penalty >=1.5x | Standalone copy / norm / pointwise ratios: 1.007420x / 1.045894x / 1.001754x. These do not establish the cost of the executed stack's fusions. | UNRESOLVED |
| P-L3: strided views recover >=60% | Wrapper not run; prerequisite unavailable | UNRESOLVED |
| P-L4: no existing compiler flag removes the cost | Knobs not run; prerequisite unavailable | UNRESOLVED |

Sources: [microbench records, including output dtypes and strides](evidence/raw/phase1/microbench_bwref.json), [attribution audit](evidence/raw/phase1/attribution_audit.json), [session summary](evidence/raw/summary.json). Norm and pointwise microbench outputs retain permuted strides; those timings cannot supply a verified materialization penalty for this baseline.

## Per-layer attribution table

| Layer | (A) | (B) | Materialization count |
|---|---|---|---|
| L0 | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| L1 | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| L2 | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| L3 | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| L4 | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| L5 | UNRESOLVED | UNRESOLVED | UNRESOLVED |

The partial [kernel catalog](evidence/inductor_kernel_map.csv) names all traced Inductor kernels and their possible source sites. It is not a completed A/B contribution ledger. [Generated sites](evidence/raw/phase1/generated_kernel_sites.csv) provide file and line references. Sublayer ownership is unavailable because the unmodified public model graph-breaks around attention.

## Materialization inventory

See [inventory and limitations](evidence/materialization_inventory.md). No generated wrapper contains SDPA, so neither head splitting nor merging can be proved view-only from the requested wrapper-to-attention mapping. [Generated code](evidence/output_code_baseline/), [attribution audit](evidence/raw/phase1/attribution_audit.json)

## What failed / could not be determined

- The fixed C1 baseline settles and validates, but Dynamo records graph breaks involving `FlashAttentionConfig._asdict`, generators, and a recompile limit. Attention runs outside the generated Inductor graphs. [Baseline counters](evidence/baseline_timing.json), [execution log](evidence/raw/session_stdout_20260915.log)
- The trace has **50 passes, 1,396 GPU events/pass, and 12 Flash Attention launches/pass**. All exported kernel totals match independently. The corresponding **33 generated-code files contain zero SDPA calls**, so the mandatory anchor check fails. No mismatch override was used. [Trace audit](evidence/raw/phase1/attribution_audit.json)
- Generic COPY-class kernels inside the first-to-last attention window account for **2.156783 ms/pass**, a diagnostic total. They do not identify which bytes are axis permutations: CUDA-graph input staging also uses `_foreach_copy_`. Counting that duration as (A) would be unsupported. [Trace audit](evidence/raw/phase1/attribution_audit.json), [installed Torch source excerpt](evidence/raw/provenance/cudagraph_input_copy_excerpt.txt)
- The primary trace used outer pass ranges. A supplemental run then tested the original P8 per-layer NVTX hooks with the same C1 call. It produced annotations but changed the GPU-event inventory from **1,396 to 1,390 events/pass** and failed tight validation on seeds **4242 and 90210**. That trace is rejected for baseline attribution under the predeclared equivalence rule. [Equivalence audit](evidence/raw/hook_diagnostic/equivalence_audit.json), [inventory differences](evidence/raw/hook_diagnostic/kernel_inventory_diff.csv), [validation](evidence/raw/hook_diagnostic/validation.json), [annotated trace](evidence/raw/hook_diagnostic/trace.nsys-rep)
- The public source, checkpoint, C1 compile call, and tolerances were held fixed. Phase 2 was not run. The public clone and `sm120-nulltest` remain unchanged. [Post-session checks](evidence/raw/provenance/post_session.txt), [audit record](evidence/raw/provenance/audit_record.json)
- The primary pipeline lasted **308 seconds**. Including the supplemental annotation cross-check, all GPU work used the same host/GPU within **1175 seconds** of the original study start, below the registered four-hour limit. [Session audit](evidence/raw/provenance/audit_record.json)

## Files

- [Evidence download and restoration](evidence/README.md): full lossless archive published to the same private repository's evidence branch.
- [Run manifest](evidence/raw/manifest.json): measuring-code hashes, checkpoint/config/model pins, generation identifier.
- [Baseline timing](evidence/baseline_timing.json), [validation](evidence/validation_baseline.json), [environment](evidence/env.json), [independent audit](evidence/raw/independent_audit.json).
- [Nsight report](evidence/raw/phase1/trace_baseline.nsys-rep), [SQLite export](evidence/raw/phase1/trace_baseline.sqlite), [GPU trace](evidence/raw/phase1/trace_baseline_cuda_gpu_trace.csv), [kernel summary](evidence/raw/phase1/trace_baseline_cuda_gpu_kern_sum.csv).
- [NVTX trace](evidence/raw/phase1/trace_baseline_nvtx_gpu_proj_trace.csv), [NVTX summary](evidence/raw/phase1/trace_baseline_nvtx_gpu_proj_sum.csv), [export status](evidence/raw/phase1/trace_baseline_exports.json).
- [All kernel classes](evidence/raw/phase1/all_kernel_classes.csv), [20-row sample](evidence/raw/phase1/classifier_sample.csv), [classifier source](evidence/raw/provenance/classify.py).
- [Generated-code dump](evidence/output_code_baseline/), [partial map](evidence/inductor_kernel_map.csv), [bandwidth diagnostics](evidence/raw/phase1/microbench_bwref.json), [complete session log](evidence/raw/session_stdout_20260915.log).
