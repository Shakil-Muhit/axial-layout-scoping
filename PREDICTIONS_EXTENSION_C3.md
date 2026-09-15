# Registered bounded extension — C3 graph-break repair

Date: 2026-09-15. Registered and pushed before this extension's GPU execution.
The original [predictions](PREDICTIONS.md) and C1 evidence remain unchanged.

## The one additional prediction (owner supplied, verbatim)

> With graph breaks removed, compiled latency drops from 35.4 ms to ≤ 31 ms (the staging copies and segment boundaries were ≥ 12% of wall), and R_layout lands in P-L1's 3–5% branch.

## Scope and acceptance rules

- One extension, at most 3,600 seconds of GPU session on the same RTX 5090,
  GPU 4 of `ml-beast`. Run Phase 1 only, with the baseline qualification and
  unprofiled timing needed to evaluate the new prediction. No Phase 2, flag
  sweep, layout intervention, new kernel, or changes to `sm120-nulltest`.
- C3 is the full pinned public model with an external replacement for
  `Attend.flash_attn`. Its only semantic-source difference is replacing
  `config._asdict()` with explicit reads of the same three configuration
  fields. Preserve scale, dropout, backend flags (including implicit
  defaults), tensor layouts, weights, precision, and the C1 compile call.
  Verify the source transformation mechanically. Never edit the public clone.
- The existing checkpoint/config/model pins, input size, seeds, AMP setting,
  five stable warmup passes, 5a timing, and five repeats apply unchanged.
- Require C3 vs saved stock eager outputs at the existing loose tier
  (`rtol=1e-2, atol=1e-3`) on all three seeds to qualify this new compiled
  baseline. Also report C3 vs saved C1 outputs at the unchanged tight tier
  (`rtol=1e-4, atol=1e-5`) on all three seeds. A failed tight comparison
  prevents a claim of tight equivalence to C1. This compiler graph repair is
  not the Phase-2 layout intervention; no Phase-2 acceptance is inferred.
- Require zero graph breaks with `TORCH_LOGS=graph_breaks,recompiles`, checked
  against counters and the complete log. Require settled execution, no
  suppressed compiler errors, and generated SDPA anchors matching every
  captured pass. Missing proof stops dependent attribution.
- Trace the same fixed C3 arm in a separate process with node-level CUDA
  graph tracing and generated code from that process. Use outer pass NVTX
  ranges; do not reinstall the rejected per-module hooks. Attribute layers
  using generated source/dataflow and verified SDPA order, recording limits.
- Repeat Phase-1 kernel classification, materialization inventory, and the
  three existing contiguous/permuted class microbenchmarks. Report A and B
  separately. SDPA anchors alone do not prove complete attribution. Unknown
  ownership, missing accesses, uncertain byte counts, or mismatched reference
  kernels remain unresolved; generic runtime copies are not layout cost.
- Preserve all original decision thresholds. A complete `R_layout < 0.03`
  yields **NO-GO by threshold** for C3. A complete value in `[0.03, 0.05)`
  yields **TICKET by threshold**. At or above 0.05, the unrun Phase-2 inputs
  remain necessary. An A-only lower bound below 0.03 cannot establish NO-GO.
- Compare C3's unprofiled timing to the retained C1 value, recording clocks
  and other-GPU load. This historical comparison tests the latency clause;
  it does not independently isolate staging-copy causality. The 3–5% clause
  requires complete attribution. Do not tune the wrapper to the prediction.
- Keep failed attempts and raw evidence in a new generation; publish the
  code, amended report, and lossless evidence to the authorized repository.
