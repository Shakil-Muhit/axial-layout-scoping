# PREDICTIONS — axial-layout scoping study

Committed: 2026-09-15, BEFORE any measurement run (handoff rule 1).
Text below is Section 3 of docs/handoff.md, verbatim.

## 3. Pre-registered predictions (commit as `PREDICTIONS.md` before any run)

Derivation anchors (8 s, batch 1): tokens = 60 × 801 = 48,060; activation (tokens × 384 ×
2 B) ≈ 37 MB; QKV output (tokens × 1536 × 2 B) ≈ 147 MB; 6 layers × 2 axial sublayers.

- **P-L1 (share).** Materialized layout changes attributable to the axis swaps account for
  **3–5 %** of compiled forward wall at 8 s on the 5090. It reaches **≥ 10 %** only if the
  per-attention head-split/merge reshapes are *also* materialized (each ≈ 147 MB read+write).
  Mechanism: axis swaps ≈ 4 × 37 MB × 2 per layer ≈ 1.8 GB/forward ≈ 1 ms at 1.8 TB/s.
- **P-L2 (hidden form).** At least part of the swap cost is *not* a visible copy kernel: Inductor
  fuses a permuted read into the following norm/elementwise kernel, so the transpose shows up
  as an uncoalesced-load penalty. Those fused kernels run **≥ 1.5×** slower than the same
  kernel on a contiguous input at equal shape.
- **P-L3 (recoverability).** Passing strided q/k/v views into SDPA (flash or mem-efficient) with
  no other change removes **≥ 60 %** of `R_layout`, validated tight-tier against the compiled
  baseline, because both backends accept arbitrary batch/seq/head strides given a contiguous
  head_dim.
- **P-L4 (compiler).** No existing Inductor flag removes it (`layout_optimization`,
  `keep_output_stride`, `permute_fusion`, `triton.persistent_reductions`, and the
  max-autotune modes). If one does, this is a configuration ticket, not a research lane.

Owner's dissent line (optional, fill before the run or leave blank): ______
