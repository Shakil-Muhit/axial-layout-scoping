# Materialization inventory — incomplete attribution

The fixed C1 baseline's attention executes outside Inductor's graphs. The
[trace audit](raw/phase1/attribution_audit.json) establishes the mismatch:
Flash Attention launches exist, but generated SDPA wrapper calls do not.

The [generated-site catalog](raw/phase1/generated_kernel_sites.csv) lists all
parsed Inductor call sites with source file/line and argument information.
The [partial kernel map](inductor_kernel_map.csv) matches observed kernel names
to their possible source sites. Names can be reused by different compiled
graph variants; those possibilities are retained rather than arbitrarily chosen.

## Per-sublayer inventory

Time and frequency attention in every layer: materialization counts, tensor
sizes at the SDPA boundary, and consuming-op identity are **UNRESOLVED**.
Head SPLIT: **UNRESOLVED**. Head MERGE: **UNRESOLVED**.

Compiled pure-clone sites in the dump concern the front-end or reconstruction;
none of the traced Inductor clone events fall between the first and last
attention launches. Generic Copy kernels do occur there, but their names
cannot distinguish axis permutation from CUDA-graph input copying. Thus no
nonzero layout-specific A-only lower bound is established. This is not evidence
that layout cost is zero. [Trace audit](raw/phase1/attribution_audit.json),
[installed input-copy implementation](raw/provenance/cudagraph_input_copy_excerpt.txt)

No patched forward was measured. No materialization-reduction claim is made.

## Protocol-hook cross-check

The original P8 hooks were also tested on the fixed C1 call. They produced
per-layer annotations but changed the kernel inventory and failed strict
numerical equivalence. Their ranges cannot be applied to the baseline.
[Equivalence audit](raw/hook_diagnostic/equivalence_audit.json),
[changed inventory](raw/hook_diagnostic/kernel_inventory_diff.csv).
