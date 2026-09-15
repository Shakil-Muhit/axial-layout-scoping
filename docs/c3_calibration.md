# C3 Phase-1 reference refinement (before execution)

The original three generic chains leave 48 C3 candidate sites unpriced:
36 have different input/output dtypes and 12 use sigmoid gating rather
than the reference's GELU chain. The final axial norm also fuses into
mask-head consumers, which is a remaining coverage issue.

Within the original extension's one-hour deadline, evaluate standalone
PyTorch chains for the actual normalization (single/double, residual dtype,
axis), rotary, and gating families. Compare identical tensor values in
contiguous and model-like physical orders. Keep AMP, loose comparison to
eager, and tight layout-only comparison unchanged. Use the existing
CUDA-event/CUDA-graph microbench timing, preserving generated code.

These are Phase-1 calibration diagnostics, not new predictions, model
interventions, or Phase-2 runs. Do not use them as B until generated fusion,
operation chains, tensor dtypes/shapes/traffic, and full boundary coverage
are verified. Failure or incomplete coverage leaves B unresolved. Retain
failed cases and the original generic reference measurements.
