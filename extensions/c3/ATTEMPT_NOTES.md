# C3 extension execution notes

The prediction in `PREDICTIONS_EXTENSION_C3.md` remains unchanged.

## Attempt 1 — direct fields, legacy context

Code `45274f5`; run `c3_20260915`, generation
`13ae324f-3c27-436f-a533-43469fd5d668`. Eager outputs matched the saved stock
eager outputs exactly for every registered seed. The explicit-field repair
removed `_asdict`, exposing a second graph break: the installed Torch 2.8
legacy `torch.backends.cuda.sdp_kernel` returns an unsupported generator
context. The graph-break gate rejected it before timing. The complete failed
attempt is retained.

## Attempt 2 — equivalent supported context (recorded before execution)

Replace the legacy context with `torch.nn.attention.sdpa_kernel`, which the
installed Dynamo explicitly supports. Precompute its enum lists outside the
compiled method from the original configuration. Preserve the legacy API's
enum ordering and its implicit `enable_cudnn=True` default; do not silently
pin to Flash alone. Compare the enabled backend states and restoration under
both APIs before compilation. The external method is checked against the
original AST with only the configuration selection and context call changed.

This corrects the implementation of the authorized graph-break repair. It
does not change tensor operations, precision, thresholds, or the prediction.
The new attempt has a distinct evidence generation and may use only the
time remaining in the original extension's 3,600-second window. As the owner
confirmed, loose agreement to stock eager qualifies C3 for Phase 1; any tight
disagreement with C1 is reported separately.
