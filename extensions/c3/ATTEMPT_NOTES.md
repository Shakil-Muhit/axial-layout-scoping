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

## Attempt 2 result and native-tail compatibility boundary

Code `65dc05a`, generation `29764068-1927-4392-a3df-5d041a43054a`.
Eager outputs again matched stock exactly on all three seeds. Removing the
attention breaks exposed a full-forward AOT compiler assertion inside the
iSTFT decomposition: `copy_` writes to an internal `permute_357`, while
`assert_functional_graph` requires its destination to be a graph input.
Compilation failed before any accepted timing or trace. No compiler check
was disabled and no result from this attempt is accepted as C3 timing.

### Attempt 3 plan (recorded before execution)

Keep the existing native `torch.istft` call opaque through a custom-operator
registration. This registration contains no new GPU kernel or replacement
arithmetic. An external, mechanically checked copy of the public forward
changes only that call's callee. The native return shape, stride, dtype, and
storage offset were inspected on the fixed operating point; the fake
signature is checked by `torch.library.opcheck` and every real invocation.
The native call's host-side checks make it unsafe for CUDA-graph capture, so
its registration declares `cudagraph_unsafe`; no Inductor flag is changed.
The existing C1 compile call decides whether it can use CUDA graphs. Report
that decision explicitly, including any consequence for the prediction.

This compatibility boundary is an addition to the attention-only repair:
the earlier assumption that that repair alone enables whole-model Inductor
execution was false in the pinned environment. The final C3 arm must be
identified as attention repair **plus native iSTFT boundary**. Preserve both
failed attempts, require eager equivalence and the same baseline validation,
and retain the original extension deadline and all decision thresholds.

## Attempt 3 result; correction of the compiler-failure diagnosis

Code `f815538`. The native-iSTFT adapter passed schema/fake-tensor checks and
the full eager model remained exactly equal to stock. Full compilation
failed at the same internal `copy_`, now visibly **before** the opaque iSTFT
call. The initial attribution to iSTFT decomposition was therefore incorrect.
An isolated `index_fill` reproducer using the actual complex shape and
stride produced the same AOT assertion. This identifies the DC-filter
decomposition as the blocker; it is not evidence of an iSTFT bug.

### Attempt 4 plan (recorded before execution)

Remove the unnecessary native-iSTFT boundary. Wrap only the original
`Tensor.index_fill` DC-filter call as an opaque operator, with its native
shape, stride, dtype, and storage offset preserved. Restore the public
`torch.istft` call. Mechanically compare the external forward to the pinned
source, allowing just that one callee substitution. Before full-model work,
check schema/fake-tensor behavior, exact native outputs, CUDA-graph capture
and replay, and isolated full-graph compilation. No CUDA-graph-unsafe tag,
compiler flag, backend change, new GPU kernel, or arithmetic rewrite is used.

The final arm, if qualified, is **attention repair plus native DC-filter
boundary**. Keep all earlier failed attempts and the unchanged prediction.
Every GPU invocation remains inside the first attempt's one-hour window.
