# Design notes — axial-layout scoping (pre-implementation)

Grounded against msst @ ea7eb9c20ea0e3f94368a30fc1654b51cdd55789 (the pin in
sm120-nulltest/configs/msst_commit.txt) and the KJ config
(`configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml`):
dim 384, depth 6, heads 8, dim_head 64 → dim_inner 512, to_qkv 384→1536,
flash_attn=True, both axial transformer depths 1, stereo. Attend pins the
flash SDPA backend on cap ≥ 8.0 (`cuda_config = (True, False, False)`),
applies no extra arithmetic at eval (scale=None, dropout=0).

## Execution environment (deviation from handoff §2, owner-directed 2026-09-15)

GPU host = **ml-beast** (5× RTX 5090, driver 595.84), not a RunPod pod.
`sm120-nulltest/setup.sh` cannot run as-is (no sudo); its EFFECTS are
reproduced by `scripts/beast_bootstrap.sh`: venv torch 2.8.0+cu128 (triton
3.4.0), user-space Nsight Systems (dpkg -x, graph-trace verified), msst clone
@ pin, checkpoint via `sm120-nulltest/scripts/fetch_checkpoint.sh` (rule 2
verbatim). All study runs pin `CUDA_VISIBLE_DEVICES=4`; **GPU 0 carries an
unrelated external workload for the whole study** — recorded for checklist
item 6 (clock stability read on GPU 4 only). nsys runs add `-s none
--cpuctxsw=none` (no root → no CPU sampling; CUDA kernel + NVTX tracing
unaffected).

## Baseline (fixed)

`torch.compile(model, mode="reduce-overhead", dynamic=False)` — the verbatim
arm-C1 call from sm120-nulltest@a997e27 `arms/common.py` (`build_arm_step`,
C1/C2 branch: `compiled = torch.compile(fe, mode=mode, dynamic=False)` with
mode='reduce-overhead' for C1). Input `torch.randn(1, C, 352800)`, C from
`model.stereo` (=2), seed 4242 via CPU generator → cuda. Autocast
`torch.autocast("cuda", enabled=config.training.use_amp)`, `no_grad`, eval.
Settle: dynamo-counters unchanged for 5 consecutive passes under
`TORCH_LOGS=recompiles` (sm120 `settle_compile` copied).

Known dynamo behavior at this pin/torch: complex STFT/iSTFT graph-break; the
transformer stack compiles inside Inductor graph(s); SDPA lowers to the
flash fallback kernel. cudagraph trees active (reduce-overhead) → all traces
use `--cuda-graph-trace=node`, with the kernels-per-pass floor guard
inherited from sm120 (a graph arm showing ~1 kernel/pass = broken tracing).

## Attribution (Phase 1) — the NVTX-inside-compile problem

`p8_trace.py`-style hooks on modules INSIDE a compiled forward either graph-
break dynamo (nvtx calls are unsupported side effects) or don't fire —
both perturb the measurand. Two-path plan:

1. **Primary: kernel-order segmentation.** Per-pass kernel sequence from the
   node-level trace, segmented on the 12 SDPA anchor kernels per pass
   (time-axis: batch 60 × seq 801; freq-axis: batch 801 × seq 60 —
   unambiguous by grid/duration), cross-referenced with the wrapper `call()`
   order in `output_code_baseline/` (Inductor emits calls in execution
   order). NVTX only on the OUTER OptimizedModule (eager wrapper hooks:
   per-pass iteration ranges) — safe, no dynamo interference.
2. **Optional cross-check: hooked-compile run.** Inner hooks accepted IF the
   kernel name+count inventory matches the unhooked baseline (fusions are
   bounded by module boundaries); else discarded. Never the headline path.

`R_layout` denominator: the CLEAN (untraced) 5a median (min-of-5). Kernel
durations (A)/(B) come from the trace (device-side, ~nsys-invariant); the
traced run's own wall is also recorded and both ratios reported.

## Phase 2 patch (monkeypatched forward copy, grounded)

Canonical activation: token-major contiguous `(T=801, F=60, D=384)`; flat
view `(T·F, D)` for every per-token op. Per layer, per sublayer, calling the
ORIGINAL modules' weights (attn.norm, attn.to_qkv, attn.rotary_embed,
attn.attend, attn.to_gates, attn.to_out, ff, tx.norm — identical math):

- `qkv = attn.to_qkv(attn.norm(x_flat))` → view `(T, F, 3, H, dh)` [true
  view of the contiguous mm output].
- time-attn: `q/k/v = view.permute → (F, H, T, dh)`, strides
  `(1536, 64, 92160, 1)` — head_dim contiguous, flash-legal, ZERO copies.
- freq-attn: `(T, H, F, dh)`, strides `(92160, 64, 1536, 1)` — same.
- rotary via the sublayer's own `rotary_embed` (rotates dim -2 = the correct
  axis in both orientations).
- SDPA via the module's own `attn.attend` (keeps the flash-pinned backend
  and exact math; satisfies handoff §5.3's backend-pinning requirement
  through the model's own mechanism — documented).
- gating: `attn.to_gates(normed)` viewed `(F,H,T,1)` / `(T,H,F,1)` —
  strided elementwise, no copy.
- head-merge before `to_out`: ONE `.reshape` copy per attention —
  **parity with stock** (stock's `'b h n d -> b n (h d)'` merge is equally a
  copy), so it cancels in the delta; documented in the materialization
  inventory.
- residuals + FF + final `tx.norm` on the canonical flat — per-token.

What the patch removes: the per-layer `rearrange + pack` axis-swap
materializations (2×37 MB/layer minimum) and the permuted-read penalties
feeding norms. What it keeps (parity): head-merge copies, rotary allocs.

Validation ladder: (i) CPU fp32, chunk 4410, random weights: patched vs
stock eager — expected ≤1e-5 (same ops, GEMM batching differs → tiny
accumulation drift); (ii) GPU: patched-compiled vs baseline-compiled, tight
tier per handoff, 3 seeds; a looser result is reported as a finding.

## Knobs (Phase 2 pre-step, unpatched baseline)

`torch._inductor.config.layout_optimization` (True/False),
`keep_output_stride` (False), `permute_fusion` (True),
`triton.persistent_reductions` (False), `mode="max-autotune"`,
`"max-autotune-no-cudagraphs"`. Each: compile+settle (timeout 25 min →
UNTESTED(timeout)), output_code dump, materialization inventory diff vs
baseline, loose validation, one 5a block. Full trace (for (A)+(B)) only when
the inventory changes; an identical inventory is recorded as no-change with
the diff as evidence.

## Evidence & repo hygiene

evidence/ git-ignored except *.md (handoff §2 literal); every artifact lives
in the working tree on the shared mount for the owner's local audit;
RESULT.md carries a path per number. GPU-run outputs stage on ml-beast local
disk (`~/axial/runs/`) and are rsynced into `evidence/` (sshfs is slow +
noexec). Git operations run on muhit-pc only (sshfs noexec breaks hook
execution on ml-beast — established empirically; the pre-commit gate is
live on the muhit-pc side).

## Post-review-01 updates (docs/codex/01_design.md, gpt-6-astra ultra)

Codex's sandbox (bwrap) cannot initialize on ml-beast (`bwrap: loopback:
Failed RTM_NEWADDR`) — reviews therefore receive file contents INLINED in the
prompt (read-only by construction); noted as review-loop mechanics, not a
study deviation. Provisional findings adopted:

1. Stride math independently verified (temporal (1536,64,92160,1), frequency
   (92160,64,1536,1), both flash-legal) with upstream refs.
2. Verification signal for 1d sharpened: an SDPA input that aliases the qkv
   GEMM allocation (reinterpret_tensor) = view; a fresh buffer written by a
   preceding triton kernel = materialization EVEN IF fused with RoPE/cast and
   carrying no "clone" in its name. kernel_map records each SDPA input's
   producer. Expected clean signal: v aliases qkv; q/k are rotary outputs.
3. Patch v2 — head-merge via views: flash output memory is (b,s,h,d)-
   contiguous presented as (b,h,s,d). Gating and to_out therefore run in the
   OUTPUT-NATIVE memory order ((F,T,H·dh) for time-attn), where the flat
   (F·T, 512) view is free; the canonical (T,F,·) orientation is restored
   only at the residual through a strided view feeding the elementwise add.
   Zero additional materializations in eager; whether Inductor preserves this
   is exactly what the patched output_code dump shows.
4. sdpa_kernel backend pinning is dynamo-supported in 2.8; the patched
   forward reuses the model's own Attend (flash-pinned) and the patched
   output_code dump verifies which backend kernel actually ran.
5. Ordinal-attribution caveat recorded: segmentation validates the 12 SDPA
   anchors per pass; intervening fused kernels are attributed to their
   segment by order (wrapper cross-reference), not proven ownership.

## Pre-execution fixes (caught by the CPU smoke, 2026-09-15)

- **Drift-guard sha corrected by live-fire.** The first recorded
  `PINNED_FORWARD_SHA16` did not match `inspect.getsource` of the installed
  pinned clone (`cc67576c4ab72fd0` vs true `0db2007396ab2646`); the
  fail-closed guard refused to swap, exactly as designed. Root cause: the
  constant was derived from a differently-normalized text, and the
  transcription had silently normalized two cosmetic spots
  (`rearrange(stft_repr,'…')` comma spacing; `tensor(0, device = device)`)
  and replaced the loss branch with a raise while claiming verbatim.
- Fix: transcription made byte-verbatim (loss branch included) modulo three
  DECLARED deviations (D1 4-space dedent, D2 marked import block, D3 the one
  PATCH region); constant updated; and the claim is now checked MECHANICALLY
  by `scoping/verify_transcription.py` (line-by-line vs the installed clone),
  which `run_phase2.sh` runs before any swap and whose report lands in
  `evidence/phase2/transcription_check.json`. Swap smoke (checkpointed model,
  CPU) passes: drift guard + all structural asserts.

## Post-review-02 updates (docs/codex/02_code.md, gpt-6-astra ultra)

Gate-02 (pre-execution code review) returned 9 BLOCKERs + 7 MAJORs; all
adopted:

1. (f1) output_code kernel metadata now anchors on the
   `name = async_compile.triton('name', ...)` assignment (last `Original
   ATen:` comment before it), with ops-from-name fallback recorded as
   `ops_source=name` and full coverage counters; unresolved coverage flags
   R_layout as a lower bound.
2. (f2) call args parsed IN ORDER and role-tagged from the kernel def's own
   `in_ptr/out_ptr/in_out_ptr` params (extern: `out=` kwarg; fallback:
   assigned name = output); dtype resolved through the alias chain to the
   producing alloc; `is_permuted` is a stride-ORDER test (catches
   (60,801,384)/(384,23040,1)); inventory.py reuses the same parser and
   counts permuted READS only.
3. (f3) microbench v2: per-CLASS BW_ref (copy/norm/pointwise, no cross-class
   median), CUDA-event device timing (matches the device-side (B) numerator),
   copy chain is a true layout conversion
   (`clone(memory_format=contiguous_format)`), output strides recorded.
4. (f4) (A)/(B) restricted to the transformer stack: interior SDPA window +
   kernel-NAME reuse for the layer-0 prologue / layer-5 epilogue (Inductor
   dedupes identical kernels across the 6 layers); memcpys split into
   in-window (in A) vs boundary (reported, never in A), averaged over ALL
   passes.
5. (f5) hard pass validation: n_pass must equal meta iters (nvtx-window
   splitting primary, gap heuristic fallback, method recorded), per-pass SDPA
   anchor count checked, axis order checked; mismatches fail or flag.
6. (f6) heads-views verdict via producer ancestry (alias of the qkv GEMM =
   view; triton-copy producer = materialization; rotary = compute-output) +
   merge-consumer inspection per SDPA; file/line cited.
7. (f7) the CSV is a full ledger (unrounded ns, contrib_A_ns/contrib_B_ns,
   memcpy rows); JSON totals are SUMMED FROM the ledger.
8. (f8) `require_settled`: capped settle exits 4 before any measurement
   (drivers), knob capped settle -> UNTESTED.
9. (f9) compare_outputs: exact-shape + finiteness required; failure reason
   serialized.
10. (f10) knobs validate TIGHT vs compiled-baseline outputs (acceptance)
    plus loose-vs-eager diagnostic.
11. (f11) finalize gates R_recovered on: cpu parity, settle ok, tight pass,
    headline timing, patched map present, in-stack (A) fell, no copy feeding
    SDPA; otherwise FAILED(reasons), raw delta kept as *_DIAGNOSTIC.
12. (f12) `persistent_reductions_false` knob added (nested config path);
    changed-inventory knobs get an automated follow-up nsys trace +
    kernel_map; finalize emits flag_recovers True/False/UNTESTED(reason).
13. (f13) sustained 1 Hz clock sampler (handoff rule 9 query set) during
    every phase; timed blocks carry unix_start/unix_end; finalize computes
    baseline-vs-patched SM-clock medians within the timed windows (5% gate).
14. (f14) run manifest fingerprints the measuring code (scoping/*.py +
    run scripts) + ckpt sha; later phases refuse on drift (ALLOW_DRIFT=1
    escape recorded). Knob resume only skips recorded results.
15. (f15) threshold-skip path still runs finalize (summary.json exists for
    RESULT); summary carries throughput/spread/per-seed validation.
16. (f16) nvtx projection exports (trace + sum) attempted; absence writes a
    MISSING marker and kernel_map records the gap-splitting fallback in
    status_flags.
17. (f17) statistics.median everywhere.

Plus: scoping/tests/test_parse_fixture.py — an offline fixture test of the
parser layer (found and fixed a reinterpret_tensor-without-offset regex gap
before any GPU time).

## Post-review-03 updates (docs/codex/03_delta.md, gpt-6-astra ultra)

Gate-03 (delta) verified the gate-02 closures and raised 7 residual blockers
(f19-25) + majors (f26-31); all adopted:

- (f19) kernel metadata keyed by the WRAPPER symbol; inner def name recorded
  (mismatch counted); def + load/store exprs read from that kernel's own
  source slice; comment window bounded by the previous assignment, no
  char cutoff. `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` pinned on all
  trace/dump runs (verified already default-True in this torch build).
- (f20) plain buffer-reuse lines are aliases; strict arg/param alignment;
  misaligned or dtype-unresolved sites are EXCLUDED from (B) and counted
  (never priced at a silent fp16 default). Permuted reads get a SECOND
  signal: kernel-source load-vs-store index expressions (Inductor bakes
  permuted indexing into kernels behind bare base pointers). Over-detection
  is safe (clamped penalty of a coalesced kernel ~ 0); under-detection is
  what the coverage counters flag.
- (f21) microbench under autocast (model-context parity); timing = CUDA-graph
  replay of 50-launch blocks (event-timed; eager fallback recorded); the
  copy class must write contiguous output on the permuted input or it is
  dropped from BW_ref and flagged.
- (f22) the kernel-dedup assumption behind name-based stack membership is
  CHECKED: an anomalously low L0 contribution vs the interior-layer median
  raises an explicit lower-bound flag.
- (f23) anchor mismatches are fatal (like pass-count) without
  --allow-pass-mismatch; per-name durations come only from validated pass
  windows; nvtx windows must be unique/contiguous/non-overlapping; prefix
  matching cannot extend a trailing ordinal and must be unique; per-name
  launches-per-pass must equal wrapper multiplicity (else flagged);
  kern_sum aggregate cross-check; gap-heuristic fallback is itself a flag.
- (f24/f25) finalize: R_recovered requires baseline evidence accepted, BOTH
  maps status 'ok' with complete (B), parity/settle/tight/headline gates,
  in-stack (A) fell, patched split+merge head verdicts 'true', zero
  SDPA-adjacent COPY events, and clock comparability within 5% — computed
  BEFORE acceptance. flag_recovers enumerates the 7 expected knobs;
  True needs an acceptance-grade follow-up map with >=60% (A)+(B) reduction
  and tight validation; False needs every knob resolved (the
  validation-disqualified rule is recorded in the summary); else UNTESTED.
- (f26) producer map is position-aware (latest write BEFORE the consumer);
  head verdicts are tri-state with unknown ancestry never counting as a
  view; split and merge reported separately; SDPA-adjacent COPY launches
  counted from the trace as the silent-failure signal.
- (f27) knob follow-up traces generate their OWN debug dump (dump_trace) and
  inventory; stale metadata cleared; trace-result status checked before
  mapping.
- (f28) manifest rebind protection: phase 0 refuses to re-stamp existing
  phase-1/2 artifacts after a code change (ARCHIVE_OLD=1 auto-archives);
  later phases verify code sha + ckpt sha + config path.
- (f29) nvtx exports tracked per report; incomplete exports leave a marker,
  and a gap-split map is FLAGGED, which finalize refuses to accept.
- (f30) every phase-2 failure path runs finalize (summary.json always
  exists) and writes phase2.failed. RESULT.md itself remains agent-assembled
  from summary.json + evidence paths per the handoff.
- (f31) the transcription checker enforces the EXACT permitted import block
  inside the ADAPTATION markers, asserts marker uniqueness/order, and
  declares its normalizations (dedent, def rename, seam blanks) in the
  module docstring. Re-verified VERBATIM-OK on ml-beast.

## Orchestration decisions (scripts/run_phase*.sh)

- output_code dumps come from the SAME process as the nsys trace
  (trace-iters modes) so the wrapper<->trace cross-reference is exact; the
  clean timed runs carry no debug env. `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`
  on every dump/knob run: a warm FX-graph cache can skip codegen (empty
  dump), and knobs must never reuse cached baseline artifacts.
- Kernels-per-pass floor: the exported gpu-trace CSV must have >= 1000 rows
  for 50 passes (sm120 lesson: ~1 kernel/pass = graph replays not expanded).
- Phase-2 gate (R_layout >= 0.03) enforced at the top of run_phase2.sh;
  knob timeouts (25 min) recorded as UNTESTED; every phase writes a .done
  marker and full logs under ~/axial/runs/logs (synced to evidence/logs).
