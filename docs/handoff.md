# Handoff — Axial-layout scoping study (offline regime, MelBandRoformer)

**Owner:** Muhit. **Executor:** coding agent, autonomous. **Time box:** ≤ 2 agent-days total,
≤ 1 GPU session (≤ 4 h) on the RTX 5090 pod. **Output:** one `RESULT.md` + evidence files,
committed and pushed to a new repo `axial-layout-scoping`. The owner audits `RESULT.md` and
makes a GO / TICKET / NO-GO decision. **You do not make that decision, and you do not build
anything beyond Phase 2.**

---

## 0. The question (read twice)

In the compiled forward of MelBandRoformer at the offline operating point (8 s chunk, batch 1,
fp16 autocast), the transformer stack alternates attention over two axes (time: sequences of
T≈801 frames; frequency: sequences of 60 bands). Every op *except* the attention core is
per-token and therefore layout-agnostic; only attention groups tokens into sequences. In
principle the model needs one canonical activation layout plus two *strided views* of it.
In practice the code — and then `torch.compile`/Inductor — appears to **materialize** the
permuted layout (clone/contiguous kernels, or uncoalesced reads inside fused kernels) around
the opaque SDPA and GEMM nodes.

**Question:** what fraction of compiled offline wall time is attributable to these materialized
layout changes, and how much of it is recoverable *without a new kernel* by passing strided
views to SDPA? That fraction, measured, decides whether this is a research lane
(layout propagation through opaque compiler boundaries), a production ticket, or nothing.

You are producing **three numbers** and the evidence behind them:

1. `R_layout` — layout-attributable share of compiled forward wall at 8 s.
2. `R_recovered` — share of `R_layout` removed by the strided-view intervention (Phase 2).
3. `flag_recovers` — whether any *existing* Inductor configuration already removes it.

---

## 1. Non-negotiable rules

1. **Predictions before measurements.** Your first commit is `PREDICTIONS.md` (Section 3,
   verbatim, with the date). No measurement runs before that commit exists and is pushed.
2. **Provenance.** Model code comes only from the pinned public clone
   (ZFTurbo/Music-Source-Separation-Training @ the commit in `sm120-nulltest/configs/
   msst_commit.txt`) via its own `get_model_from_config` / `load_start_checkpoint` path;
   checkpoint via `sm120-nulltest/scripts/fetch_checkpoint.sh` (sha256-verified); config =
   line 3 of `configs/checkpoint.txt`. **Never edit files inside the clone.** All interventions
   are external: wrapper modules, forward hooks, monkeypatched *copies* of methods living in
   this repo.
3. **Precision mirrors the public inference path:** `torch.autocast("cuda",
   enabled=config.training.use_amp)`, `no_grad`, `eval()`. Record `use_amp` in every JSON.
4. **Reuse, don't reinvent:** timing protocol 5a (per-iteration `synchronize` latency block +
   one free-running throughput block), the two-tier validation gate, the nsys invocation with
   `--cuda-graph-trace=node`, the NVTX hook pattern from `sm120-nulltest/scripts/p8_trace.py`,
   and the pre-commit integrity hook (`scripts/precommit.sh`, `core.hooksPath=.githooks`).
   Copy them into this repo with their headers intact; cite the source file in each copy.
5. **Do not touch `sm120-nulltest`.** Read from it; never write to it.
6. **Every number in `RESULT.md` must point at a file** (JSON/CSV/`.nsys-rep`/`output_code`)
   in `evidence/`. No number without a path. No "seems", "appears", "roughly" in the results
   table — a number, its source file, or `UNRESOLVED` with the reason.
7. **Loud failure over graceful degradation.** If a step cannot be completed as specified,
   stop that phase, write what failed and why into `RESULT.md`, and continue only with phases
   that do not depend on it. Never substitute a proxy measurement silently.
8. **No scope creep.** You will be tempted to write a kernel, to fix Inductor, to try FP8, to
   look at the streaming chunk. Don't. The deliverable is the three numbers at 8 s.
9. **Hash-pin your own artifacts:** `RESULT.md` cites the commit hash of this repo, the
   `sm120-nulltest` hash you copied tooling from, the msst commit, the checkpoint sha256,
   `torch.__version__`, `triton.__version__`, driver, GPU name, and sustained SM/memory
   clocks during the timed runs (`nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw
   --format=csv -l 1` logged in the background).

---

## 2. Setup (Phase 0)

- Repo: `axial-layout-scoping`, private, `main`. Layout: `scoping/` (code), `scripts/`,
  `evidence/` (git-ignored except `evidence/*.md`), `docs/`. Install the pre-commit hook first.
- Pod: same image/setup as `sm120-nulltest/setup.sh` (run it; it is idempotent). Confirm
  `nsys profile --help | grep cuda-graph-trace` succeeds.
- **Baseline definition (fixed, do not vary):** the full public model, compiled exactly as
  `sm120-nulltest` arm **C1** compiles it (copy the call verbatim from `arms/common.py` and
  cite the line). Input: `torch.randn(1, C, 352800)` with `C` from `model.stereo`, seed 4242.
  Warmup until no recompilation for 5 consecutive passes (log `TORCH_LOGS=recompiles`).
- **Validation gate:** compiled baseline vs eager stock forward on seeds 4242/1337/90210 at
  the *loose* tier (`rtol=1e-2, atol=1e-3`) — Inductor regenerates kernels. Any intervention
  in Phase 2 is validated against the **compiled baseline** at the *tight* tier
  (`rtol=1e-4, atol=1e-5`): passing strided views changes no arithmetic, so agreement must be
  fp32-accumulation-grade; a looser result is a finding, not a tolerance problem.
  Write `evidence/validation_*.json` with `max_abs_diff` per seed.
- **Timing:** 50 iterations latency (5a) + one 50-iteration free-running block. Report median
  latency and throughput ms/pass. Five repeats of the whole block; report min-of-medians and
  the spread. Record clocks alongside.
- Dump Inductor's generated code: `TORCH_COMPILE_DEBUG=1` or
  `torch._inductor.config.debug=True`; keep the `output_code.py` for the baseline in
  `evidence/output_code_baseline/`. You will read it in Phase 1.

Deliverable of Phase 0: `evidence/baseline_timing.json`, `evidence/validation_baseline.json`,
`evidence/output_code_baseline/`, `evidence/env.json`. If validation fails at the loose tier,
**stop** and report — nothing downstream is meaningful.

---

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

---

## 4. Phase 1 — Attribution (no code changes to the model)

**1a. Trace.** `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop
--cuda-graph-trace=node -o evidence/trace_baseline` around the 5a latency block of the compiled
baseline, with the `p8_trace.py` NVTX hooks extended per layer/sublayer (time-attn, freq-attn,
rearrange boundaries, FFN). Export `cuda_gpu_kern_sum` **and** `nvtx_gpu_proj_sum` CSVs.

**1b. Classify every kernel row** into: `GEMM` (cuBLAS/cutlass names), `SDPA`
(`flash`/`fmha`/`cutlass…attention`), `INDUCTOR_FUSED` (triton_* names), `COPY` (memcpy,
`copy_`, `clone`, `contiguous`, `elementwise_kernel` copies), `FFT`, `OTHER`. Write the
classifier as a pure function over kernel-name strings; include its source and a 20-row
labeled sample in `evidence/` so the owner can spot-check.

**1c. Map Inductor kernels to their fused ops.** Inductor names encode the fusion
(`triton_poi_fused_clone_…`, `triton_per_fused__to_copy_add_native_layer_norm_…`). For every
`INDUCTOR_FUSED` kernel: from `output_code_baseline/`, extract (i) the fused op list, (ii)
whether any *input* is read through a permuted stride pattern (non-unit stride on the
fastest-varying dim, or `as_strided`/`permute` upstream of it), (iii) whether it is a pure
materialization (a `clone`/`copy` with no arithmetic). Produce
`evidence/inductor_kernel_map.csv` with columns:
`kernel_name, nvtx_range, fused_ops, is_pure_copy, reads_permuted, bytes_moved, duration_us,
achieved_GBps`. `bytes_moved` = sum of input+output tensor bytes from the shapes in the
wrapper code. `achieved_GBps = bytes_moved / duration`.

**1d. Count materializations per layer** by reading `output_code`: list every place a permuted
tensor is materialized (explicit `.contiguous()`/`clone`/`copy_`, or an `empty_strided` output
with a different stride order than its producer). Report per sublayer: number of
materializations, their tensor sizes, and which op consumes the result (SDPA, extern mm,
norm). **State explicitly whether the head-split/merge reshapes around SDPA are views or
copies** — this alone separates P-L1's 3–5 % branch from its ≥ 10 % branch.

**1e. Compute `R_layout`** as the sum of:
- (A) duration of all `COPY` kernels and pure-copy `INDUCTOR_FUSED` kernels whose NVTX range is
  in the transformer stack and whose fused ops contain a layout change; plus
- (B) the *penalty* portion of non-pure fused kernels that read permuted inputs:
  `duration − bytes_moved / BW_ref`, where `BW_ref` is the achieved bandwidth of the same
  kernel class on contiguous inputs (measure it: run three representative fused kernels'
  op-chains standalone on contiguous vs permuted-view inputs at identical shapes; this is the
  P-L2 test). Clamp penalty at ≥ 0.
- divided by the 5a median latency of the compiled baseline.

Report `R_layout` with (A) and (B) separately, and a per-layer breakdown.

**Stop condition:** if `R_layout < 0.03`, Phase 2 is skipped (write "NO-GO by threshold" and
still complete 1a–1e's evidence). If a step in 1c/1d cannot be completed from the artifacts,
mark `R_layout` as a lower bound `(A)` only and say so.

---

## 5. Phase 2 — Minimal intervention (existence proof; only if `R_layout ≥ 0.03`)

Goal: remove materializations **without writing a kernel**, via an external wrapper.

- Identify the attention forward in the clone (the `Attend`/attention module and the axial
  wrappers). Implement, *in this repo*, a subclass or replacement `forward` that:
  1. keeps the activation in one canonical layout for norms/QKV/out-proj/FFN;
  2. builds q/k/v for each axis as **strided views** (`permute`/`view`, no `.contiguous()`),
     with head_dim contiguous;
  3. calls `F.scaled_dot_product_attention` inside `torch.nn.attention.sdpa_kernel([...])`
     pinning the backend to the one the baseline used (read it from the trace: flash vs
     mem-efficient); if that backend rejects the strided layout, record the error and try the
     other backend — document which accepted;
  4. writes the output back in the canonical layout via a strided view, not a copy.
- Swap it in with the same fail-closed pattern as `swap_banded_modules`: build all replacements
  first, commit only if all succeed; any exception leaves the stock model untouched.
- Compile the patched model with the identical C1 call. Dump `output_code_patched/`.
- **Verify it did what you think:** re-run 1a–1d on the patched model. The materialization count
  per sublayer must fall; if a `.contiguous()` reappears inside SDPA's dispatch (visible as a
  copy kernel adjacent to the attention kernel), the intervention failed silently — report it
  as such, do not report the timing.
- Validate tight-tier vs compiled baseline (3 seeds); time per 5a, five repeats.
- `R_recovered = (latency_baseline − latency_patched) / (R_layout × latency_baseline)`.
  Also report the raw delta in ms and as % of forward.

Phase 2 must also try, **before** the wrapper, the P-L4 knobs on the *unpatched* baseline —
each as a separate compiled run with its own `output_code` dump and materialization count:
`torch._inductor.config.layout_optimization` (both values), `keep_output_stride`,
`permute_fusion` (if present in this torch), `mode="max-autotune"` and
`"max-autotune-no-cudagraphs"`. `flag_recovers = True` if any single knob reduces (A)+(B) by
≥ 60 % with validation passing. If `flag_recovers` is True, say so in the first line of
`RESULT.md`: it converts the lane into a configuration ticket.

---

## 6. `RESULT.md` — fixed format

```
# Axial-layout scoping — RESULT
repo: <hash>   sm120-nulltest tooling @ <hash>   msst @ <commit>   ckpt sha256 <…>
torch <ver> · triton <ver> · driver <ver> · GPU <name> · sustained clocks <sm/mem MHz>

## Decision inputs (mechanical)
R_layout        = <0.xxx>   (A)=<0.xxx> (B)=<0.xxx>   [evidence/… , evidence/…]
R_recovered     = <0.xx> | SKIPPED (R_layout<0.03) | FAILED (<reason>)
flag_recovers   = True/False (<knob>) | UNTESTED (<reason>)
head reshapes materialized: Yes/No  [output_code line refs]
baseline compiled latency (5a median, min-of-5): <ms>   throughput: <ms/pass>
patched  compiled latency (5a median, min-of-5): <ms>   delta: <ms> (<%>)
validation: baseline vs eager (loose) PASS/FAIL per seed; patched vs baseline (tight) PASS/FAIL per seed, max_abs_diff = […]

## Threshold application (do not editorialize; just apply)
GO      if R_layout ≥ 0.05 AND R_recovered ≥ 0.60 AND flag_recovers == False
TICKET  if 0.03 ≤ R_layout < 0.05, OR flag_recovers == True, OR (R_layout ≥ 0.05 AND R_recovered < 0.60)
NO-GO   if R_layout < 0.03
Result: <GO | TICKET | NO-GO>  — computed from the numbers above.

## Predictions vs measurements
P-L1 … P-L4: predicted / measured / status (CONFIRMED | FALSIFIED | UNRESOLVED)

## Per-layer attribution table  (from evidence/inductor_kernel_map.csv)
## Materialization inventory   (from output_code; per sublayer)
## What failed / could not be determined
## Files
```

The first line under "Decision inputs" is the sentence the owner reads; everything else is
evidence for the audit.

---

## 7. Owner audit checklist (15 minutes; the agent must make each item checkable)

1. `PREDICTIONS.md` commit timestamp precedes every file in `evidence/` (git log + mtimes).
2. Pick 5 random rows of `inductor_kernel_map.csv`; open `output_code_baseline/` at the cited
   kernel; confirm `fused_ops`, `reads_permuted`, and `is_pure_copy` are correct.
3. Recompute `R_layout` from the CSV with one `awk`/pandas line; it must match `RESULT.md`.
4. In the patched trace, confirm no copy kernel sits adjacent to the SDPA kernel inside a
   freq-attention NVTX range (the silent-`.contiguous()` failure mode).
5. Validation JSONs: seeds present, tiers as specified, `max_abs_diff` values plausible
   (tight tier for patched vs baseline: expect ≤ 1e-4).
6. Clock log: sustained SM clock during timed blocks within 5 % across baseline and patched
   runs (otherwise the delta is a thermal artifact, not a layout result).
7. The threshold section's `Result` follows mechanically from the three numbers.

If items 1–7 pass and `Result = GO`, the owner opens the research lane (layout propagation
through opaque compiler boundaries; generality study on a second axial model). If `TICKET`, the
Phase 2 wrapper ships to production as-is and the lane closes. If `NO-GO`, the study is filed as
a negative result with its evidence and nothing further is built.
