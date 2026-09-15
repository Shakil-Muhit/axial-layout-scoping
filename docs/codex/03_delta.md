# Codex review — 03_delta

model: gpt-6-astra · effort: ultra · sandbox: read-only · 2026-09-15T17:36:37+06:00

CODE-BLOCKERS: 7

Review of the supplied versions. The sandbox failed before starting the shell (`bwrap: Failed RTM_NEWADDR`), so I could not rerun the fixture or transcription checker. The reported ml-beast smoke results are treated as supplied evidence.

| Gate-02 # | Status |
|---|---|
| 1 | PARTIALLY-CLOSED — assignment detection improved; metadata still uses the wrong symbol. |
| 2 | PARTIALLY-CLOSED — ordinary argument order and stride-order detection fixed; reuse, accesses and dtype resolution remain incomplete. |
| 3 | PARTIALLY-CLOSED — per-class references added; precision and timing correspondence remain unverified. |
| 4 | PARTIALLY-CLOSED — boundaries narrowed; name reuse does not establish stack ownership. |
| 5 | PARTIALLY-CLOSED — pass-count mismatch stops; anchor and call-site validation remain insufficient. |
| 6 | PARTIALLY-CLOSED — ancestry/merge records added; the views verdict can still falsely pass. |
| 7 | PARTIALLY-CLOSED — CSV→JSON arithmetic is fixed; ownership and per-layer attribution remain heuristic. |
| 8 | CLOSED — capped drivers stop before measurement. |
| 9 | CLOSED — exact shape and finiteness now gate acceptance. |
| 10 | CLOSED — knobs validate tight against compiled-baseline references. |
| 11 | PARTIALLY-CLOSED — headline gates exist but accept incomplete evidence. |
| 12 | PARTIALLY-CLOSED — follow-up workflow added; both Boolean verdicts have unsupported paths. |
| 13 | PARTIALLY-CLOSED — sampling/windows added; clock comparability does not gate recovery. |
| 14 | PARTIALLY-CLOSED — code fingerprint added; reused artifacts remain unbound to their prerequisites. |
| 15 | PARTIALLY-CLOSED — threshold skip finalizes; failure reporting and final artifact assembly remain absent. |
| 16 | PARTIALLY-CLOSED — projection trace added; export failures still permit completion. |
| 17 | CLOSED — medians are consistent. |
| 18 | PARTIALLY-CLOSED — transcription/CPU evidence improved; compiled layout claims still await evidence. |

The following are new defects or concrete residual mechanisms in the revised code, numbered after gate-02.

19. **BLOCKER — metadata identity still breaks with generic inner names.** [scoping/kernel_map.py:97](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/kernel_map.py:97), continuing **#1**.

    For `triton_poi_fused_clone_0 = async_compile.triton('triton_', ...)`, the parser stores metadata under `triton_`, while the call lookup uses `triton_poi_fused_clone_0`. Repeated generic definitions overwrite that entry. This remains a supported configuration, although PyTorch 2.8 defaults to unique names. The 1,200-character comment window also retains an arbitrary metadata cutoff. [PyTorch 2.8 code generation](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/codegen/triton.py)

    **Minimal fix:** key by assignment LHS; retain the inner name separately; bound definition lookup to its enclosing source string; associate the complete metadata block or mark it unresolved.

20. **BLOCKER — missing argument/access metadata can silently corrupt B.** [scoping/kernel_map.py:170](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/kernel_map.py:170), continuing **#2**.

    Ordinary literal scalars and the supplied `out=buf1` case work. However, `align_ok` accepts extra matched tokens, `.item()` expressions can be mistaken for tensor arguments, and unsupported pointer names disappear before positional alignment.

    More directly, Inductor’s ordinary `buf7 = buf2; del buf2  # reuse` is unparsed. Unknown shape/dtype then becomes one fp16 element—**two bytes**. Unknown fp32 inputs likewise default to fp16. Underestimated bytes make the subtracted ideal time too small and **inflate B**; `dtype_unresolved` does not invalidate the result. This is not necessarily a lower bound. [PyTorch reuse generation](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/codegen/wrapper.py)

    Wrapper strides are also insufficient: fused permutation indexing can reside inside `tl.load` while `.run` receives a bare contiguous buffer. Full allocation size need not equal the region read.

    **Minimal fix:** parse complete arguments before assigning roles; support plain reuse; resolve types and effective accesses from signatures/generated code; make unsupported relevant sites unresolved. Extend the fixture with these cases, including omitted-offset views.

21. **BLOCKER — BW_ref still does not establish the corresponding device-kernel reference.** [scoping/microbench_p_l2.py:66](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/microbench_p_l2.py:66), continuing **#3**.

    The norm reference explicitly uses fp16 outside autocast and hard-codes two tensors’ traffic, despite the admitted difference from the model’s promoted normalization chain. A caveat does not close that correspondence requirement.

    Events around Python-fed blocks also include GPU queue idle whenever dispatch cannot keep up. That understates BW_ref and biases B downward. Initial calls, warmup and synchronization reasonably exclude initial compilation/autotuning; they do not exclude queue bubbles.

    **Minimal fix:** reproduce the mapped chain’s precision, casts and output layout; inspect both generated variants and their traffic; measure kernel durations or a warmed captured block. Keep unmatched classes unresolved. Captured repetition is also used by [Triton’s benchmark implementation](https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/testing.py).

22. **BLOCKER — stack boundaries remain dependent on unverified kernel deduplication.** [scoping/kernel_map.py:411](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/kernel_map.py:411), continuing **#4**.

    If kernels are not deduplicated across layers, first-layer preparation and final-layer post-attention work are excluded. That biases A/B **downward without reporting incompleteness**. Conversely, names shared with another graph can admit unrelated boundary work.

    The COPY window always excludes copies before the first and after the last SDPA, including legitimate stack preparation or head-merge copies at those boundaries.

    **Minimal fix:** establish graph-aware stack boundaries and provenance. Until then, enumerate uncertain boundary contributions and mark ownership unresolved; remove the assertion that totals are exact.

23. **BLOCKER — pass and launch attribution still accepts ambiguous evidence.** [scoping/kernel_map.py:483](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/kernel_map.py:483), continuing **#5**.

    Anchor mismatch merely adds a flag, even without `--allow-pass-mismatch`. Twelve alternating known-axis anchors are not required. Gap fallback can return `status=ok`; NVTX windows need not have unique, complete, non-overlapping pass IDs. Meanwhile, `per_name` sums **all trace rows**, rather than the validated pass union.

    Prefix matching lets missing `foo_1` borrow `foo_10`’s duration, potentially counting it twice. Equal sharing remains unverified, cross-graph collisions only increment a counter, and `--kern-sum-csv` is never read.

    **Minimal fix:** require complete passes, unambiguous kernel matches and launch multiplicities, verified site equivalence, and the aggregate cross-check. Explicitly flagged exploratory output must remain ineligible for acceptance.

24. **BLOCKER — numeric R_recovered still bypasses required evidence.** [scoping/finalize.py:114](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/finalize.py:114), continuing **#11/#13**.

    Neither map’s status/coverage nor B completeness is checked. Baseline settlement and validation are reported rather than required. Clock comparability is calculated **after** accepting the numeric result and never invalidates it.

    Consequently, a patched map flagged for missing relevant kernels can report artificially reduced A and still pass. Missing clocks or `within_5pct=False` also leave numeric recovery intact.

    **Minimal fix:** require accepted baseline evidence, complete applicable attribution, positive no-copy evidence, and available/comparable clocks before calculating recovery. Otherwise emit the specified failure string and diagnostics only.

25. **BLOCKER — flag_recovers can emit either Boolean without sufficient evidence.** [scoping/finalize.py:183](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/finalize.py:183), continuing **#12**.

    A screen can have `status='ok'` and failed tight validation. With unchanged inventory totals, it still produces `False`. Only existing knob directories are examined, so an incomplete sweep can also produce `False`. Equal aggregate counts do not establish unchanged A+B.

    Conversely, a changed knob map with `B=None` substitutes zero and can produce `True`; flagged maps are accepted too.

    **Minimal fix:** enumerate the expected knob set, require tight validation for accepted results, and require valid comparable A+B evidence. Treat missing B, incomplete sweeps and insufficient no-change evidence as `UNTESTED`.

26. **MAJOR — ancestry can use future writes, and unknown ancestry passes the views gate.** [scoping/kernel_map.py:378](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/kernel_map.py:378), continuing **#6**.

    The producer map retains the final writer to each `(graph, base)`. A later reuse/in-place write can therefore become the alleged producer of an earlier SDPA input.

    Separately, `heads_are_views` accepts unknown producers and unrecognized copies. It can remain `True` even when `merge_consumer` explicitly reports a materialized copy. Internal SDPA dispatch copies are not inspected.

    **Minimal fix:** track reaching definitions/storage versions at each call; use tri-state input and merge verdicts; require traced adjacent-copy inspection for acceptance.

27. **MAJOR — knob follow-up traces use another compilation’s dump.** [scripts/run_phase2.sh:108](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_phase2.sh:108), continuing **#12**.

    The follow-up recompiles with caches disabled but does not generate its own debug dump. Mapping uses the earlier screening dump, although autotuning may select different kernels.

    A capped trace also returns zero without replacing `trace_run_meta_knob.json`; the orchestrator never requires current `knob_trace_result.status == 'ok'`, leaving stale metadata usable.

    **Minimal fix:** generate and consume a fresh trace-process dump; clear prior trace artifacts; check current status and completed-pass metadata before export/mapping.

28. **MAJOR — the manifest does not bind resumed evidence.** [scripts/beast_env.sh:69](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/beast_env.sh:69), continuing **#14**.

    Phase 0 overwrites the global manifest while retaining old Phase-1/2 artifacts. The stored checkpoint hash is never checked; config contents and environment are unbound. Existing failed/untested screens and existing maps still cause skips.

    **Minimal fix:** attach immutable run/prerequisite identities to artifacts, including code, weights, config and environment. Reuse only matching complete records, with explicit retry handling.

29. **MAJOR — projection exports still fail open.** [scripts/beast_env.sh:96](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/beast_env.sh:96), continuing **#16**.

    The shared `ok` flag treats either report succeeding as sufficient. A missing required summary is unrecorded when trace succeeds, and vice versa. Both failures still return success; dependent attribution ignores the marker.

    **Minimal fix:** track each required report separately and fail or propagate an incomplete status consumed by mapping/finalization.

30. **MAJOR — failed outcomes still bypass final reporting.** [scripts/run_phase2.sh:137](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_phase2.sh:137), continuing **#15**.

    CPU parity, patched settlement and trace failures exit before finalization. The supplied code still does not assemble `RESULT.md` or stage the required evidence and provenance links.

    **Minimal fix:** route success, skip and failure through common reporting that preserves the failure status and explicitly identifies missing evidence.

31. **MAJOR — the transcription checker does not enforce “imports only” for D2.** [scoping/verify_transcription.py:77](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/verify_transcription.py:77), related to **#18**.

    Arbitrary contents between ADAPTATION markers are deleted before comparison. Adding `raw_audio = raw_audio * 2` there would still yield `VERBATIM-OK`.

    **Minimal fix:** require the exact permitted import block and unique ordered markers. Also declare def-line renaming, blank-line normalization and seam stripping explicitly: the checker proves equality modulo those additional normalizations, rather than literal byte equality under D1–D3 alone. Its length checks correctly prevent `zip` truncation from hiding differences.

Several scrutinized points are sound:

- The ledger now reproduces JSON A/B totals numerically. With valid positive BW_ref, bytes/(GB/s) gives nanoseconds, the zero clamp is correct, and pure-copy sites enter A **or** B. Triton-first classification prevents overlap with COPY summary rows. However, a nonempty BW dictionary currently labels B “ok” even if a required class is missing.
- Contiguous-format clone expresses the intended conversion; this packet does not prove single-kernel code generation for both variants. Assert the recorded output layout and inspect generated code. `penalty_ratio=t_p/t_c` correctly means slowdown multiplier.
- Baseline/patched exit 4 becomes shell exit 1 but still stops execution. Screening timeout 124 and crashes are distinguished correctly.
- References load sequentially. A single-stem stereo fp32 audio output is about **2.82 MB**, so these are not three model-sized activation tensors.
- The Bash heredoc is valid. Nsight profiling `timeout` is not inherently broken because executed descendants are traced. Unquoted optional-argument strings should become Bash arrays; the shown fixed paths do not trigger their whitespace/globbing problem. [NVIDIA profiling documentation](https://docs.nvidia.com/nsight-systems/UserGuide/)
- The documented NVTX projection columns match the parser, and the sampler’s timestamp/`MHz` format matches the clock parser. Cross-host finalization should record timezone or epoch timestamps. [NVTX report documentation](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html), [nvidia-smi query format](https://nvidia.custhelp.com/app/answers/detail/a_id/3751)
- The restored loss branch introduces no evident Dynamo hazard under `target=None`; the early return keeps it dead.

Phase 0’s isolated settle/validation path is now sound. Launching it as the frozen prerequisite is premature: fixing the remaining measuring code changes its manifest fingerprint and requires rerunning Phase 0 under the current policy.

EXECUTION-READY: no
---
exit: 0
