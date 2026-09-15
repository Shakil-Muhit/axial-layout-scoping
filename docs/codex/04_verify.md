# Codex review — 04_verify

model: gpt-6-astra · effort: ultra · sandbox: read-only · 2026-09-15T17:57:38+06:00

CODE-BLOCKERS: 7

Reviewed the supplied source and accepted the reported host checks. Local execution remained unavailable because sandbox startup failed at `bwrap`. The blockers below concern acceptance leaks, not the permitted ordinal attribution heuristic.

| Finding | Status |
|---|---|
| f19 | CLOSED — metadata uses the wrapper symbol; generic inner names are covered. |
| f20 | PARTIALLY-CLOSED — unresolved access semantics can still escape coverage flags. |
| f21 | PARTIALLY-CLOSED — output traffic and timing-fallback uncertainty do not gate BW_ref acceptance. |
| f22 | CLOSED — accepted under the stated scoping allowance. |
| f23 | PARTIALLY-CLOSED — pass/name checks improved; missing aggregate evidence still passes. |
| f24 | CLOSED — recovery gates and emitted key names match. |
| f25 | PARTIALLY-CLOSED — incomplete no-change screens can still produce `False`. |
| f26 | PARTIALLY-CLOSED — reaching producers improved; the merge verdict stops too early. |
| f27 | CLOSED — follow-up mapping uses the current trace process’s dump and status. |
| f28 | PARTIALLY-CLOSED — the manifest can still rebind retained evidence. |
| f29 | PARTIALLY-CLOSED — report failures are recorded but not consumed by acceptance. |
| f30 | PARTIALLY-CLOSED — failure finalization can accept a stale patched map. |
| f31 | CLOSED — exact ADAPTATION contents and ordered unique markers are enforced. |

32. **BLOCKER — incomplete access analysis becomes “not permuted.”** [kernel_map.py:690](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/kernel_map.py:690), [inventory.py:51](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/inventory.py:51).

    Matching a load against **any** store is insufficient for multiple-output kernels: an unrelated output with the same index expression can hide a permutation feeding another output. An empty parsed store set also suppresses detection without flagging uncertainty.

    Missing `in_out_ptr` loads bypass the missing-load counter. Conversely, a store-only `in_out_ptr` is always charged as a reader, overstating ideal traffic and understating B. These paths can leave `status/B_status='ok'`.

    **Minimal fix:** make access classification tri-state; flag absent or ambiguous expressions and output correspondence before deciding relevance. Derive actual reads/writes for `in_out_ptr`, and share completeness with inventory. The fixture currently asserts that its store-only `in_out_ptr0` is a reader, so `FIXTURE-OK` does not validate these semantics.

33. **BLOCKER — unresolved BW_ref correspondence remains acceptance-grade.** [microbench_p_l2.py:151](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/microbench_p_l2.py:151).

    Every chain prices traffic as two fp16 tensors. The norm chain has no cast restoring fp16 after normalization; fp32 output is expected from the documented CUDA autocast behavior. Its recorded output dtype never affects pricing or acceptance. [PyTorch AMP reference](https://docs.pytorch.org/docs/2.8/amp.html#cuda-ops-that-can-autocast-to-float32)

    Separately, `eager-event-blocks` fallback remains eligible for `B_status='ok'`: mapping reads only `BW_ref_GBps`, ignoring the recorded timing limitation.

    **Minimal fix:** enforce the modeled boundary dtype and corresponding traffic, or exclude the class. Propagate unresolved timing/reference correspondence into `FLAGGED` output; retaining diagnostic measurements is within scope.

34. **BLOCKER — a first consumer is accepted as proof of a view-only head merge.** [kernel_map.py:514](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/kernel_map.py:514).

    `SDPA → gate/mul → clone → projection` receives `merge_verdict='true'` at the gate/mul consumer; scanning stops before the clone. A Triton clone is `INDUCTOR_FUSED`, so COPY-adjacency inspection does not repair this. Other removed copies can satisfy `in_stack_A_fell`, allowing numeric recovery despite the surviving merge materialization.

    **Minimal fix:** keep the verdict unresolved until the path through gating to projection is established; mark an intervening copy false. Propagate unresolved merge ancestry into map status, as already done for split ancestry.

35. **BLOCKER — knob acceptance still bypasses explicit incompleteness.** [finalize.py:219](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/finalize.py:219).

    With a baseline map already `FLAGGED`, all seven successful screens can have equal inventory totals and resolve to boolean `False`. This branch ignores baseline attribution acceptance and inventory unknown-operation/coverage information. The shell suppresses follow-up tracing on the same comparison.

    Missing tight-validation evidence is also treated as validation failure. Explicit failed validation is a legitimate disqualification; missing validation is unresolved. The positive branch additionally omits `baseline_evidence_ok`.

    **Minimal fix:** require complete evidence for accepted no-change screens; otherwise trace or return `UNTESTED`. Distinguish missing validation from explicit failure, and require accepted baseline prerequisites for positive recovery.

36. **BLOCKER — Phase 0 can rebind retained downstream evidence.** [beast_env.sh:69](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/beast_env.sh:69).

    Re-running Phase 0 with unchanged measuring code but changed checkpoint overwrites the manifest while retaining old maps/screens. Subsequent manifest checks accept the newly stamped checkpoint, and Phase 2 reuses the old artifacts. Config contents remain unbound because only their path is checked.

    **Minimal fix:** invalidate/archive downstream evidence when creating a new baseline generation; bind reused artifacts to that generation and hash config contents. An allowed drift must invalidate headline acceptance.

37. **BLOCKER — failure finalization can accept fresh timing with a stale map.** [run_phase2.sh:53](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_phase2.sh:53), [run_phase2.sh:170](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_phase2.sh:170).

    On a rerun, successful patched timing/validation and current clock samples can precede a failed trace/export/map. Only the dump directory was cleared; an earlier `r_layout_patched.json` survives. `finalize_and_fail` passes no failure state to finalization, so that old map can satisfy every attribution gate and produce numeric `R_recovered`.

    **Minimal fix:** invalidate attempt-specific maps before rerunning, and pass the failed step into an explicit acceptance gate for the affected headline.

38. **BLOCKER — required export incompleteness does not reach acceptance.** [beast_env.sh:125](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/beast_env.sh:125), [kernel_map.py:616](/mnt/Muhit/Home/ms/axial-layout-scoping/scoping/kernel_map.py:616).

    Successful NVTX TRACE plus failed SUM writes a marker but still permits accepted output; neither mapping nor finalization consumes it. Likewise, an unreadable/header-only kernel summary can yield an empty aggregate map, whose missing totals are skipped without a status flag.

    **Minimal fix:** propagate required-report completeness into map acceptance. Require a successfully parsed kernel summary with relevant-name coverage, and clear stale exports before retrying.

39. **NONBLOCKER — automatic archiving loses the Phase-0 sampler directory.** [run_phase0.sh:8](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_phase0.sh:8), [beast_env.sh:76](/mnt/Muhit/Home/ms/axial-layout-scoping/scripts/beast_env.sh:76).

    `OUT` is created before `write_manifest` can move `RUNS`. The sampler then starts before the Python driver recreates `OUT`. **Fix:** initialize output/log paths after archiving. Missing clock evidence rejects recovery, so this is not counted above.

The requested bookkeeping checks are sound: every exclusion-counter increment sets its CSV exclusion reason; missing A duration in the B exclusion bucket is conservative. The missing-class and contiguous-copy checks now exclude invalid references. Finalizer keys match, including `sdpa_adjacent_copy_events_rep_pass`. `clocks_stop` is safe twice under `set -u`, and Phase-0/1 function calls are consistent.

No clean first-run shell crash was identified. The acceptance fixes change the measuring-code fingerprint, so Phase 0 should freeze the manifest after those fixes.

EXECUTION-READY: no
---
exit: 0
