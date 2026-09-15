# Axial-layout scoping

**Completed report:** [RESULT.md](RESULT.md). The bounded C3 extension reaches
30.306 ms with zero graph breaks. Verified head-merge copies account for
0.887% of its forward; the full layout share remains unresolved because
hidden-cost reference coverage is incomplete. Phase 2 was not run.

Registered study of materialized layout changes in the compiled MelBandRoformer
forward at the 8-second operating point. [PREDICTIONS.md](PREDICTIONS.md) was the
first commit. [docs/handoff.md](docs/handoff.md) is the specification.

Execution uses the owner's existing `ml-beast` host, GPU 4, and its prepared
`~/axial` environment. The repository is shared there at
`/mnt/Muhit/Home/ms/axial-layout-scoping`. Git operations run on the local host
because the remote shared mount is not executable.

## Integrity and checks

```bash
git config core.hooksPath .githooks
python3 scoping/tests/test_parse_fixture.py
python3 -m unittest discover -s scoping/tests -p 'test_acceptance.py' -v
```

Commit and push the measuring code before starting. The orchestrator refuses
uncommitted code and verifies the clean public clone, checkpoint hash, config
contents, and measurement-code hashes. Each baseline attempt creates a new
generation and archives previous artifacts. Later phases refuse changed inputs.

## Execute

Run once on ml-beast, with the registered four-hour limit:

```bash
timeout --signal=TERM --kill-after=30s 14400 bash /mnt/Muhit/Home/ms/axial-layout-scoping/scripts/run_session.sh
```

Phase 0 validates the stock compiled model against eager outputs on all three
registered seeds. A loose-tier failure stops the study before timing or
attribution. Phase 1 exports node-level traces and generated code. Phase 2 runs
only when its threshold is established; incomplete attribution cannot establish
a NO-GO result merely because its lower bound is below the threshold.

```bash
bash scripts/sync_evidence.sh
```

This sync preserves partial and failure artifacts, logs, outputs, generated
code, and complete traces under `evidence/raw/`. Convenient handoff paths link
to this collection. Generated artifacts are ignored by Git as prescribed;
`evidence/*.md` and the final `RESULT.md` are versioned. The complete lossless archive is published on the evidence branch; restoration
and checksum verification are documented in [evidence/README.md](evidence/README.md).

The current acceptance rules and remaining methodological limits are recorded
in [docs/codex/05_takeover.md](docs/codex/05_takeover.md). Earlier review documents
are historical records, not claims that GPU execution passed.

## Bounded C3 extension

[Registered prediction and scope](PREDICTIONS_EXTENSION_C3.md),
[attempt notes](extensions/c3/ATTEMPT_NOTES.md), and
[reference-calibration plan](docs/c3_calibration.md) document the extension.
The external wrapper fixes backend-context tracing and preserves native
complex DC filtering behind a compiler boundary. It uses the pinned model
and original compile call; it implements no new GPU kernel.

The final runner selects the executed steady graph by its exact ordered
Triton inventory, then maps buffer metadata before reuse and identifies
layout copies from their stored values. Its postprocessing incorporates
the independently verified corrections made after the recorded run.

```bash
python3 scripts/test_c3_ledger.py -v
```

Reproduction on ml-beast requires a fresh output directory and the retained
C1 references at `~/axial/runs`. Run with a hard one-hour timeout:

```bash
timeout --signal=TERM --kill-after=30s 3600 bash /mnt/Muhit/Home/ms/axial-layout-scoping/extensions/c3/session.sh "$HOME/axial/extensions/c3_fresh"
```

The resulting A-only share does not establish a NO-GO threshold while B is
unresolved. Raw outputs, all failed attempts, and the calibration refinement
are available through [evidence restoration instructions](evidence/README.md).
