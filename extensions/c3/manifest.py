"""Bind the extension to committed code and the retained C1 references."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scoping'))
from run_manifest import current, git, sha

PREDICTION_COMMIT = '9b1fefdefb4bc9962f2be4f867e77e1905b627df'
C1_GENERATION = 'b23aca5d-3315-42c4-832f-8668c8b36ceb'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('create', 'check'))
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--c1', type=Path, required=True)
    args = ap.parse_args()
    run = current(args.repo)
    if git(args.repo, 'status', '--porcelain'):
        raise RuntimeError('extension requires committed, clean study code')
    additions = [*(p for p in args.repo.glob('extensions/c3/*') if p.is_file()),
                 args.repo / 'PREDICTIONS_EXTENSION_C3.md']
    run['extension_files'] = {str(p.relative_to(args.repo)): sha(p)
                              for p in sorted(additions)}
    run['prediction_commit'] = PREDICTION_COMMIT
    run['prediction_committed_unix'] = int(git(args.repo, 'show', '-s', '--format=%ct',
                                               PREDICTION_COMMIT))
    original = json.loads((args.c1 / 'manifest.json').read_text())
    assert original['generation_id'] == C1_GENERATION
    for key in ('ckpt_sha256', 'config_sha256', 'msst_commit', 'tooling_commit', 'use_amp'):
        assert original[key] == run[key], key
    refs = [args.c1 / 'manifest.json', args.c1 / 'phase0/baseline_timing.json',
            args.c1 / 'phase0/validation_baseline.json']
    refs += [args.c1 / f'phase0/ref_{kind}_{seed}.pt'
             for kind in ('eager', 'baseline') for seed in (4242, 1337, 90210)]
    run['c1_generation_id'] = C1_GENERATION
    run['c1_references'] = {str(p.relative_to(args.c1)): sha(p) for p in refs}
    path = args.out / 'manifest.json'
    if args.action == 'create':
        assert not path.exists(), 'extension generation already exists'
        args.out.mkdir(parents=True, exist_ok=True)
        for p in refs:
            dest = args.out / 'c1_reference' / p.relative_to(args.c1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        run.update(generation_id=str(uuid.uuid4()),
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   arm='C3_external_asdict_repair', gpu_session_limit_seconds=3600)
        path.write_text(json.dumps(run, indent=1, allow_nan=False) + '\n')
    else:
        old = json.loads(path.read_text())
        for key, value in run.items():
            assert old[key] == value, f'manifest drift: {key}'
        for rel, expected in old['c1_references'].items():
            assert sha(args.out / 'c1_reference' / rel) == expected, rel
    print(f"{args.action}: {path}")


if __name__ == '__main__':
    main()
