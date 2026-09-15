"""Create/check one baseline generation; never rebind retained evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def current(repo):
    import yaml
    msst = Path(os.environ['MSST_DIR'])
    config = Path(os.environ['CONFIG_FILE'])
    ckpt = Path(os.environ['CKPT_FILE'])
    tooling = repo.parent / 'sm120-nulltest'
    pin = [x for x in (tooling / 'configs/msst_commit.txt').read_text().splitlines()
           if x and not x.startswith('#')][0]
    ckpt_spec = [x for x in (tooling / 'configs/checkpoint.txt').read_text().splitlines()
                 if x and not x.startswith('#')]
    if git(msst, 'rev-parse', 'HEAD') != pin or git(msst, 'status', '--porcelain'):
        raise RuntimeError('public model clone differs from the clean pinned source')
    if config.resolve() != (msst / ckpt_spec[2]).resolve():
        raise RuntimeError('configuration path is not the registered checkpoint configuration')
    ckpt_sha = sha(ckpt)
    if ckpt_sha != ckpt_spec[1]:
        raise RuntimeError('checkpoint SHA256 differs from the registered value')
    amp = yaml.load(config.read_text(), Loader=yaml.FullLoader)['training']['use_amp']
    if not isinstance(amp, bool):
        raise TypeError('use_amp is not an explicit boolean')
    paths = sorted([*repo.glob('scoping/*.py'), *repo.glob('scripts/run*.sh'),
                    repo / 'scripts/beast_env.sh'])
    hashes = {str(p.relative_to(repo)): sha(p) for p in paths}
    code_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'code_sha256': code_sha, 'code_files': hashes,
            'ckpt_sha256': ckpt_sha, 'config': str(config), 'config_sha256': sha(config),
            'use_amp': amp, 'msst_commit': pin,
            'tooling_commit': git(tooling, 'rev-parse', 'HEAD'),
            'repo_commit': git(repo, 'rev-parse', 'HEAD')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['create', 'check'])
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--runs', required=True, type=Path)
    args = ap.parse_args()
    cur = current(args.repo)
    path = args.runs / 'manifest.json'
    if args.action == 'create':
        if git(args.repo, 'status', '--porcelain'):
            raise RuntimeError('measurement requires committed, clean study code')
        if path.exists() or any(args.runs.glob('phase*')):
            dst = args.runs.with_name('runs_archive_' + datetime.now().strftime('%Y%m%d_%H%M%S')
                                     + '_' + uuid.uuid4().hex[:8])
            args.runs.rename(dst)
            print(f'Preserved previous generation at {dst}')
        args.runs.mkdir(parents=True, exist_ok=True)
        cur.update(generation_id=str(uuid.uuid4()),
                   created_utc=datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(cur, indent=1) + '\n')
        print(f"Created generation {cur['generation_id']}")
    else:
        old = json.loads(path.read_text())
        for key in ('code_sha256', 'ckpt_sha256', 'config_sha256', 'config',
                    'use_amp', 'msst_commit', 'tooling_commit'):
            if old.get(key) != cur[key]:
                raise RuntimeError(f'{key} changed since baseline; archive and restart Phase 0')
        print(f"Verified generation {old['generation_id']}")


if __name__ == '__main__':
    main()
