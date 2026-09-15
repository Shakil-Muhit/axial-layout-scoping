"""Create and verify a lossless evidence archive for a separate Git branch.

The main branch keeps the handoff's evidence/*.md-only versioning rule.
Archive parts stay below GitHub's individual-file size limit. The publisher
commits this output directory to the same private repository's evidence branch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise SystemExit('output directory must be empty')
    files = sorted(p for p in (args.evidence / 'raw').rglob('*') if p.is_file())
    checksums = {str(p.relative_to(args.evidence)): digest(p) for p in files}
    with tempfile.TemporaryDirectory(prefix='axial-archive-') as td:
        archive = Path(td) / 'artifacts.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(args.evidence / 'raw', arcname='raw')
            for p in sorted(args.evidence.iterdir()):
                if p.is_symlink():
                    tar.add(p, arcname=p.name, recursive=False)
        # Verify every member's bytes directly from the compressed archive.
        with tarfile.open(archive) as tar:
            for rel, expected in checksums.items():
                data = tar.extractfile(rel)
                h = hashlib.sha256()
                for block in iter(lambda: data.read(1024 * 1024), b''):
                    h.update(block)
                assert h.hexdigest() == expected, rel
        parts = []
        with archive.open('rb') as f:
            while data := f.read(40 * 1024 * 1024):
                part = args.out / f'artifacts.tar.gz.part{len(parts):03d}'
                part.write_bytes(data)
                parts.append({'name': part.name, 'bytes': len(data), 'sha256': digest(part)})
        run = json.loads((args.evidence / 'raw/manifest.json').read_text())
        info = {'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
                'parts': parts, 'files': checksums, 'file_count': len(files),
                'generation_id': run['generation_id'], 'use_amp': run['use_amp']}
        (args.out / 'ARTIFACTS.json').write_text(json.dumps(info, indent=1) + '\n')
        (args.out / 'SHA256SUMS').write_text(''.join(f"{p['sha256']}  {p['name']}\n" for p in parts))
        print(json.dumps({k: v for k, v in info.items() if k != 'files'}, indent=1))


if __name__ == '__main__':
    main()
