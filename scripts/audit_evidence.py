"""Independently recompute saved-output validation and timing arithmetic.

Run on the GPU host's venv, on CPU only, after measurement completes. This
does not call the measurement driver's comparison or timing functions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', type=Path)
    args = ap.parse_args()
    root = args.runs
    manifest = json.loads((root / 'manifest.json').read_text())
    p0 = root / 'phase0'
    v = json.loads((p0 / 'validation_baseline.json').read_text())
    t = json.loads((p0 / 'baseline_timing.json').read_text())
    assert v['generation_id'] == t['generation_id'] == manifest['generation_id']
    checks = {}
    for seed in (4242, 1337, 90210):
        a = torch.load(p0 / f'ref_baseline_{seed}.pt', map_location='cpu', weights_only=True)
        b = torch.load(p0 / f'ref_eager_{seed}.pt', map_location='cpu', weights_only=True)
        assert a.shape == b.shape == (1, 2, 352800)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        error = (a - b).abs()
        failed = int((error > 1e-3 + 1e-2 * b.abs()).sum())
        maximum = float(error.max())
        assert failed == 0 and v['seeds'][str(seed)]['ok'] is True
        assert maximum == v['seeds'][str(seed)]['max_abs_diff']
        checks[str(seed)] = {'max_abs_diff': maximum, 'failed_elements': failed,
                             'shape': list(a.shape), 'finite': True}
    blocks = t['blocks']
    assert len(blocks) == 5 and all(len(b['latency_ms_per_iter']) == 50 for b in blocks)
    medians = [statistics.median(b['latency_ms_per_iter']) for b in blocks]
    assert medians == t['latency_medians_ms']
    assert min(medians) == t['min_of_medians_ms']
    assert max(medians) - min(medians) == t['median_spread_ms']
    assert min(b['throughput_ms_per_pass'] for b in blocks) == t['throughput_ms_per_pass_best']
    sm, mem = [], []
    with open(p0 / 'clocks_sampled_phase0.csv', newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            stamp = row[0].strip()
            try:
                whole, _, frac = stamp.partition('.')
                epoch = time.mktime(time.strptime(whole, '%Y/%m/%d %H:%M:%S'))
                epoch += float('0.' + frac) if frac else 0
                if any(b['unix_start'] <= epoch <= b['unix_end'] for b in blocks):
                    sm.append(int(row[1].strip().split()[0]))
                    mem.append(int(row[2].strip().split()[0]))
            except ValueError:
                continue
    code_files = list((p0 / 'dump_baseline').rglob('output_code.py'))
    compiled_sdpa = []
    for path in code_files:
        for num, line in enumerate(path.read_text().splitlines(), 1):
            if 'scaled_dot_product' in line:
                compiled_sdpa.append({'file': str(path.relative_to(root)), 'line': num})
    result = {'status': 'PASS', 'generation_id': manifest['generation_id'],
              'use_amp': manifest['use_amp'], 'validation_recomputed': checks,
              'latency_medians_ms': medians, 'min_of_medians_ms': min(medians),
              'throughput_ms_per_pass_best': t['throughput_ms_per_pass_best'],
              'median_spread_ms': t['median_spread_ms'],
              'clock_samples_inside_timed_blocks': len(sm),
              'sm_mhz_median': statistics.median(sm) if sm else None,
              'sm_mhz_range': [min(sm), max(sm)] if sm else None,
              'memory_mhz_median': statistics.median(mem) if mem else None,
              'output_code_files': len(code_files), 'compiled_sdpa_mentions': compiled_sdpa,
              'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / 'independent_audit.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
