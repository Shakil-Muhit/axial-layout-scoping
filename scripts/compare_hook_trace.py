"""Apply the predeclared equivalence gate to the supplemental hook trace."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scoping'))
from classify import classify
from kernel_map import read_gpu_trace, nvtx_pass_windows, split_passes
from finalize import validation_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', type=Path)
    args = ap.parse_args()
    root = args.raw
    hook = root / 'hook_diagnostic'
    baseline = read_gpu_trace(str(root / 'phase1/trace_baseline_cuda_gpu_trace.csv'))
    observed = read_gpu_trace(str(hook / 'trace_cuda_gpu_trace.csv'))
    windows, error = nvtx_pass_windows(str(hook / 'trace_nvtx_gpu_proj_trace.csv'))
    assert not error and len(windows) == 50
    passes, _ = split_passes(observed, windows, classify)
    assert sum(map(len, passes)) == len(observed)
    assert all(sum(classify(r['name']) == 'SDPA' for r in block) == 12 for block in passes)
    a = collections.Counter(r['name'] for r in baseline)
    b = collections.Counter(r['name'] for r in observed)
    differences = [{'kernel_name': name, 'baseline_launches': a[name],
                    'hooked_launches': b[name]} for name in sorted(a.keys() | b.keys())
                   if a[name] != b[name]]
    with open(hook / 'kernel_inventory_diff.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['kernel_name', 'baseline_launches', 'hooked_launches'])
        w.writeheader()
        w.writerows(differences)
    v = json.loads((hook / 'validation.json').read_text())
    m = json.loads((root / 'manifest.json').read_text())
    assert v['generation_id'] == m['generation_id']
    with open(hook / 'trace_nvtx_gpu_proj_trace.csv', newline='') as f:
        labels = sorted({r['Name'] for r in csv.DictReader(f) if 'p8_' in r['Name']})
    valid = validation_state(v, 'tight') is True
    same = a == b
    report = {'generation_id': m['generation_id'], 'use_amp': m['use_amp'],
              'status': 'ACCEPT' if valid and same and labels else 'REJECT',
              'exact_kernel_inventory_match': same, 'tight_validation_pass': valid,
              'baseline_events': len(baseline), 'hooked_events': len(observed),
              'baseline_events_per_pass': len(baseline) / 50,
              'hooked_events_per_pass': len(observed) / 50,
              'changed_kernel_name_counts': len(differences),
              'failing_seeds': [k for k, x in v['seeds'].items() if not x['ok']],
              'nvtx_labels': labels,
              'rule': 'Both exact inventory equivalence and tight validation must pass',
              'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (hook / 'equivalence_audit.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
