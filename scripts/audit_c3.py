"""Independent CPU audit of the bounded C3 outputs, timing, and trace."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
import torch


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--repo', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    sys.path.insert(0, str(args.repo / 'scoping'))
    from classify import classify
    from kernel_map import (parse_output_code, read_gpu_trace, read_kern_sum,
                            nvtx_pass_windows, split_passes)
    manifest = json.loads((root / 'manifest.json').read_text())
    def read(rel):
        obj = json.loads((root / rel).read_text())
        assert obj['generation_id'] == manifest['generation_id'], rel
        assert obj['use_amp'] is True, rel
        return obj
    for rel, expected in manifest['c1_references'].items():
        assert sha(root / 'c1_reference' / rel) == expected, rel
    comparisons = {}
    for directory, source in [('eager_check', 'c3_eager'),
                              ('qualification', 'c3'), ('phase1', 'c3')]:
        checks = {}
        for seed in (4242, 1337, 90210):
            out = torch.load(root / directory / f'ref_{source}_{seed}.pt',
                             map_location='cpu', weights_only=True)
            seed_checks = {}
            pairs = [('eager', 'eager', 1e-2, 1e-3), ('c1', 'baseline', 1e-4, 1e-5)]
            if directory == 'eager_check':
                pairs = [('external_eager', 'eager', 1e-4, 1e-5)]
            for label, reference, rtol, atol in pairs:
                ref = torch.load(root / 'c1_reference/phase0' / f'ref_{reference}_{seed}.pt',
                                 map_location='cpu', weights_only=True)
                assert out.shape == ref.shape == (1, 2, 352800)
                assert torch.isfinite(out).all() and torch.isfinite(ref).all()
                error = (out - ref).abs()
                failures = int((error > atol + rtol * ref.abs()).sum())
                filename = ('validation_external_eager.json' if directory == 'eager_check'
                            else f'validation_vs_{label}.json')
                recorded = read(f'{directory}/{filename}')['seeds'][str(seed)]
                assert recorded['rtol'] == rtol and recorded['atol'] == atol
                assert recorded['ok'] == (failures == 0)
                assert recorded['max_abs_diff'] == float(error.max())
                seed_checks[label] = {'failed_elements': failures,
                                      'max_abs_diff': float(error.max()),
                                      'bitwise_equal': bool(torch.equal(out, ref))}
            if directory == 'phase1':
                ref = torch.load(root / 'qualification' / f'ref_c3_{seed}.pt',
                                 map_location='cpu', weights_only=True)
                diff = (out-ref).abs()
                failed = int((diff > 1e-5 + 1e-4 * ref.abs()).sum())
                assert failed == 0, 'trace differs from timed C3 at tight tier'
                seed_checks['timed_c3'] = {'failed_elements': failed,
                                          'max_abs_diff': float(diff.max()),
                                          'bitwise_equal': bool(torch.equal(out, ref))}
            checks[str(seed)] = seed_checks
        comparisons[directory] = checks
    graph_checks = {}
    for directory in ('qualification', 'phase1'):
        g = [read(f'{directory}/graph_gate_{stage}.json')
             for stage in ('settled', 'validated', 'finished')]
        assert all(x['status'] == 'PASS' and x['graph_break_count'] == 0 and
                   x['unimplemented_count'] == 0 and not x['suppress_errors'] for x in g)
        log = (root / directory / 'driver.log').read_text()
        assert not re.search(r'\[__graph_breaks\]|Graph break in user code', log)
        stable = read(f'{directory}/counter_stability.json')
        assert stable['status'] == 'PASS' and stable['before'] == stable['after']
        graph_checks[directory] = {'unique_graphs': g[-1]['counters']['stats']['unique_graphs'],
                                   'graph_breaks': 0, 'log_graph_break_markers': 0,
                                   'counter_stability': True}
    timing = read('qualification/timing.json')
    blocks = timing['blocks']
    assert len(blocks) == 5 and all(len(b['latency_ms_per_iter']) == 50 for b in blocks)
    medians = [statistics.median(b['latency_ms_per_iter']) for b in blocks]
    assert medians == timing['latency_medians_ms']
    assert min(medians) == timing['min_of_medians_ms']
    assert max(medians)-min(medians) == timing['median_spread_ms']
    assert min(b['throughput_ms_per_pass'] for b in blocks) == timing['throughput_ms_per_pass_best']
    clocks = []
    with open(root / 'clocks.csv', newline='') as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            whole, _, fraction = row[0].strip().partition('.')
            stamp = time.mktime(time.strptime(whole, '%Y/%m/%d %H:%M:%S'))
            stamp += float('0.' + fraction) if fraction else 0
            if any(b['unix_start'] <= stamp <= b['unix_end'] for b in blocks):
                clocks.append({'sm_mhz': int(row[1].strip().split()[0]),
                               'memory_mhz': int(row[2].strip().split()[0]),
                               'power_w': float(row[4].strip().split()[0])})
    assert clocks, 'no clock samples in unprofiled timing blocks'
    trace = read_gpu_trace(str(root / 'phase1/trace_c3_cuda_gpu_trace.csv'))
    aggregate = read_kern_sum(str(root / 'phase1/trace_c3_cuda_gpu_kern_sum.csv'))
    windows, error = nvtx_pass_windows(str(root / 'phase1/trace_c3_nvtx_gpu_proj_trace.csv'))
    meta = read('phase1/trace_run_meta.json')
    exports = read('phase1/trace_c3_exports.json')
    assert exports['status'] == 'ok' and all(exports['reports'].values())
    assert not error and len(windows) == meta['iters_completed'] == 50
    passes, method = split_passes(trace, windows, classify)
    assert sum(map(len, passes)) == len(trace)
    assert all(sum(classify(r['name']) == 'SDPA' for r in p) == 12 for p in passes)
    counts, durations = collections.Counter(), collections.Counter()
    for row in trace:
        counts[row['name']] += 1
        durations[row['name']] += row['dur']
    assert all(durations[name] == duration for name, duration in aggregate.items())
    parsed = {}
    for directory in ('qualification', 'phase1'):
        graphs = parse_output_code(str(root / directory / 'active_dump'))
        anchors = [c for g in graphs for c in g['calls'] if c['kind'] == 'fallback' and 'scaled_dot_product' in c['name']]
        parsed[directory] = {'graphs': len(graphs), 'sdpa_anchors': len(anchors),
                             'wrapper_calls': sum(len(g['calls']) for g in graphs)}
        assert len(anchors) == 12, parsed[directory]
    classes = [{'class': classify(n), 'kernel_name': n, 'launches': counts[n],
                'duration_us_per_pass': durations[n] / 50 / 1000}
               for n in sorted(durations, key=durations.get, reverse=True)]
    with open(root / 'phase1/all_kernel_classes.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(classes[0]))
        writer.writeheader()
        writer.writerows(classes)
    c1 = json.loads((root / 'c1_reference/phase0/baseline_timing.json').read_text())
    finish = int((root / 'finished_unix.txt').read_text())
    first = int((root / 'extension_first_started_unix.txt').read_text())
    assert finish-first <= 3600
    assert manifest['prediction_committed_unix'] < first
    result = {'status': 'PASS', 'generation_id': manifest['generation_id'], 'use_amp': True,
              'validation_recomputed': comparisons, 'graph_checks': graph_checks,
              'latency_medians_ms': medians, 'min_of_medians_ms': min(medians),
              'median_spread_ms': max(medians)-min(medians),
              'throughput_ms_per_pass_best': timing['throughput_ms_per_pass_best'],
              'historical_c1_ms': c1['min_of_medians_ms'],
              'historical_latency_reduction_fraction': 1-min(medians)/c1['min_of_medians_ms'],
              'clock_samples': len(clocks),
              'sm_mhz_median': statistics.median(r['sm_mhz'] for r in clocks),
              'sm_mhz_range': [min(r['sm_mhz'] for r in clocks),max(r['sm_mhz'] for r in clocks)],
              'memory_mhz_median': statistics.median(r['memory_mhz'] for r in clocks),
              'trace_passes': len(passes), 'trace_events': len(trace),
              'events_per_pass': sorted(set(map(len, passes))),
              'sdpa_per_pass': 12, 'generated_code': parsed,
              'aggregate_crosscheck': 'PASS', 'gpu_window_elapsed_seconds': finish-first,
              'audit_script_sha256': sha(Path(__file__))}
    (root / 'independent_audit.json').write_text(json.dumps(result, indent=1, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'validation_recomputed'}, indent=1))


if __name__ == '__main__':
    main()
