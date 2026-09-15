"""Audit a completed trace when graph breaks prevent wrapper attribution.

This produces a partial kernel catalog, not a replacement R_layout estimate.
Generic CUDA-graph staging copies cannot be relabeled as axis permutations.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scoping'))
from classify import classify
from kernel_map import parse_output_code, read_gpu_trace, nvtx_pass_windows, split_passes, read_kern_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('evidence', type=Path)
    args = ap.parse_args()
    ev = args.evidence
    root = ev / 'raw'
    p = root / 'phase1'
    manifest = json.loads((root / 'manifest.json').read_text())
    meta = json.loads((p / 'trace_run_meta.json').read_text())
    exports = json.loads((p / 'trace_baseline_exports.json').read_text())
    assert exports['status'] == 'ok' and all(exports['reports'].values())
    assert exports['generation_id'] == meta['generation_id'] == manifest['generation_id']
    trace = read_gpu_trace(str(p / 'trace_baseline_cuda_gpu_trace.csv'))
    windows, error = nvtx_pass_windows(str(p / 'trace_baseline_nvtx_gpu_proj_trace.csv'))
    assert not error and len(windows) == meta['iters_completed'] == 50
    passes, method = split_passes(trace, windows, classify)
    assert sum(map(len, passes)) == len(trace)
    assert all(sum(classify(r['name']) == 'SDPA' for r in block) == 12 for block in passes)
    graphs = parse_output_code(str(p / 'dump_baseline'))
    source = collections.defaultdict(list)
    sites = []
    sdpa_calls = 0
    for g in graphs:
        relative = str(Path(g['path']).relative_to(ev))
        for c in g['calls']:
            sdpa_calls += 'scaled_dot_product' in c['name']
            if c['kind'] == 'triton':
                k = g['kernels'].get(c['name'], {})
                row = {'kernel_name': c['name'], 'source_file': relative,
                       'source_line': c['source_line'], 'fused_ops': ';'.join(k.get('ops', [])),
                       'ops_source': k.get('ops_source'), 'arguments': json.dumps(c['args'])}
                sites.append(row)
                source[c['name']].append(row)
    with open(p / 'generated_kernel_sites.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sites[0]))
        w.writeheader()
        w.writerows(sites)
    durations = collections.Counter()
    counts = collections.Counter()
    inside = collections.Counter()
    for block in passes:
        sdpa = [r for r in block if classify(r['name']) == 'SDPA']
        for r in block:
            durations[r['name']] += r['dur']
            counts[r['name']] += 1
            if sdpa[0]['start'] <= r['start'] <= sdpa[-1]['start']:
                inside[r['name']] += r['dur']
    aggregate = read_kern_sum(str(p / 'trace_baseline_cuda_gpu_kern_sum.csv'))
    mismatches = [n for n, total in aggregate.items() if durations.get(n) != total]
    assert not mismatches, mismatches
    all_rows, mapped = [], []
    for name, duration in durations.most_common():
        klass = classify(name)
        all_rows.append({'class': klass, 'kernel_name': name, 'launches': counts[name],
                         'duration_us_per_pass': duration / len(passes) / 1000})
        if klass != 'INDUCTOR_FUSED':
            continue
        variants = source[name]
        ops = sorted({v['fused_ops'] for v in variants})
        mapped.append({'kernel_name': name, 'nvtx_range': 'pass_*; sublayer UNRESOLVED',
                       'fused_ops': ' | '.join(ops) or 'UNRESOLVED (no source match)',
                       'is_pure_copy': str(bool(ops) and all(o == 'aten.clone' for o in ops)),
                       'reads_permuted': 'UNRESOLVED (executed graph variant not identified)',
                       'bytes_moved': 'UNRESOLVED', 'duration_us': duration / len(passes) / 1000,
                       'achieved_GBps': 'UNRESOLVED',
                       'source_refs': ';'.join(f"{v['source_file']}:{v['source_line']}" for v in variants),
                       'status': 'PARTIAL: no compiled SDPA anchors; not an A/B ledger'})
    for filename, rows in (('all_kernel_classes.csv', all_rows),
                           ('inductor_kernel_map_partial.csv', mapped)):
        with open(p / filename, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    clone_in_stack = sum(v for n, v in inside.items() if 'clone' in n and classify(n) == 'INDUCTOR_FUSED')
    copy_in_stack = sum(v for n, v in inside.items() if classify(n) == 'COPY')
    report = {'status': 'UNRESOLVED', 'R_layout': None, 'A_ms': None, 'B_ms': None,
              'reason': 'No compiled SDPA anchors; generic copies cannot prove layout materialization',
              'generation_id': manifest['generation_id'], 'use_amp': manifest['use_amp'],
              'trace_passes': len(passes), 'gpu_events': len(trace),
              'events_per_pass': sorted(set(map(len, passes))), 'sdpa_per_pass': 12,
              'sdpa_backend': 'pytorch_flash', 'output_code_files': len(graphs),
              'compiled_sdpa_calls': sdpa_calls, 'aggregate_crosscheck_mismatches': mismatches,
              'inductor_kernel_names_in_trace': len(mapped),
              'copy_class_in_attention_window_ms_DIAGNOSTIC': copy_in_stack / len(passes) / 1e6,
              'compiled_clone_in_attention_window_ms': clone_in_stack / len(passes) / 1e6,
              'nonzero_A_lower_bound_established': False,
              'class_duration_ms_per_pass_DIAGNOSTIC': {
                  klass: sum(v for n, v in durations.items() if classify(n) == klass) / len(passes) / 1e6
                  for klass in ('GEMM', 'SDPA', 'INDUCTOR_FUSED', 'COPY', 'FFT', 'OTHER')},
              'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (p / 'attribution_audit.json').write_text(json.dumps(report, indent=1) + '\n')
    link = ev / 'inductor_kernel_map.csv'
    if link.is_symlink():
        link.unlink()
    link.symlink_to('raw/phase1/inductor_kernel_map_partial.csv')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
