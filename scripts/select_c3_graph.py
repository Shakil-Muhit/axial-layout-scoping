"""Identify the executed C3 wrapper by every traced Triton multiplicity.

Both the cold rotary-cache graph and its replacement contain SDPA. Counting
both is invalid. Accept exactly one source graph whose complete Triton
inventory matches every captured pass; preserve all original dumps.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scoping'))
from classify import classify
from kernel_map import parse_output_code, read_gpu_trace, nvtx_pass_windows, split_passes


def select(root, directory, observed):
    candidates = []
    paths = {}
    for graph in parse_output_code(str(root / directory / 'dump')):
        calls = collections.Counter(c['name'] for c in graph['calls'] if c['kind'] == 'triton')
        anchors = [c for c in graph['calls'] if c['kind'] == 'fallback' and
                   'scaled_dot_product' in c['name']]
        differences = {n: {'wrapper': calls[n], 'trace': observed[n]}
                       for n in sorted(set(calls) | set(observed)) if calls[n] != observed[n]}
        path = Path(graph['path'])
        rel = str(path.relative_to(root))
        candidates.append({'source': rel, 'triton_calls': sum(calls.values()),
                           'sdpa_anchors': len(anchors), 'differences': differences,
                           'accepted': not differences and len(anchors) == 12})
        paths[rel] = path
    accepted = [c for c in candidates if c['accepted']]
    assert len(accepted) == 1, f'{directory}: no unique complete source/trace match'
    chosen = accepted[0]
    destination = root / directory / 'active_dump'
    assert not destination.exists(), 'do not overwrite a previous selection'
    shutil.copytree(paths[chosen['source']].parent, destination)
    digest = hashlib.sha256(paths[chosen['source']].read_bytes()).hexdigest()
    assert hashlib.sha256((destination / 'output_code.py').read_bytes()).hexdigest() == digest
    return {'candidates': candidates, 'selected_source': chosen['source'],
            'selected_sha256': digest, 'active_copy': str(destination.relative_to(root)),
            'selection_rule': 'unique exact all-Triton inventory match in every pass; 12 SDPA calls'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    root = args.root
    run = json.loads((root/'manifest.json').read_text())
    trace = read_gpu_trace(str(root/'phase1/trace_c3_cuda_gpu_trace.csv'))
    windows, error = nvtx_pass_windows(str(root/'phase1/trace_c3_nvtx_gpu_proj_trace.csv'))
    assert not error and len(windows) == 50
    passes, method = split_passes(trace, windows, classify)
    inventories = [collections.Counter(r['name'] for r in p
                                       if classify(r['name']) == 'INDUCTOR_FUSED') for p in passes]
    assert all(c == inventories[0] for c in inventories)
    assert all(sum(classify(r['name']) == 'SDPA' for r in p) == 12 for p in passes)
    selected = {d: select(root, d, inventories[0]) for d in ('qualification','phase1')}
    result = {'status': 'PASS', 'generation_id': run['generation_id'], 'use_amp': run['use_amp'],
              'passes': len(passes), 'triton_events_per_pass': sum(inventories[0].values()),
              'all_pass_inventories_identical': True, 'selection': selected,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'active_graph_selection.json').write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps({d: s['selected_source'] for d,s in selected.items()},indent=1))


if __name__ == '__main__':
    main()
