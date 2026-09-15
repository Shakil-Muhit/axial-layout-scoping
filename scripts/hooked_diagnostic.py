"""Protocol-hook cross-check after the ordinal attribution path failed.

Fixed C1 compile and public model; exact original P8 hook implementation.
Accept annotations only if the trace inventory matches the unhooked run
and tight numerical comparison passes. This run is not headline timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scoping'))
import torch
import build_model as bm
import common_timing as ct
from p8_nvtx_hooks import install_nvtx_hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--refs', required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = bm.paths_from_env()
    model, config, amp, ctx = bm.load_public_model(p['msst'], p['ckpt'], p['config'])
    handles, mode = install_nvtx_hooks(model)
    compiled = bm.compile_baseline_c1(model)
    x = ct.make_input(352800, 2 if model.stereo else 1, 4242)
    settle = ct.settle_compile(compiled, x, ctx)
    ct.require_settled(settle, args.out / 'settle_failed.json')
    validation = {'use_amp': amp, 'tier': 'tight', 'seeds': {}}
    with torch.no_grad(), ctx():
        for seed in ct.VAL_SEEDS:
            y = compiled(ct.make_input(352800, 2 if model.stereo else 1, seed)).clone()
            torch.cuda.synchronize()
            ref = torch.load(args.refs / f'ref_baseline_{seed}.pt', map_location='cuda', weights_only=True)
            validation['seeds'][str(seed)] = ct.compare_outputs(y, ref, 'tight')
            del y, ref
    validation['all_pass'] = all(v['ok'] for v in validation['seeds'].values())
    ct.write_json(args.out / 'validation.json', validation)
    settle = ct.settle_compile(compiled, x, ctx)
    ct.require_settled(settle, args.out / 'settle_failed.json')
    counter = 0
    def step(t):
        nonlocal counter
        torch.cuda.nvtx.range_push(f'pass_{counter}')
        try:
            compiled(t)
        finally:
            torch.cuda.nvtx.range_pop()
            counter += 1
    lat, thr = ct.timed_run(step, x, ctx, iters=50)
    ct.write_json(args.out / 'trace_meta.json', {
        'use_amp': amp, 'iters_completed': len(lat), 'latency_ms_per_iter': lat,
        'latency_median_ms': ct.median(lat), 'throughput_ms_per_pass': thr,
        'settle': settle, 'boundary_mode': mode,
        'tight_validation_pass': validation['all_pass'],
        'status': 'DIAGNOSTIC ONLY; inventory equivalence not established',
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for h in handles:
        h.remove()


if __name__ == '__main__':
    main()
