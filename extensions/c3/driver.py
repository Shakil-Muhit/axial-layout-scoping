"""Qualify/time or trace the registered C3 arm using the existing 5a code."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scoping'))
import torch
import build_model as bm
import common_timing as ct
from evidence_io import read_json, write_json
from graphbreak_fix import install
import native_tail

CHUNK = 352800


def counters():
    from torch._dynamo.utils import counters as values
    return {str(k): {str(name): int(count) for name, count in v.items()}
            for k, v in values.items()}


def graph_gate(out, name):
    snapshot = counters()
    breaks = sum(snapshot.get('graph_break', {}).values())
    unsupported = sum(snapshot.get('unimplemented', {}).values())
    ok = (breaks == 0 and unsupported == 0 and
          snapshot.get('stats', {}).get('unique_graphs', 0) > 0 and
          not torch._dynamo.config.suppress_errors)
    record = {'status': 'PASS' if ok else 'FAILED',
              'graph_break_count': breaks, 'unimplemented_count': unsupported,
              'suppress_errors': torch._dynamo.config.suppress_errors,
              'TORCH_LOGS': os.environ.get('TORCH_LOGS'), 'counters': snapshot}
    write_json(out / f'graph_gate_{name}.json', record)
    if not ok:
        raise SystemExit('C3 graph-break gate FAILED; dependent work stopped')


def validate(compiled, model, ctx, refs, out):
    records = {kind: {'tier': tier, 'seeds': {}}
               for kind, tier in (('eager', 'loose'), ('c1', 'tight'))}
    with torch.no_grad(), ctx():
        for seed in ct.VAL_SEEDS:
            x = ct.make_input(CHUNK, 2 if model.stereo else 1, seed)
            y = compiled(x).clone()
            torch.cuda.synchronize()
            torch.save(y.float().cpu(), out / f'ref_c3_{seed}.pt')
            for kind, record in records.items():
                source = 'baseline' if kind == 'c1' else 'eager'
                ref = torch.load(refs / f'ref_{source}_{seed}.pt',
                                 map_location='cuda', weights_only=True)
                record['seeds'][str(seed)] = ct.compare_outputs(y, ref, record['tier'])
                del ref
            del x, y
    for kind, record in records.items():
        record['all_pass'] = all(v['ok'] for v in record['seeds'].values())
        write_json(out / f'validation_vs_{kind}.json', record)
    if not records['eager']['all_pass']:
        raise SystemExit('C3 loose validation FAILED; no timing or attribution accepted')
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('eager-check', 'qualify', 'trace'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--refs', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'trace':
        qualification = read_json(args.out.parent / 'qualification/validation_vs_eager.json')
        assert qualification['all_pass']
    paths = bm.paths_from_env()
    model, config, amp, ctx = bm.load_public_model(
        paths['msst'], paths['ckpt'], paths['config'])
    installed = install(model)
    write_json(args.out / 'source_verification.json', installed)
    tail = native_tail.install(model, args.out)
    write_json(args.out / 'native_tail_verification.json', tail)
    write_json(args.out / 'env.json', {**ct.gpu_env(os.environ.get('NSYS_PATH')), 'use_amp': amp})
    if args.mode == 'eager-check':
        write_json(args.out / 'native_tail_opcheck.json', native_tail.verify_operator())
        eager_checks = {}
        with torch.no_grad(), ctx():
            for seed in ct.VAL_SEEDS:
                y = model(ct.make_input(CHUNK, 2 if model.stereo else 1, seed))
                ref = torch.load(args.refs / f'ref_eager_{seed}.pt',
                                 map_location='cuda', weights_only=True)
                rec = ct.compare_outputs(y, ref, 'tight')
                rec['bitwise_equal'] = bool(torch.equal(y.float(), ref.float()))
                eager_checks[str(seed)] = rec
                torch.save(y.float().cpu(), args.out / f'ref_c3_eager_{seed}.pt')
                del y, ref
        write_json(args.out / 'validation_external_eager.json', {
            'tier': 'tight', 'seeds': eager_checks,
            'all_pass': all(v['ok'] for v in eager_checks.values())})
        if not all(v['ok'] for v in eager_checks.values()):
            raise SystemExit('external eager method changed outputs at tight tier')
        return  # keep eager rotary-cache state out of both compiled processes
    torch._dynamo.reset()
    from torch._dynamo.utils import counters as values
    values.clear()
    compiled = bm.compile_baseline_c1(model)
    x = ct.make_input(CHUNK, 2 if model.stereo else 1, 4242)
    settle = ct.settle_compile(compiled, x, ctx)
    write_json(args.out / 'settle.json', settle)
    ct.require_settled(settle)
    graph_gate(args.out, 'settled')
    validation = validate(compiled, model, ctx, args.refs, args.out)
    graph_gate(args.out, 'validated')
    final_settle = ct.settle_compile(compiled, x, ctx)
    ct.require_settled(final_settle)
    before = counters()
    if args.mode == 'qualify':
        timing = ct.five_repeats(compiled, x, ctx, iters=50, repeats=5)
        timing.update(settle=settle, settle_after_validation=final_settle,
                      use_amp=amp, input_seed=4242, chunk=CHUNK,
                      channels=2 if model.stereo else 1,
                      compile_call="torch.compile(model, mode='reduce-overhead', dynamic=False)",
                      arm='C3_external_asdict_repair')
        write_json(args.out / 'timing.json', timing)
    else:
        n = 0
        def step(t):
            nonlocal n
            torch.cuda.nvtx.range_push(f'pass_{n}')
            try:
                compiled(t)
            finally:
                torch.cuda.nvtx.range_pop()
                n += 1
        lat, thr = ct.timed_run(step, x, ctx, iters=50)
        write_json(args.out / 'trace_run_meta.json', {
            'iters_completed': len(lat), 'latency_ms_per_iter': lat,
            'latency_median_ms': ct.median(lat), 'throughput_ms_per_pass': thr,
            'settle': settle, 'settle_after_validation': final_settle,
            'note': 'nsys-attached trace; no headline latency from this process'})
    after = counters()
    write_json(args.out / 'counter_stability.json', {
        'status': 'PASS' if before == after else 'FAILED', 'before': before, 'after': after})
    assert before == after, 'compiler counters changed during measurement'
    graph_gate(args.out, 'finished')


if __name__ == '__main__':
    main()
