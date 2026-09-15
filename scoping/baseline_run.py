"""Phase 0 driver (handoff section 2).

Modes (one process per mode; the orchestrator sequences them):
  eager-ref  : stock eager forward, 3 validation seeds -> saved fp32 outputs
  baseline   : C1-compiled model. Settle (TORCH_LOGS=recompiles in env),
               loose-tier validation vs the saved eager refs, 5x(50-iter 5a
               latency + 50-iter free-running) timing, baseline outputs saved
               (they are the tight-tier reference for Phase 2), env.json.
  trace-iters: compiled model, settle, then exactly ONE 5a block (the nsys
               capture range wraps its latency half) — used by Phase 1a under
               nsys; timing from this mode is never the headline number.

Outputs land in --out-dir (ml-beast local disk; the orchestrator syncs the
curated set into evidence/).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import common_timing as ct  # noqa: E402
import build_model as bm  # noqa: E402

CHUNK = 352800  # handoff: the offline operating point; fixed, do not vary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['eager-ref', 'baseline', 'trace-iters'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--settle-max', type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    assert torch.cuda.is_available()

    p = bm.paths_from_env()
    model, config, use_amp, actx = bm.load_public_model(
        p['msst'], p['ckpt'], p['config'])
    channels = 2 if bool(getattr(model, 'stereo', False)) else 1
    print(f'[phase0] mode={args.mode} use_amp={use_amp} channels={channels}')
    env = ct.gpu_env(os.environ.get('NSYS_PATH'))
    env['use_amp'] = use_amp
    ct.write_json(os.path.join(args.out_dir, 'env.json'), env)

    if args.mode == 'eager-ref':
        with torch.no_grad(), actx():
            for seed in ct.VAL_SEEDS:
                x = ct.make_input(CHUNK, channels, seed)
                out = model(x)
                torch.cuda.synchronize()
                torch.save(out.float().cpu(),
                           os.path.join(args.out_dir, f'ref_eager_{seed}.pt'))
                print(f'  eager ref seed {seed}: {tuple(out.shape)}')
        ct.write_json(os.path.join(args.out_dir, 'eager_ref_meta.json'),
                      {'use_amp': use_amp, 'seeds': list(ct.VAL_SEEDS),
                       'chunk': CHUNK, 'channels': channels})
        return

    compiled = bm.compile_baseline_c1(model)
    x_time = ct.make_input(CHUNK, channels, 4242)   # handoff: seed 4242
    step = lambda t: compiled(t)  # noqa: E731

    settle = ct.settle_compile(step, x_time, actx, max_iters=args.settle_max)
    print(f'[phase0] settled: {settle["settle_iters"]} passes, '
          f'capped={settle["settle_capped"]}')
    ct.require_settled(settle, os.path.join(args.out_dir, 'settle_failed.json'))

    if args.mode == 'trace-iters':
        # single 5a block; the latency half sits in the nsys capture range.
        # NVTX pass ranges are pushed EAGERLY around the compiled call (outer
        # side of dynamo — no graph perturbation) so nvtx_gpu_proj_sum has
        # per-pass ranges (handoff 1a) and pass-splitting has hard anchors.
        counter = {'i': 0}
        def step_traced(t):
            torch.cuda.nvtx.range_push(f"pass_{counter['i']}")
            try:
                compiled(t)
            finally:
                torch.cuda.nvtx.range_pop()
                counter['i'] += 1
        lat, thr = ct.timed_run(step_traced, x_time, actx, args.iters)
        ct.write_json(os.path.join(args.out_dir, 'trace_run_meta.json'),
                      {'use_amp': use_amp, 'iters_completed': len(lat),
                       'latency_ms_per_iter': lat,
                       'latency_median_ms': ct.median(lat),
                       'throughput_ms_per_pass': thr, 'settle': settle,
                       'note': 'nsys-attached run; walls inflated by tracing '
                               '— never a headline number'})
        return

    # -------- mode == baseline --------
    validation = {'tier': 'loose', 'use_amp': use_amp, 'seeds': {}}
    with torch.no_grad(), actx():
        for seed in ct.VAL_SEEDS:
            x = ct.make_input(CHUNK, channels, seed)
            out = compiled(x)
            torch.cuda.synchronize()
            out = out.clone()   # borrow discipline: own before the next replay
            ref = torch.load(os.path.join(args.out_dir, f'ref_eager_{seed}.pt'),
                             map_location='cuda')
            validation['seeds'][str(seed)] = ct.compare_outputs(out, ref, 'loose')
            # baseline outputs = the tight-tier reference for Phase 2
            torch.save(out.float().cpu(),
                       os.path.join(args.out_dir, f'ref_baseline_{seed}.pt'))
            del ref
    validation['all_pass'] = all(v['ok'] for v in validation['seeds'].values())
    ct.write_json(os.path.join(args.out_dir, 'validation_baseline.json'), validation)
    if not validation['all_pass']:
        print('FATAL: loose-tier validation FAILED — handoff section 2: stop; '
              'nothing downstream is meaningful.', file=sys.stderr)
        ct.write_json(os.path.join(args.out_dir, 'baseline_timing.json'),
                      {'status': 'NOT_RUN', 'reason': 'loose validation failed'})
        sys.exit(2)

    timing = ct.five_repeats(step, x_time, actx, args.iters, args.repeats)
    timing.update({'use_amp': use_amp, 'input_seed': 4242, 'chunk': CHUNK,
                   'channels': channels, 'settle': settle,
                   'compile_call':
                       "torch.compile(model, mode='reduce-overhead', dynamic=False)"
                       ' [verbatim arm C1, sm120-nulltest@a997e27 arms/common.py]'})
    ct.write_json(os.path.join(args.out_dir, 'baseline_timing.json'), timing)
    print(f'[phase0] min-of-medians = {timing["min_of_medians_ms"]:.3f} ms | '
          f'spread = {timing["median_spread_ms"]:.3f} ms')


if __name__ == '__main__':
    main()
