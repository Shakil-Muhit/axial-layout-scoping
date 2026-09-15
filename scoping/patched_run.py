"""Phase 2 driver — the strided-view intervention (handoff section 5).

Modes (one process per mode; orchestrated by scripts/run_phase2.sh):
  cpu-parity : validation-ladder step (i), docs/design_notes.md — CPU fp32,
               chunk 4410, RANDOM weights (no checkpoint): patched eager vs
               stock eager on 3 seeds, expected max|diff| <= 1e-5.
  gpu        : load public model, swap_axial_forward, compile with the
               IDENTICAL C1 call, settle, TIGHT-tier validation vs the
               Phase-0 compiled-baseline outputs (ref_baseline_<seed>.pt),
               5a timing (five repeats) -> patched_timing.json.
  trace-iters: swapped + compiled, settle, ONE 5a block with eager NVTX pass
               ranges — run under nsys with TORCH_COMPILE_DEBUG for the
               output_code_patched dump + the re-run of 1a-1d (handoff §5
               "verify it did what you think").
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import common_timing as ct  # noqa: E402
import build_model as bm  # noqa: E402
import axial_patch  # noqa: E402

CHUNK = 352800
PARITY_CHUNK = 4410
PARITY_TOL = 1e-5   # design notes: same ops, GEMM batching differs -> tiny drift


def cpu_parity(out_dir, p):
    """Stock vs patched EAGER forward, CPU fp32, random weights."""
    sys.path.insert(0, p['msst'])
    from utils.settings import get_model_from_config
    torch.manual_seed(7)   # deterministic random init
    model, config = get_model_from_config('mel_band_roformer', p['config'])
    model = model.eval()
    channels = 2 if bool(getattr(model, 'stereo', False)) else 1
    res = {'chunk': PARITY_CHUNK, 'channels': channels, 'device': 'cpu fp32',
           'use_amp': False,
           'weights': 'random init (seed 7), no checkpoint',
           'tolerance_max_abs': PARITY_TOL, 'seeds': {}}
    with torch.no_grad():
        refs = {}
        for seed in ct.VAL_SEEDS:
            refs[seed] = model(ct.make_input(PARITY_CHUNK, channels, seed,
                                             device='cpu'))
        unswap = axial_patch.swap_axial_forward(model)
        try:
            for seed in ct.VAL_SEEDS:
                out = model(ct.make_input(PARITY_CHUNK, channels, seed,
                                          device='cpu'))
                d = float((out - refs[seed]).abs().max().item())
                res['seeds'][str(seed)] = {'max_abs_diff': d,
                                           'ok': d <= PARITY_TOL}
                print(f'  parity seed {seed}: max|diff| = {d:.3e}')
        finally:
            unswap()
    res['all_pass'] = all(v['ok'] for v in res['seeds'].values())
    ct.write_json(os.path.join(out_dir, 'parity_cpu.json'), res)
    if not res['all_pass']:
        print('FATAL: CPU parity failed — the patch is NOT math-identical; '
              'do not proceed to GPU runs.', file=sys.stderr)
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['cpu-parity', 'gpu', 'trace-iters'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--baseline-dir', default=None,
                    help='Phase-0 out dir holding ref_baseline_<seed>.pt '
                         '(required for --mode gpu)')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--settle-max', type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    p = bm.paths_from_env()

    if args.mode == 'cpu-parity':
        cpu_parity(args.out_dir, p)
        return

    assert torch.cuda.is_available()
    model, config, use_amp, actx = bm.load_public_model(
        p['msst'], p['ckpt'], p['config'])
    channels = 2 if bool(getattr(model, 'stereo', False)) else 1
    unswap = axial_patch.swap_axial_forward(model)   # noqa: F841 — fail-closed
    print(f'[phase2] forward swapped (pinned sha ok) | mode={args.mode} '
          f'use_amp={use_amp}')

    compiled = bm.compile_baseline_c1(model)   # IDENTICAL C1 call (handoff §5)
    x_time = ct.make_input(CHUNK, channels, 4242)
    step = lambda t: compiled(t)  # noqa: E731
    settle = ct.settle_compile(step, x_time, actx, max_iters=args.settle_max)
    print(f'[phase2] settled: {settle["settle_iters"]} passes, '
          f'capped={settle["settle_capped"]}')
    ct.require_settled(settle, os.path.join(args.out_dir,
                                            'settle_failed_patched.json'))

    if args.mode == 'trace-iters':
        counter = {'i': 0}
        def step_traced(t):
            torch.cuda.nvtx.range_push(f"pass_{counter['i']}")
            try:
                compiled(t)
            finally:
                torch.cuda.nvtx.range_pop()
                counter['i'] += 1
        lat, thr = ct.timed_run(step_traced, x_time, actx, args.iters)
        ct.write_json(os.path.join(args.out_dir, 'trace_run_meta_patched.json'),
                      {'use_amp': use_amp, 'iters_completed': len(lat),
                       'latency_ms_per_iter': lat,
                       'latency_median_ms': ct.median(lat),
                       'throughput_ms_per_pass': thr, 'settle': settle,
                       'note': 'nsys-attached patched run; never a headline'})
        return

    # -------- mode == gpu --------
    assert args.baseline_dir, '--baseline-dir required for --mode gpu'
    validation = {'tier': 'tight', 'use_amp': use_amp,
                  'reference': 'compiled baseline outputs (Phase 0)',
                  'seeds': {}}
    with torch.no_grad(), actx():
        for seed in ct.VAL_SEEDS:
            x = ct.make_input(CHUNK, channels, seed)
            out = compiled(x)
            torch.cuda.synchronize()
            out = out.clone()   # own before the next cudagraph replay
            ref = torch.load(
                os.path.join(args.baseline_dir, f'ref_baseline_{seed}.pt'),
                map_location='cuda')
            rec = ct.compare_outputs(out, ref, 'tight')
            rec['loose_ok'] = ct.compare_outputs(out, ref, 'loose')['ok']
            validation['seeds'][str(seed)] = rec
            torch.save(out.float().cpu(),
                       os.path.join(args.out_dir, f'out_patched_{seed}.pt'))
            del ref
    validation['all_pass'] = all(v['ok'] for v in validation['seeds'].values())
    ct.write_json(os.path.join(args.out_dir, 'validation_patched.json'),
                  validation)

    timing = ct.five_repeats(step, x_time, actx, args.iters, args.repeats)
    timing.update({'use_amp': use_amp, 'input_seed': 4242, 'chunk': CHUNK,
                   'channels': channels, 'settle': settle,
                   'patch': 'axial_patch.swap_axial_forward (strided-view '
                            'canonical-layout stack)',
                   'compile_call':
                       "torch.compile(model, mode='reduce-overhead', dynamic=False)"
                       ' [identical C1 call, handoff section 5]'})
    if not validation['all_pass']:
        # handoff §5 + design notes: a looser result is a FINDING; the wall of
        # a non-equivalent forward is diagnostic only, never a headline.
        timing['status'] = 'NOT_HEADLINE'
        timing['reason'] = 'tight-tier validation failed (see validation_patched.json)'
        print('WARNING: tight validation FAILED — patched timing recorded as '
              'diagnostic only.', file=sys.stderr)
    ct.write_json(os.path.join(args.out_dir, 'patched_timing.json'), timing)
    print(f'[phase2] patched min-of-medians = {timing["min_of_medians_ms"]:.3f} ms'
          f' | spread = {timing["median_spread_ms"]:.3f} ms')


if __name__ == '__main__':
    main()
