"""Phase 2 pre-step — P-L4 knob sweep on the UNPATCHED baseline (handoff §5).

Each knob runs in its own process (Inductor config is process-global).
--mode screen (default): apply the knob, compile, settle (capped settle =>
UNTESTED, recorded, exit 0), validate — TIGHT vs the Phase-0 COMPILED-
baseline outputs (handoff §2: "any intervention in Phase 2" — this is the
acceptance tier for flag_recovers; gate-02 finding 10) plus LOOSE vs eager as
a diagnostic — then ONE 5a block. The orchestrator sets TORCH_COMPILE_DEBUG*
so each knob gets an output_code dump; scoping/inventory.py diffs it vs
baseline. A knob whose inventory CHANGES gets a follow-up --mode trace run
under nsys (gate-02 finding 12) so its (A)+(B) can be computed by kernel_map
before any flag_recovers claim.

A knob absent from this torch build is recorded UNTESTED(absent) — handoff:
`permute_fusion` (if present in this torch) — and the process exits 0.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import common_timing as ct  # noqa: E402
import build_model as bm  # noqa: E402

CHUNK = 352800

# knob -> (inductor-config overrides by dotted path, torch.compile mode)
KNOBS = {
    'layout_opt_true': {'cfg': {'layout_optimization': True},
                        'mode': 'reduce-overhead'},
    'layout_opt_false': {'cfg': {'layout_optimization': False},
                         'mode': 'reduce-overhead'},
    'keep_output_stride_false': {'cfg': {'keep_output_stride': False},
                                 'mode': 'reduce-overhead'},
    'permute_fusion_true': {'cfg': {'permute_fusion': True},
                            'mode': 'reduce-overhead'},
    'persistent_reductions_false': {'cfg': {'triton.persistent_reductions': False},
                                    'mode': 'reduce-overhead'},
    'max_autotune': {'cfg': {}, 'mode': 'max-autotune'},
    'max_autotune_no_cudagraphs': {'cfg': {},
                                   'mode': 'max-autotune-no-cudagraphs'},
}


def resolve_cfg(root, dotted):
    """('triton.persistent_reductions') -> (config.triton, 'persistent_reductions');
    returns (None, None) if any hop is absent."""
    obj = root
    parts = dotted.split('.')
    for p in parts[:-1]:
        if not hasattr(obj, p):
            return None, None
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        return None, None
    return obj, parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--knob', required=True, choices=sorted(KNOBS))
    ap.add_argument('--mode', default='screen', choices=['screen', 'trace'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--refs-dir', required=True,
                    help='Phase-0 out dir (ref_eager_*.pt + ref_baseline_*.pt)')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--settle-max', type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    assert torch.cuda.is_available()
    spec = KNOBS[args.knob]

    import torch._inductor.config as ind_cfg
    record = {'knob': args.knob, 'mode': spec['mode'], 'run_mode': args.mode,
              'cfg_defaults': {}, 'cfg_applied': {}}
    result_name = ('knob_result.json' if args.mode == 'screen'
                   else 'knob_trace_result.json')
    for dotted, val in spec['cfg'].items():
        obj, attr = resolve_cfg(ind_cfg, dotted)
        if obj is None:
            record['status'] = (f'UNTESTED: torch._inductor.config.{dotted} '
                                'absent in this torch')
            ct.write_json(os.path.join(args.out_dir, result_name), record)
            print(f"[knob {args.knob}] {record['status']}")
            return
        record['cfg_defaults'][dotted] = getattr(obj, attr)
        setattr(obj, attr, val)
        record['cfg_applied'][dotted] = val

    p = bm.paths_from_env()
    model, config, use_amp, actx = bm.load_public_model(
        p['msst'], p['ckpt'], p['config'])
    channels = 2 if bool(getattr(model, 'stereo', False)) else 1
    compiled = torch.compile(model, mode=spec['mode'], dynamic=False)
    x_time = ct.make_input(CHUNK, channels, 4242)
    step = lambda t: compiled(t)  # noqa: E731

    settle = ct.settle_compile(step, x_time, actx, max_iters=args.settle_max)
    record['settle'] = settle
    print(f"[knob {args.knob}] settled: {settle['settle_iters']} passes, "
          f"capped={settle['settle_capped']}")
    if settle['settle_capped']:
        # a knob that never settles is UNTESTED, not a crash (gate-02 f8/f12)
        record['status'] = 'UNTESTED: compile did not settle within cap'
        ct.write_json(os.path.join(args.out_dir, result_name), record)
        return

    if args.mode == 'trace':
        # follow-up attribution run under nsys (inventory changed): NVTX pass
        # ranges + ONE 5a block, same shape as baseline trace-iters.
        counter = {'i': 0}
        def step_traced(t):
            torch.cuda.nvtx.range_push(f"pass_{counter['i']}")
            try:
                compiled(t)
            finally:
                torch.cuda.nvtx.range_pop()
                counter['i'] += 1
        lat, thr = ct.timed_run(step_traced, x_time, actx, args.iters)
        record.update({'status': 'ok', 'use_amp': use_amp,
                       'iters_completed': len(lat),
                       'latency_ms_per_iter': lat,
                       'latency_median_ms': ct.median(lat),
                       'throughput_ms_per_pass': thr,
                       'note': 'nsys-attached knob run; never a headline'})
        ct.write_json(os.path.join(args.out_dir, 'trace_run_meta_knob.json'),
                      record)   # kernel_map --trace-meta compatible
        ct.write_json(os.path.join(args.out_dir, result_name), record)
        return

    # -------- mode == screen --------
    validation = {'acceptance_tier': 'tight vs compiled baseline (handoff §2)',
                  'use_amp': use_amp, 'seeds': {}}
    with torch.no_grad(), actx():
        for seed in ct.VAL_SEEDS:
            x = ct.make_input(CHUNK, channels, seed)
            out = compiled(x)
            torch.cuda.synchronize()
            out = out.clone()
            ref_b = torch.load(
                os.path.join(args.refs_dir, f'ref_baseline_{seed}.pt'),
                map_location='cuda')
            rec = ct.compare_outputs(out, ref_b, 'tight')
            del ref_b
            ref_e = torch.load(
                os.path.join(args.refs_dir, f'ref_eager_{seed}.pt'),
                map_location='cuda')
            rec['loose_vs_eager'] = ct.compare_outputs(out, ref_e, 'loose')
            del ref_e, out
            validation['seeds'][str(seed)] = rec
    validation['tight_all_pass'] = all(
        v['ok'] for v in validation['seeds'].values())
    validation['loose_all_pass'] = all(
        v['loose_vs_eager']['ok'] for v in validation['seeds'].values())

    lat, thr = ct.timed_run(step, x_time, actx, args.iters)   # ONE 5a block
    record.update({
        'status': 'ok', 'use_amp': use_amp,
        'validation': validation,
        'latency_median_ms': ct.median(lat),
        'latency_ms_per_iter': lat,
        'throughput_ms_per_pass': thr,
        'note': 'screening run (one 5a block, TORCH_COMPILE_DEBUG on): the '
                'flag verdict rests on the materialization-inventory diff; '
                'a knob whose inventory changes gets a --mode trace follow-up '
                'and kernel_map (A)+(B) before any flag_recovers claim',
    })
    ct.write_json(os.path.join(args.out_dir, result_name), record)
    print(f"[knob {args.knob}] median {record['latency_median_ms']:.3f} ms | "
          f"tight={validation['tight_all_pass']} "
          f"loose_vs_eager={validation['loose_all_pass']}")


if __name__ == '__main__':
    main()
