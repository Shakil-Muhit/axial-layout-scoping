"""Shared timing/settle/validation machinery.

COPIED (with adaptations noted inline) from sm120-nulltest@a997e27
arms/common.py — handoff rule 4: reuse the 5a timing protocol, the settle
detector, and the validation mechanics; cite the source. The 5a docstring is
kept verbatim in spirit: PRIMARY per-iteration latency with a synchronize
after every pass inside the cudaProfilerApi capture range; SECONDARY one
free-running block (sync, N iters, sync, total/N) outside it.
"""

import json
import statistics
import subprocess
import time

import torch
from evidence_io import write_json


def median(xs):
    """statistics.median everywhere (gate-02 finding 17: s[n//2] is the upper
    middle for even n, disagreeing with five_repeats)."""
    return statistics.median(xs)


def make_input(chunk, channels, seed, device='cuda'):
    """Deterministic input independent of device RNG: CPU generator -> cuda.
    (Pattern from sm120-nulltest arms/common.py::make_input.)"""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(1, channels, chunk, generator=gen).to(device)


def settle_compile(step, x, autocast_ctx, max_iters=60, consecutive=5):
    """Run until recompilation stops: torch._dynamo counters unchanged for
    `consecutive` passes. COPIED from sm120-nulltest@a997e27
    arms/common.py::settle_compile (str-keyed snapshot included)."""
    import torch._dynamo.utils as dynamo_utils

    def snap():
        return {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in dynamo_utils.counters.items()}

    t0 = time.perf_counter()
    prev, stable, it = snap(), 0, 0
    with torch.no_grad(), autocast_ctx():
        while stable < consecutive and it < max_iters:
            step(x)
            it += 1
            cur = snap()
            stable = stable + 1 if cur == prev else 0
            prev = cur
    torch.cuda.synchronize()
    return {'settle_iters': it, 'settle_stable_tail': stable,
            'settle_capped': stable < consecutive,
            'settle_wall_s': round(time.perf_counter() - t0, 3),
            'dynamo_counters': prev}


def require_settled(settle, out_path=None):
    """Gate-02 finding 8: a capped settle must FAIL CLOSED — nothing measured
    on a still-recompiling model is meaningful. Writes the settle record (so
    the failure is evidence) and exits 4."""
    if settle['settle_capped']:
        if out_path:
            write_json(out_path, {'status': 'SETTLE_FAILED', 'settle': settle})
        import sys
        print('FATAL: compile did not settle (recompilations still occurring '
              f'after {settle["settle_iters"]} passes) — aborting before any '
              'measurement.', file=sys.stderr)
        sys.exit(4)


def timed_run(step, x, autocast_ctx, iters=50):
    """PREREG-5a TIMING PROTOCOL — COPIED from sm120-nulltest@a997e27
    arms/common.py::timed_run, with one adaptation: the forward runs inside
    the caller-supplied autocast context (handoff rule 3 — precision mirrors
    the public inference path), matching how the baseline executes.

    PRIMARY: per-iteration latency, synchronize after every pass, inside the
    cudaProfilerApi capture range (a no-op unless nsys is attached).
    SECONDARY: one free-running block (sync, N iters, sync; total/N).
    """
    with torch.no_grad(), autocast_ctx():
        torch.cuda.synchronize()
        torch.cuda.profiler.start()              # nsys capture range: open
        latency_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            step(x)
            torch.cuda.synchronize()             # 5a PRIMARY: sync every pass
            latency_ms.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.profiler.stop()               # nsys capture range: close

        torch.cuda.synchronize()                 # 5a SECONDARY: free-running
        t0 = time.perf_counter()
        for _ in range(iters):
            step(x)
        torch.cuda.synchronize()
        throughput_ms = (time.perf_counter() - t0) * 1e3 / iters
    return latency_ms, throughput_ms


def five_repeats(step, x, autocast_ctx, iters=50, repeats=5):
    """Handoff section 2: five repeats of the whole block; min-of-medians and
    spread. Returns the full record for the evidence JSON. Wall-clock
    timestamps per block let the background clock log (handoff rule 9) be
    cross-referenced with the timed windows (gate-02 finding 13)."""
    blocks = []
    for r in range(repeats):
        t_start = time.time()
        lat, thr = timed_run(step, x, autocast_ctx, iters)
        t_end = time.time()
        s = sorted(lat)
        blocks.append({'repeat': r, 'latency_ms_per_iter': lat,
                       'latency_median_ms': median(lat),
                       'latency_min_ms': s[0], 'latency_max_ms': s[-1],
                       'throughput_ms_per_pass': thr,
                       'unix_start': t_start, 'unix_end': t_end})
    medians = [b['latency_median_ms'] for b in blocks]
    return {'blocks': blocks,
            'latency_medians_ms': medians,
            'min_of_medians_ms': min(medians),
            'median_spread_ms': max(medians) - min(medians),
            'throughput_ms_per_pass_best': min(b['throughput_ms_per_pass'] for b in blocks),
            'iters_per_block': iters, 'repeats': repeats}


# Two-tier validation gate — values from sm120-nulltest@a997e27
# arms/common.py::VAL_TOL, per the handoff section 2:
#   loose (compiled vs eager; Inductor regenerates kernels): rtol 1e-2, atol 1e-3
#   tight (patched vs compiled baseline; views change no arithmetic):
#         rtol 1e-4, atol 1e-5 — a looser result is a finding, not a knob.
TIERS = {'loose': (1e-2, 1e-3), 'tight': (1e-4, 1e-5)}
VAL_SEEDS = (4242, 1337, 90210)   # handoff section 2


def compare_outputs(out, ref, tier):
    """Gate-02 finding 9 hardening: torch.allclose broadcasts and treats
    matching infs as equal — require exact shape, finiteness of BOTH sides,
    and report an explicit reason on failure."""
    rtol, atol = TIERS[tier]
    out = out.float()
    ref = ref.float()
    rec = {'ok': False, 'rtol': rtol, 'atol': atol,
           'shape_out': list(out.shape), 'shape_ref': list(ref.shape)}
    if out.shape != ref.shape:
        rec['reason'] = 'shape mismatch'
        rec['max_abs_diff'] = None
        return rec
    fin_out = bool(torch.isfinite(out).all())
    fin_ref = bool(torch.isfinite(ref).all())
    rec['finite_out'], rec['finite_ref'] = fin_out, fin_ref
    rec['max_abs_diff'] = (float((out - ref).abs().max().item())
                           if fin_out and fin_ref else None)
    if not (fin_out and fin_ref):
        rec['reason'] = 'non-finite values present'
        return rec
    rec['ok'] = bool(torch.allclose(out, ref, rtol=rtol, atol=atol))
    if not rec['ok']:
        rec['reason'] = 'tolerance exceeded'
    return rec


def gpu_env(nsys_path=None):
    def q(query):
        try:
            return subprocess.run(
                ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader', '-i',
                 str(_physical_gpu_index())],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return 'unavailable'
    import triton
    import os
    return {'torch': torch.__version__, 'triton': triton.__version__,
            'cuda_build': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0)),
            'driver': q('driver_version'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'nsys': nsys_path or 'n/a'}


def _physical_gpu_index():
    import os
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    try:
        return int(v.split(',')[0])
    except Exception:
        return 0
