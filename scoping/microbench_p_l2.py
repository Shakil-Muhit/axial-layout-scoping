"""P-L2 microbench — per-class BW_ref and the permuted-read penalty
(handoff 1e(B); reworked per codex gates 02 f3 and 03 f21).

Three representative fused-kernel op-chains, each torch.compile'd standalone
(Inductor generates the same class of fused kernel as in the model), run on
CONTIGUOUS vs PERMUTED-VIEW inputs at identical logical shapes:

  copy      : torch.clone(x, memory_format=contiguous_format)
              (a true layout conversion: permuted read -> CONTIGUOUS write)
  norm      : F.normalize(x, dim=-1) * scale * gamma  (the RMSNorm chain)
  pointwise : gelu((x + bias) * 1.5)

Precision parity: every chain runs inside torch.autocast('cuda',
enabled=True) — the context the model's kernels are generated under — so the
norm chain gets the same internal promotion the model's normalize does.
Traffic accounting stays fp16 in/out (2 tensors), the model-observed
boundary traffic for these fusions.

Timing is DEVICE time with launch-gap immunity: BLOCK back-to-back launches
are captured into one CUDA Graph and the REPLAY is timed with CUDA events
(median of REPEATS replays). If capture fails, falls back to event-timed
eager blocks and RECORDS the method (queue bubbles then bias BW_ref low ->
(B) low -> lower bound; the json says which method ran). The (B) numerator in
kernel_map is device-side kernel duration, so BW_ref must be device-side too.

Output-layout verification: the permuted-input variant's output strides and
contiguity are recorded per class (the copy class must write CONTIGUOUS
output or it dodged the conversion under test).
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from evidence_io import write_json

SHAPE = (60, 801, 384)   # logical shape at the swap
DTYPE = torch.float16
BLOCK = 50               # launches per timed block / captured graph
REPEATS = 20             # timed replays per measurement (median taken)
WARMUP = 30


def make_inputs():
    gen = torch.Generator(device='cpu').manual_seed(11)
    base = torch.randn(*SHAPE, generator=gen).to('cuda', DTYPE).contiguous()
    permuted_store = torch.randn(SHAPE[1], SHAPE[0], SHAPE[2],
                                 generator=gen).to('cuda', DTYPE).contiguous()
    permuted_view = permuted_store.permute(1, 0, 2)   # (60,801,384) view,
    #                                                   stride (384, 23040, 1)
    return base, permuted_view


def chains():
    gamma = torch.ones(SHAPE[-1], device='cuda', dtype=DTYPE)
    scale = SHAPE[-1] ** 0.5
    bias = torch.randn(SHAPE[-1], device='cuda', dtype=DTYPE)

    def c_copy(x):
        return torch.clone(x, memory_format=torch.contiguous_format)

    def c_norm(x):
        return F.normalize(x, dim=-1) * scale * gamma

    def c_pointw(x):
        return F.gelu((x + bias) * 1.5)

    return {'copy': (c_copy, 2), 'norm': (c_norm, 2), 'pointwise': (c_pointw, 2)}
    # second element: bytes multiplier (read + write of one tensor's size)


def _autocast():
    return torch.autocast('cuda', enabled=True)


def bench_device_ms(fn, x):
    """Median per-launch device ms. Preferred: time the replay of a CUDA
    Graph capturing BLOCK launches (no queue bubbles). Fallback: event-timed
    eager blocks. Returns (ms, method)."""
    with torch.no_grad(), _autocast():
        for _ in range(WARMUP):
            fn(x)
    torch.cuda.synchronize()

    graph = None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad(), _autocast():
            for _ in range(3):
                fn(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad(), _autocast():
            for _ in range(BLOCK):
                fn(x)
        graph = g
    except Exception as e:
        print(f'[p-l2] WARN: graph capture failed ({type(e).__name__}: {e}); '
              'falling back to eager event blocks')

    per_launch = []
    for _ in range(REPEATS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        if graph is not None:
            e0.record()
            graph.replay()
            e1.record()
        else:
            with torch.no_grad(), _autocast():
                e0.record()
                for _ in range(BLOCK):
                    fn(x)
                e1.record()
        torch.cuda.synchronize()
        per_launch.append(e0.elapsed_time(e1) / BLOCK)
    return statistics.median(per_launch), \
        ('cuda-graph-replay' if graph is not None else 'eager-event-blocks')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available()

    base, perm = make_inputs()
    n_bytes = base.numel() * base.element_size()
    result = {'shape': list(SHAPE), 'dtype': str(DTYPE),
              'bytes_one_tensor': n_bytes,
              'autocast': 'torch.autocast(cuda, enabled=True) around compile '
                          'and timing (model-context parity)',
              'timing': f'{BLOCK}-launch blocks x {REPEATS}, median; method '
                        'recorded per class',
              'classes': {}}
    methods = set()
    for name, (fn, mult) in chains().items():
        cfn = torch.compile(fn, dynamic=False)
        with torch.no_grad(), _autocast():
            y_c = cfn(base)
            y_p = cfn(perm)      # stride change -> its own compiled variant
        t_c, m_c = bench_device_ms(cfn, base)
        t_p, m_p = bench_device_ms(cfn, perm)
        methods.update((m_c, m_p))
        # Charge the observed output dtype. CUDA autocast may promote norm
        # outputs to fp32; charging two fp16 tensors understated its traffic.
        param_bytes = 0 if name == 'copy' else SHAPE[-1] * base.element_size()
        bytes_c = n_bytes + y_c.numel() * y_c.element_size() + param_bytes
        bytes_p = n_bytes + y_p.numel() * y_p.element_size() + param_bytes
        gb = bytes_c / 1e9
        accepted = (m_c == 'cuda-graph-replay' and m_p == 'cuda-graph-replay'
                    and y_c.dtype == y_p.dtype
                    and (name != 'copy' or y_p.is_contiguous()))
        result['classes'][name] = {
            'reference_accepted': bool(accepted),
            'bytes_contiguous': bytes_c, 'bytes_permuted': bytes_p,
            'input_dtype': str(base.dtype),
            'contiguous_ms': t_c, 'permuted_ms': t_p,
            'contiguous_GBps': gb / (t_c / 1e3),
            'permuted_GBps': (bytes_p / 1e9) / (t_p / 1e3),
            'penalty_ratio': t_p / t_c,
            'timing_method': {'contiguous': m_c, 'permuted': m_p},
            'out_dtype_contig_in': str(y_c.dtype),
            'out_stride_contig_in': list(y_c.stride()),
            'out_dtype_perm_in': str(y_p.dtype),
            'out_stride_perm_in': list(y_p.stride()),
            'out_is_contiguous_perm_in': bool(y_p.is_contiguous()),
        }
        print(f"[p-l2] {name:10s} contig {gb/(t_c/1e3):7.1f} GB/s | "
              f"perm {gb/(t_p/1e3):7.1f} GB/s | ratio {t_p/t_c:.2f}x | "
              f"perm-out contig={bool(y_p.is_contiguous())} | {m_c}/{m_p}")

    if not result['classes']['copy']['out_is_contiguous_perm_in']:
        result['copy_class_warning'] = ('copy chain did NOT write contiguous '
                                        'output on the permuted input — the '
                                        'conversion under test was dodged; '
                                        'treat copy BW_ref as unresolved')
    contig = {k: v['contiguous_GBps'] for k, v in result['classes'].items()
              if v['reference_accepted']}
    if 'copy_class_warning' in result:
        contig.pop('copy', None)
    result['BW_ref_GBps'] = {
        **contig,
        'min': min(contig.values(), default=None), 'max': max(contig.values(), default=None),
        'note': 'BW_ref per kernel CLASS = device-side achieved GB/s of that '
                'class on contiguous inputs (handoff 1e(B)); kernel_map maps '
                'each (B) site to its class — a class absent here is excluded '
                'there and flagged',
    }
    result['timing_methods_used'] = sorted(methods)
    result['p_l2_max_penalty_ratio'] = max(
        c['penalty_ratio'] for c in result['classes'].values())
    result['use_amp'] = True
    write_json(args.out, result)


if __name__ == '__main__':
    main()
