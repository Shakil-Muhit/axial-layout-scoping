"""Phase 1b — kernel-name classifier (pure function; handoff 1b requires the
source plus a 20-row labeled sample in evidence/ for owner spot-checks).

Classes: GEMM, SDPA, INDUCTOR_FUSED, COPY, FFT, OTHER.
Precedence: INDUCTOR_FUSED first (a triton_poi_fused_clone_* kernel is
INDUCTOR_FUSED here; its pure-copy nature is a 1c property, not a 1b class),
then SDPA, GEMM, FFT, COPY, OTHER.
"""

import csv
import sys


def classify(kernel_name: str) -> str:
    n = kernel_name.lower()
    if n.startswith('triton_') or '_triton_' in n:
        return 'INDUCTOR_FUSED'
    if ('flash' in n or 'fmha' in n or '_scaled_dot_product' in n
            or ('cutlass' in n and 'attention' in n) or 'attn' in n):
        return 'SDPA'
    if ('gemm' in n or 'cublas' in n or 'nvjet' in n or 'gemv' in n
            or 'cutlass' in n or 's16816' in n or 'wgmma' in n
            or n.startswith(('ampere_', 'sm80_', 'sm90_', 'sm100_', 'sm120_'))):
        return 'GEMM'
    if 'fft' in n:
        return 'FFT'
    if ('memcpy' in n or 'memset' in n or 'copy' in n or 'clone' in n
            or 'contiguous' in n
            or ('elementwise_kernel' in n and ('copy' in n or 'direct' in n))):
        return 'COPY'
    return 'OTHER'


def emit_sample(kern_sum_csv, out_csv, rows=20):
    """20-row labeled sample, largest kernels first (owner spot-check)."""
    with open(kern_sum_csv, newline='') as f:
        rd = list(csv.reader(f))
    head = rd[0]
    def col(*names):
        for name in names:
            for i, h in enumerate(head):
                if h.strip().strip('"') == name:
                    return i
        key = names[0].split(' ')[0]
        return next(i for i, h in enumerate(head)
                    if key in h and 'Range' not in h and 'Proj' not in h
                    and 'NVTX' not in h)
    t, nm = col('Total Time (ns)'), col('Name')
    body = sorted((r for r in rd[1:] if len(r) > max(t, nm) and r[t].strip()),
                  key=lambda r: -int(float(r[t].replace(',', ''))))
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'total_time_ns', 'kernel_name'])
        for r in body[:rows]:
            w.writerow([classify(r[nm]), r[t], r[nm]])
    print(f'-> {out_csv}')


if __name__ == '__main__':
    emit_sample(sys.argv[1], sys.argv[2])
