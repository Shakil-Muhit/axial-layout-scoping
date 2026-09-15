"""Offline unit test of kernel_map v3 parsing against a realistic torch-2.8
Inductor output_code.py fixture (no torch/GPU needed). Guards the regex layer
the whole attribution rests on: metadata keyed by wrapper symbol (incl. a
generic inner def name), role-tagged ordered args, plain buffer reuse,
reinterpret aliases, load/store index expressions, stride-order permutation.
Run: python3 scoping/tests/test_parse_fixture.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_map import (parse_output_code, arg_meta, is_permuted, nbytes,  # noqa: E402
                        sdpa_axis, ops_from_name)

FIXTURE = r'''
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
async_compile = AsyncCompile()

# kernel path: /tmp/torchinductor_x/ab/cabc.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.clone]
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', """
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics

@triton_heuristics.pointwise(
    size_hints=[65536, 512],
    filename=__file__,
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1 + 307584*y0), xmask & ymask)
    tl.store(out_ptr0 + (x2), tmp0, xmask & ymask)
""", device_str='cuda')

# Topologically Sorted Source Nodes: [norm], Original ATen: [aten.add, aten.linalg_vector_norm]
triton_per_fused_add_linalg_vector_norm_1 = async_compile.triton('triton_per_fused_add_linalg_vector_norm_1', """
@triton.jit
def triton_per_fused_add_linalg_vector_norm_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    tmp0 = tl.load(in_ptr0 + (92160*x1 + r0), rmask & xmask)
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
""", device_str='cuda')

# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
triton_poi_fused_gelu_2 = async_compile.triton('triton_', """
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
""", device_str='cuda')

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (60, 801, 384), (307584, 384, 1))
    assert_size_stride(arg1_1, (384, 1536), (1536, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((801, 60, 384), (23040, 384, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(arg0_1, buf0, 48060, 384, grid=grid(48060, 384), stream=stream0)
        buf1 = empty_strided_cuda((48060, 1536), (1536, 1), torch.float16)
        extern_kernels.mm(reinterpret_tensor(buf0, (48060, 384), (384, 1), 0), reinterpret_tensor(arg1_1, (384, 1536), (1, 384), 0), out=buf1)
        buf2 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf1, (60, 8, 801, 64), (1536, 64, 92160, 1), 0), reinterpret_tensor(buf1, (60, 8, 801, 64), (1536, 64, 92160, 1), 512), reinterpret_tensor(buf1, (60, 8, 801, 64), (1536, 64, 92160, 1), 1024))
        buf3 = buf2[0]
        buf4 = empty_strided_cuda((48060, 512), (512, 1), torch.float16)
        buf5 = reinterpret_tensor(buf4, (60, 801, 512), (410112, 512, 1), 0); del buf4  # reuse
        triton_per_fused_add_linalg_vector_norm_1.run(buf5, reinterpret_tensor(buf3, (60, 801, 512), (512, 92160, 1), 0), 48060, 512, grid=grid(48060), stream=stream0)
        buf6 = buf0; del buf0  # reuse
        triton_poi_fused_gelu_2.run(buf5, buf6, 18455040, grid=grid(18455040), stream=stream0)
    return (buf6, )
'''


def main():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, 'model__0'))
        with open(os.path.join(td, 'model__0', 'output_code.py'), 'w') as f:
            f.write(FIXTURE)
        graphs = parse_output_code(td)

    assert len(graphs) == 1, graphs
    g = graphs[0]
    k0 = g['kernels']['triton_poi_fused_clone_0']
    assert k0['ops'] == ['aten.clone'] and k0['ops_source'] == 'aten_comment', k0
    assert k0['params'] == ['in_ptr0', 'out_ptr0'], k0['params']
    assert not k0['inner_mismatch']
    assert k0['load_exprs']['in_ptr0'] == 'x0+384*x1+307584*y0', k0['load_exprs']
    assert k0['store_exprs']['out_ptr0'] == 'x2', k0['store_exprs']

    k1 = g['kernels']['triton_per_fused_add_linalg_vector_norm_1']
    assert k1['ops'] == ['aten.add', 'aten.linalg_vector_norm'], k1
    assert k1['params'] == ['in_out_ptr0', 'in_ptr0'], k1['params']
    assert k1['load_exprs']['in_ptr0'] == '92160*x1+r0', k1['load_exprs']

    # generic inner def name: metadata keyed by the WRAPPER symbol (gate-03 f19)
    k2 = g['kernels']['triton_poi_fused_gelu_2']
    assert k2['inner_name'] == 'triton_' and k2['inner_mismatch'], k2
    assert k2['ops'] == ['aten.gelu'] and k2['params'] == ['in_ptr0', 'out_ptr0'], k2

    assert [c['kind'] for c in g['calls']] == \
        ['triton', 'extern', 'fallback', 'triton', 'triton'], \
        [c['kind'] for c in g['calls']]

    c0 = g['calls'][0]
    assert c0['align_ok'] and len(c0['args']) == 2, c0
    assert c0['args'][0]['buf'] == 'arg0_1' and c0['args'][0]['read'] \
        and not c0['args'][0]['write']
    assert c0['args'][1]['buf'] == 'buf0' and c0['args'][1]['write'] \
        and not c0['args'][1]['read']
    m_in = arg_meta(g, c0['args'][0])
    assert m_in['shape'] == (60, 801, 384) and m_in['stride'] == (307584, 384, 1), m_in
    m_out = arg_meta(g, c0['args'][1])
    assert m_out['dtype'] == 'torch.float16' and m_out['shape'] == (801, 60, 384), m_out

    c1 = g['calls'][1]
    assert c1['name'] == 'extern.mm'
    roles = [(a['buf'], a['read'], a['write']) for a in c1['args']]
    assert roles == [('buf0', True, False), ('arg1_1', True, False),
                     ('buf1', False, True)], roles
    m = arg_meta(g, c1['args'][0])
    assert m['shape'] == (48060, 384) and m['stride'] == (384, 1) \
        and m['dtype'] == 'torch.float16' and m['base'] == 'buf0', m

    c2 = g['calls'][2]
    assert c2['assigned'] == 'buf2' and 'scaled_dot_product' in c2['name']
    q = arg_meta(g, c2['args'][0])
    assert sdpa_axis(q['shape']) == 'time', q
    assert all(a['read'] for a in c2['args'])
    assert g['aliases']['buf3'] == {'base': 'buf2', 'item': 0}, g['aliases']

    c3 = g['calls'][3]
    a_io = c3['args'][0]
    assert not a_io['read'] and a_io['write'] and a_io['role'] == 'in_out_ptr0', a_io
    m_io = arg_meta(g, a_io)   # buf5 reinterpret-alias -> base buf4
    assert m_io['base'] == 'buf4' and m_io['shape'] == (60, 801, 512), m_io
    m_r = arg_meta(g, c3['args'][1])
    assert m_r['stride'] == (512, 92160, 1), m_r

    # plain reuse rename (gate-03 f20): buf6 aliases buf0's storage
    assert g['aliases']['buf6'] == {'base': 'buf0'}, g['aliases'].get('buf6')
    c4 = g['calls'][4]
    m4_out = arg_meta(g, c4['args'][1])
    assert m4_out['base'] == 'buf0' and m4_out['dtype'] == 'torch.float16', m4_out

    # is_permuted: stride-order test
    assert is_permuted((60, 801, 384), (384, 23040, 1)) is True      # axis swap
    assert is_permuted((60, 801, 512), (512, 92160, 1)) is True      # sdpa native
    assert is_permuted((48060, 384), (384, 1)) is False              # row-major
    assert is_permuted((60, 8, 801, 64), (1536, 64, 92160, 1)) is True
    assert is_permuted((1, 384), (0, 1)) is False                    # broadcast-ish
    assert nbytes((10, 10), 'torch.float32') == 400
    assert ops_from_name('triton_poi_fused__to_copy_add_12') == '_to_copy_add'

    print('FIXTURE-OK: kernel_map v3 parser assertions all pass')


if __name__ == '__main__':
    main()
