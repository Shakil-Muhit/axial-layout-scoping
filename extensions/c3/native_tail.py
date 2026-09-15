"""Native DC-filter boundary for the confirmed Torch 2.8 AOT failure.

The isolated complex, strided index_fill reproducer fails AOT's functional
graph assertion. Call the original Tensor.index_fill without decomposing it.
No new GPU kernel or arithmetic is introduced; torch.istft stays unchanged.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

import torch


@torch.library.custom_op('axial_c3::native_dc_fill', mutates_args=())
def native_dc_fill(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    result = input.index_fill(1, index, 0.)
    assert result.shape == input.shape == (2, 1025, 801)
    assert result.stride() == input.stride() and result.storage_offset() == 0
    assert result.dtype == input.dtype == torch.complex64
    return result


@native_dc_fill.register_fake
def native_dc_fill_fake(input, index):
    assert input.shape == (2, 1025, 801) and input.dtype == torch.complex64
    assert input.stride() in ((801, 1602, 1), (821025, 801, 1))
    assert index.shape == () and index.dtype == torch.int64
    return torch.empty_strided(input.shape, input.stride(),
                               device=input.device, dtype=input.dtype)


def call_native_dc_fill(input, dim, index, value):
    assert dim == 1 and value == 0.
    return native_dc_fill(input, index)


def install(model, out_dir):
    from models.bs_roformer.mel_band_roformer import MelBandRoformer
    assert type(model) is MelBandRoformer
    assert model.forward.__func__ is MelBandRoformer.forward
    assert 'forward' not in model.__dict__
    original = textwrap.dedent(inspect.getsource(MelBandRoformer.forward))
    tree = ast.parse(original)
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and
                ast.unparse(node.func) == 'stft_repr.index_fill'):
            assert len(node.args) == 3 and not node.keywords
            assert ast.unparse(node.args[0]) == '1'
            assert ast.unparse(node.args[1]) == 'tensor(0, device=device)'
            assert isinstance(node.args[2], ast.Constant) and node.args[2].value == 0.
            node.func = ast.Name(id='c3_native_dc_fill', ctx=ast.Load())
            node.args.insert(0, ast.Name(id='stft_repr', ctx=ast.Load()))
            replacements += 1
    assert replacements == 1, 'expected the single public DC-filter call'
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    restored = ast.parse(source)
    for node in ast.walk(restored):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'c3_native_dc_fill':
            base = node.args.pop(0)
            node.func = ast.Attribute(value=base, attr='index_fill', ctx=ast.Load())
    assert ast.dump(restored) == ast.dump(ast.parse(original))
    dest = Path(out_dir) / 'generated_public_forward.py'
    dest.write_text(source)
    namespace = dict(MelBandRoformer.forward.__globals__)
    namespace['c3_native_dc_fill'] = call_native_dc_fill
    exec(compile(source, str(dest), 'exec'), namespace)
    model.forward = types.MethodType(namespace['forward'], model)
    return {'status': 'PASS', 'replacements': replacements,
            'original_forward_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'external_forward_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'generated_source': str(dest), 'native_call': 'Tensor.index_fill',
            'custom_op': 'axial_c3::native_dc_fill', 'new_gpu_kernel': False,
            'cudagraph_unsafe_tag': False,
            'scope': 'fixed inference shape; public forward differs only in DC-filter callee'}


def verify_operator():
    gen = torch.Generator(device='cpu').manual_seed(4242)
    x = torch.randn((1025, 2, 801), generator=gen, dtype=torch.complex64).to('cuda')
    x = x.permute(1, 0, 2)
    index = torch.tensor(0, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', enabled=True):
        checks = torch.library.opcheck(native_dc_fill, (x, index),
                                      test_utils=('test_schema', 'test_faketensor'))
        reference = x.index_fill(1, index, 0.)
        actual = native_dc_fill(x, index)
        assert torch.equal(actual, reference)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = native_dc_fill(x, index)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, reference)
        compiled = torch.compile(native_dc_fill, mode='reduce-overhead',
                                 dynamic=False, fullgraph=True)
        for _ in range(6):
            result = compiled(x, index).clone()
        torch.cuda.synchronize()
        assert torch.equal(result, reference)
    return {'status': 'PASS', 'opcheck': checks, 'native_bitwise_equal': True,
            'cuda_graph_capture_and_replay': 'PASS', 'compiled_fullgraph': 'PASS',
            'output_shape': list(actual.shape), 'output_stride': list(actual.stride()),
            'output_dtype': str(actual.dtype), 'output_storage_offset': actual.storage_offset()}
