"""Keep native iSTFT opaque after the attention graph-break repair.

Torch 2.8's full-forward AOT decomposition fails its functional-graph check
on an internal iSTFT copy_. This adapter calls the existing torch.istft;
it implements no FFT, overlap-add, arithmetic replacement, or GPU kernel.
Only the fixed registered inference operating point is supported.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import types
from typing import Optional

import torch


@torch.library.custom_op('axial_c3::native_istft', mutates_args=(),
                         tags=(torch.Tag.cudagraph_unsafe,))
def native_istft(input: torch.Tensor, window: torch.Tensor,
                 n_fft: int, hop_length: int, win_length: int,
                 normalized: bool, length: Optional[int]) -> torch.Tensor:
    result = torch.istft(input, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=window,
                        normalized=normalized, length=length,
                        return_complex=False)
    # Verify the fake signature against every real call, including capture.
    assert result.shape == (2, 352800)
    assert result.stride() == (352800, 1) and result.storage_offset() == 0
    assert result.dtype == torch.float32
    return result


@native_istft.register_fake
def native_istft_fake(input, window, n_fft, hop_length, win_length, normalized, length):
    assert input.shape == (2, 1025, 801) and input.dtype == torch.complex64
    assert window.shape == (2048,) and window.dtype == torch.float32
    assert (n_fft, hop_length, win_length, normalized) == (2048, 441, 2048, False)
    assert length is None or length == 352800
    return torch.empty((2, 352800), device=input.device, dtype=torch.float32)


def call_native_istft(input, n_fft, hop_length, win_length, normalized,
                     window, return_complex, length):
    assert return_complex is False
    return native_istft(input, window, n_fft, hop_length, win_length, normalized, length)


def install(model, out_dir):
    from models.bs_roformer.mel_band_roformer import MelBandRoformer
    assert type(model) is MelBandRoformer
    assert model.forward.__func__ is MelBandRoformer.forward
    assert 'forward' not in model.__dict__
    original = textwrap.dedent(inspect.getsource(MelBandRoformer.forward))
    tree = ast.parse(original)
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.istft':
            node.func = ast.Name(id='c3_native_istft', ctx=ast.Load())
            replacements += 1
    assert replacements == 1, 'expected the single public inference iSTFT call'
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    restored = ast.parse(source)
    for node in ast.walk(restored):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'c3_native_istft':
            node.func = ast.parse('torch.istft', mode='eval').body
    assert ast.dump(restored) == ast.dump(ast.parse(original))
    dest = Path(out_dir) / 'generated_public_forward.py'
    dest.write_text(source)
    namespace = dict(MelBandRoformer.forward.__globals__)
    namespace['c3_native_istft'] = call_native_istft
    exec(compile(source, str(dest), 'exec'), namespace)
    replacement = types.MethodType(namespace['forward'], model)
    model.forward = replacement
    return {'status': 'PASS', 'replacements': replacements,
            'original_forward_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'external_forward_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'generated_source': str(dest), 'native_call': 'torch.istft',
            'custom_op': 'axial_c3::native_istft', 'new_gpu_kernel': False,
            'cudagraph_unsafe_tag': True,
            'scope': 'fixed inference shape; public forward differs only in iSTFT callee'}


def verify_operator():
    x = torch.zeros((2, 1025, 801), dtype=torch.complex64, device='cuda')
    window = torch.hann_window(2048, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', enabled=True):
        result = torch.library.opcheck(
            native_istft, (x, window, 2048, 441, 2048, False, None),
            test_utils=('test_schema', 'test_faketensor'))
    torch.cuda.synchronize()
    return {'status': 'PASS', 'opcheck': result,
            'output_shape': [2, 352800], 'output_stride': [352800, 1],
            'output_dtype': 'torch.float32', 'output_storage_offset': 0}
