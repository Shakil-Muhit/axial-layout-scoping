"""External, mechanically checked _asdict repair for the bounded C3 arm.

Attend.flash_attn is copied from ZFTurbo/Music-Source-Separation-Training
@ea7eb9c20ea0e3f94368a30fc1654b51cdd55789, models/bs_roformer/attend.py:76.
Only the backend-context kwargs differ: direct fields replace _asdict().
The public source and every tensor operation remain unchanged.
"""
import ast
import hashlib
import inspect
import textwrap
import types

import torch
import torch.nn.functional as F


def exists(val):
    return val is not None


def flash_attn(self, q, k, v):
    _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

    if exists(self.scale):
        default_scale = q.shape[-1] ** -0.5
        q = q * (self.scale / default_scale)

    # Check if there is a compatible device for flash attention

    config = self.cuda_config if is_cuda else self.cpu_config

    # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

    with torch.backends.cuda.sdp_kernel(
        enable_flash=config.enable_flash,
        enable_math=config.enable_math,
        enable_mem_efficient=config.enable_mem_efficient,
    ):
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.
        )

    return out


def verify_source(original):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    replacements = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) != 'torch.backends.cuda.sdp_kernel':
            continue
        assert not node.args and len(node.keywords) == 1
        kw = node.keywords[0]
        assert kw.arg is None and ast.unparse(kw.value) == 'config._asdict()'
        node.keywords = [ast.keyword(
            arg=field,
            value=ast.Attribute(value=ast.Name(id='config', ctx=ast.Load()),
                                attr=field, ctx=ast.Load()))
            for field in ('enable_flash', 'enable_math', 'enable_mem_efficient')]
        replacements += 1
    assert replacements == 1, 'expected exactly one _asdict backend context'
    fixed_source = textwrap.dedent(inspect.getsource(flash_attn))
    assert ast.dump(tree) == ast.dump(ast.parse(fixed_source)), \
        'external method differs beyond the registered backend kwargs repair'
    return {'status': 'PASS', 'replacements': replacements,
            'original_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'replacement_source_sha256': hashlib.sha256(fixed_source.encode()).hexdigest()}


def install(model):
    from models.bs_roformer.attend import Attend
    verification = verify_source(Attend.flash_attn)
    pending = []
    for name, module in model.named_modules():
        if type(module) is Attend:
            assert module.flash and not module.training, name
            assert module.flash_attn.__func__ is Attend.flash_attn, name
            assert 'flash_attn' not in module.__dict__, name
            for config in (module.cpu_config, module.cuda_config):
                assert config is not None, name
                assert tuple(config._fields) == (
                    'enable_flash', 'enable_math', 'enable_mem_efficient'), name
                assert all(isinstance(value, bool) for value in config), name
            pending.append((name, module, types.MethodType(flash_attn, module)))
    assert len(pending) == 12, f'expected 12 Attend modules, got {len(pending)}'
    applied = []
    try:
        for name, module, method in pending:
            module.flash_attn = method
            applied.append(module)
    except Exception:
        for module in applied:
            del module.flash_attn
        raise
    return {**verification, 'modules': [name for name, _, _ in pending],
            'count': len(pending), 'cuda_flags': {
                field: getattr(pending[0][1].cuda_config, field)
                for field in pending[0][1].cuda_config._fields}}
