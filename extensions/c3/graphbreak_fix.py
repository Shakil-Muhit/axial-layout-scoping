"""External, mechanically checked _asdict repair for the bounded C3 arm.

Attend.flash_attn is copied from ZFTurbo/Music-Source-Separation-Training
@ea7eb9c20ea0e3f94368a30fc1654b51cdd55789, models/bs_roformer/attend.py:76.
Only backend context construction differs: precomputed enum lists replace
_asdict() and the unsupported legacy generator context. The enum order and
implicit cuDNN enablement match the installed torch 2.8 legacy wrapper.
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

    config = self._c3_cuda_backends if is_cuda else self._c3_cpu_backends

    # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

    with torch.nn.attention.sdpa_kernel(config):
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.
        )

    return out


def verify_source(original):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    replacements = 0
    config_replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                ast.unparse(node.targets[0]) == 'config'):
            assert ast.unparse(node.value) == 'self.cuda_config if is_cuda else self.cpu_config'
            node.value = ast.parse('self._c3_cuda_backends if is_cuda else self._c3_cpu_backends',
                                   mode='eval').body
            config_replacements += 1
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) != 'torch.backends.cuda.sdp_kernel':
            continue
        assert not node.args and len(node.keywords) == 1
        kw = node.keywords[0]
        assert kw.arg is None and ast.unparse(kw.value) == 'config._asdict()'
        node.func = ast.parse('torch.nn.attention.sdpa_kernel', mode='eval').body
        node.args = [ast.Name(id='config', ctx=ast.Load())]
        node.keywords = []
        replacements += 1
    assert replacements == config_replacements == 1
    fixed_source = textwrap.dedent(inspect.getsource(flash_attn))
    assert ast.dump(tree) == ast.dump(ast.parse(fixed_source)), \
        'external method differs beyond the backend-context repair'
    return {'status': 'PASS', 'replacements': replacements,
            'original_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'replacement_source_sha256': hashlib.sha256(fixed_source.encode()).hexdigest()}


def backend_list(config):
    # Exact ordering/default from torch 2.8.0 torch/backends/cuda/__init__.py
    # sdp_kernel. The old API's unmentioned enable_cudnn parameter is True.
    backend = torch.nn.attention.SDPBackend
    result = []
    if config.enable_flash:
        result.append(backend.FLASH_ATTENTION)
    if config.enable_mem_efficient:
        result.append(backend.EFFICIENT_ATTENTION)
    if config.enable_math:
        result.append(backend.MATH)
    result.append(backend.CUDNN_ATTENTION)
    return result


def backend_state():
    cuda = torch.backends.cuda
    return {name: bool(getattr(cuda, name + '_sdp_enabled')())
            for name in ('flash', 'math', 'mem_efficient', 'cudnn')}


def verify_backend_context(config):
    assert inspect.signature(torch.backends.cuda.sdp_kernel).parameters[
        'enable_cudnn'].default is True
    before = backend_state()
    with torch.backends.cuda.sdp_kernel(**config._asdict()):
        original = backend_state()
    assert backend_state() == before
    with torch.nn.attention.sdpa_kernel(backend_list(config)):
        replacement = backend_state()
    assert backend_state() == before
    assert original == replacement
    return {'original': original, 'replacement': replacement,
            'state_restored': True}


def install(model):
    from models.bs_roformer.attend import Attend
    verification = verify_source(Attend.flash_attn)
    pending = []
    backend_checks = {}
    for name, module in model.named_modules():
        if type(module) is Attend:
            assert module.flash and not module.training, name
            assert module.flash_attn.__func__ is Attend.flash_attn, name
            assert 'flash_attn' not in module.__dict__, name
            assert not hasattr(module, '_c3_cuda_backends'), name
            assert not hasattr(module, '_c3_cpu_backends'), name
            for config in (module.cpu_config, module.cuda_config):
                assert config is not None, name
                assert tuple(config._fields) == (
                    'enable_flash', 'enable_math', 'enable_mem_efficient'), name
                assert all(isinstance(value, bool) for value in config), name
                backend_checks[str(tuple(config))] = verify_backend_context(config)
            pending.append((name, module, types.MethodType(flash_attn, module)))
    assert len(pending) == 12, f'expected 12 Attend modules, got {len(pending)}'
    applied = []
    try:
        for name, module, method in pending:
            applied.append(module)
            module._c3_cpu_backends = backend_list(module.cpu_config)
            module._c3_cuda_backends = backend_list(module.cuda_config)
            module.flash_attn = method
    except Exception:
        for module in applied:
            for attr in ('flash_attn', '_c3_cpu_backends', '_c3_cuda_backends'):
                if attr in module.__dict__:
                    delattr(module, attr)
        raise
    return {**verification, 'modules': [name for name, _, _ in pending],
            'count': len(pending), 'backend_context_checks': backend_checks,
            'cuda_backend_order': [str(x) for x in pending[0][1]._c3_cuda_backends],
            'cuda_flags': {
                field: getattr(pending[0][1].cuda_config, field)
                for field in pending[0][1].cuda_config._fields}}
