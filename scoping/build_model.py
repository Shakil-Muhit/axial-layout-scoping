"""Model construction + baseline compile.

Provenance (handoff rule 2): the model comes ONLY from the pinned public
clone via its own path — utils.settings.get_model_from_config →
torch.load(weights_only=False, map_location='cpu') →
utils.model_utils.load_start_checkpoint(type_='inference') — the exact
3-line pattern of upstream inference.py:210-216, mirrored the same way as
sm120-nulltest@a997e27 scripts/p8_trace.py (cited source of this loader).

Baseline compile (handoff section 2): the verbatim arm-C1 call from
sm120-nulltest@a997e27 arms/common.py::build_arm_step (C1/C2 branch):

    mode = 'reduce-overhead' if arm == 'C1' else 'max-autotune'
    compiled = torch.compile(fe, mode=mode, dynamic=False)

i.e. for C1:  torch.compile(model, mode='reduce-overhead', dynamic=False).

Precision (handoff rule 3): torch.autocast('cuda',
enabled=config.training.use_amp) — mirrors upstream demix()
(utils/model_utils.py:137-139 at the pin); use_amp recorded in every JSON.
"""

import argparse
import functools
import os
import sys

import torch


def load_public_model(msst_dir, ckpt, config_path,
                      model_type='mel_band_roformer', device='cuda'):
    sys.path.insert(0, msst_dir)
    from utils.settings import get_model_from_config
    from utils.model_utils import load_start_checkpoint

    model, config = get_model_from_config(model_type, config_path)
    ckpt_ns = argparse.Namespace(
        start_check_point=ckpt, model_type=model_type,
        lora_checkpoint='', lora_checkpoint_loralib='')
    checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')
    load_start_checkpoint(ckpt_ns, model, checkpoint, type_='inference')
    model = model.to(device).eval()

    use_amp = config.training.use_amp
    if not isinstance(use_amp, bool):
        raise TypeError('config.training.use_amp must be an explicit boolean')
    autocast_ctx = functools.partial(torch.autocast, 'cuda', enabled=use_amp)
    return model, config, use_amp, autocast_ctx


def compile_baseline_c1(model):
    """Verbatim arm-C1 compile call (see module docstring for the citation)."""
    compiled = torch.compile(model, mode='reduce-overhead', dynamic=False)
    return compiled


def paths_from_env():
    """Default artifact locations on ml-beast (scripts/beast_bootstrap.sh)."""
    ax = os.path.expanduser(os.environ.get('AXIAL_HOME', '~/axial'))
    return {
        'msst': os.environ.get('MSST_DIR', f'{ax}/msst'),
        'ckpt': os.environ.get('CKPT_FILE', f'{ax}/ckpt/MelBandRoformer.ckpt'),
        'config': os.environ.get('CONFIG_FILE',
                                 f'{ax}/msst/configs/KimberleyJensen/'
                                 'config_vocals_mel_band_roformer_kj.yaml'),
        'runs': os.environ.get('RUNS_DIR', f'{ax}/runs'),
    }
