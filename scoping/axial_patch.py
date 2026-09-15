"""Phase 2 intervention — canonical-layout axial stack, no new kernels.

MONKEYPATCHED COPY of a pinned-clone method, living in this repo (handoff
rule 2: interventions are external; the clone is never edited).
`patched_forward` below is a transcription of `MelBandRoformer.forward` from
models/bs_roformer/mel_band_roformer.py @ ea7eb9c20ea0e3f94368a30fc1654b51cdd55789
(source lines 538-747; sha256[:16] of the exact `inspect.getsource` text =
0db2007396ab2646 — the drift guard below refuses to patch a clone whose
forward hashes differently). The transcription is VERBATIM, byte-for-byte
with the loss branch included, except for exactly three declared deviations:
  (D1) a uniform 4-space dedent (module-level function vs class method);
  (D2) a marked ADAPTATION block of function-local imports right after the
       docstring (these names are module-level imports in the pinned file);
  (D3) ONE replaced region, marked PATCH BEGIN/END: the
       "# axial / hierarchical attention" stack loop.
scoping/verify_transcription.py checks D1-D3 mechanically against the
installed clone and is run by scripts/run_phase2.sh before any swap.

The replacement (docs/design_notes.md, incl. post-review-01 updates):
  - canonical activation: token-major contiguous (T, F, D); every per-token
    op (norms, to_qkv, gates, to_out, FF) runs on the flat (T*F, D) view;
  - q/k/v per axis are STRIDED VIEWS of the qkv GEMM output
    (time: (F,H,T,dh) strides (1536,64,92160,1);
     freq: (T,H,F,dh) strides (92160,64,1536,1)) — no .contiguous();
  - SDPA through the model's own Attend (flash-pinned backend, handoff §5.3);
  - gating + to_out run in the SDPA output's NATIVE memory order
    ((B,N,H,dh) contiguous presented as (B,H,N,dh)), where the flat
    (B·N, H·dh) view is free; canonical orientation is restored only at the
    residual through a strided view feeding the elementwise add;
  - per-sublayer math is exactly stock: y = norm; qkv; rotary(q,k); attend;
    gate·sigmoid; merge; to_out; +residual; ff; +residual; tx.norm.

Fail-closed swap (the sm120-nulltest build-all-then-commit pattern the
handoff §5 cites): every structural assumption is asserted BEFORE
model.forward is replaced; any exception leaves the stock model untouched.
"""

import hashlib
import inspect
import types

import torch
from einops import rearrange, repeat, pack, unpack

PINNED_FORWARD_SHA16 = '0db2007396ab2646'


class PatchUnsupported(RuntimeError):
    pass


# --------------------------------------------------------------- sublayer ---

def _axial_sublayer(tx, x_c, axis):
    """One axial Transformer application in canonical layout.

    Math-identical to `tx(x_packed)` for a depth-1 Transformer
    (x = attn(x)+x; x = ff(x)+x; return tx.norm(x)), with layout as the only
    difference. x_c: (T, F, D) contiguous. axis: 'time' | 'freq'.
    """
    (attn, ff), = tx.layers          # depth==1 asserted at swap time
    T, F, D = x_c.shape
    H = attn.heads
    dh = attn.to_qkv.out_features // (3 * H)

    flat = x_c.view(T * F, D)
    normed = attn.norm(flat)                        # per-token RMSNorm
    qkv = attn.to_qkv(normed)                       # (T*F, 3*H*dh), contiguous
    qkv5 = qkv.view(T, F, 3, H, dh)
    if axis == 'time':                              # batch=F, seq=T
        q = qkv5[:, :, 0].permute(1, 2, 0, 3)       # (F,H,T,dh) — view
        k = qkv5[:, :, 1].permute(1, 2, 0, 3)
        v = qkv5[:, :, 2].permute(1, 2, 0, 3)
    else:                                           # batch=T, seq=F
        q = qkv5[:, :, 0].permute(0, 2, 1, 3)       # (T,H,F,dh) — view
        k = qkv5[:, :, 1].permute(0, 2, 1, 3)
        v = qkv5[:, :, 2].permute(0, 2, 1, 3)

    if attn.rotary_embed is not None:               # rotates dim -2 (the seq)
        q = attn.rotary_embed.rotate_queries_or_keys(q)
        k = attn.rotary_embed.rotate_queries_or_keys(k)

    out = attn.attend(q, k, v)                      # (B,H,N,dh); flash-pinned

    gates = attn.to_gates(normed)                   # (T*F, H)
    if axis == 'time':
        g = gates.view(T, F, H).permute(1, 0, 2).unsqueeze(-1)   # (F,T,H,1)
    else:
        g = gates.view(T, F, H).unsqueeze(-1)                    # (T,F,H,1)

    # SDPA output memory is (B,N,H,dh)-contiguous presented as (B,H,N,dh);
    # stay in the native order for gating + head-merge + out-projection
    out_native = out.transpose(1, 2)                # (B,N,H,dh) — view
    out_native = out_native * g.sigmoid()           # elementwise, native order
    B, N = out_native.shape[0], out_native.shape[1]
    merged = out_native.reshape(B * N, H * dh)      # free view when contiguous
    proj = attn.to_out(merged)                      # (B*N, D)

    if axis == 'time':                              # rows in (f, t) order
        attn_out = proj.view(F, T, D).permute(1, 0, 2)   # (T,F,D) strided view
    else:                                           # rows already (t, f)
        attn_out = proj.view(T, F, D)
    x_c = x_c + attn_out                            # strided add -> canonical

    flat2 = x_c.view(T * F, D)
    x_c = (flat2 + ff(flat2)).view(T, F, D)         # FF + residual, per-token
    x_c = tx.norm(x_c.view(T * F, D)).view(T, F, D)  # Transformer output norm
    return x_c


# ---------------------------------------------------- patched forward copy ---
# Verbatim transcription of the pinned MelBandRoformer.forward — deviations
# D1 (dedent), D2 (ADAPTATION import block), D3 (PATCH region) only; checked
# mechanically by scoping/verify_transcription.py.

def patched_forward(
        self,
        raw_audio,
        target=None,
        active_stem_ids=None,
        return_loss_breakdown=False
):
    """
    einops

    b - batch
    f - freq
    t - time
    s - audio channel (1 for mono, 2 for stereo)
    n - number of 'stems'
    c - complex (2)
    d - feature dimension
    """
    # >>> ADAPTATION BEGIN — function-local imports (module-level names in the
    # pinned file; the only non-pinned lines outside the PATCH region) <<<
    from models.bs_roformer.mel_band_roformer import (
        pack_one, unpack_one, exists)
    import torch.nn.functional as F
    from torch import tensor
    from torch.utils.checkpoint import checkpoint
    # >>> ADAPTATION END <<<

    device = raw_audio.device

    if raw_audio.ndim == 2:
        raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

    batch, channels, raw_audio_length = raw_audio.shape

    istft_length = raw_audio_length if self.match_input_audio_length else None

    assert (not self.stereo and channels == 1) or (
                self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

    # to stft

    raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

    stft_window = self.stft_window_fn(device=device)

    stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
    stft_repr = torch.view_as_real(stft_repr)

    stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

    # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
    stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')

    # index out all frequencies for all frequency ranges across bands ascending in one go

    batch_arange = torch.arange(batch, device=device)[..., None]

    # account for stereo

    x = stft_repr[batch_arange, self.freq_indices]

    # fold the complex (real and imag) into the frequencies dimension

    x = rearrange(x, 'b f t c -> b t (f c)')

    if self.use_torch_checkpoint:
        x = checkpoint(self.band_split, x, use_reentrant=False)
    else:
        x = self.band_split(x)

    # >>> PATCH BEGIN — canonical-layout axial stack; replaces the pinned
    # region from "# axial / hierarchical attention" up to (not including)
    # "if active_stem_ids is None:" (see module docstring). The replaced
    # region's linear_transformer / skip_connection / use_torch_checkpoint
    # branches are guarded OFF at swap time. <<<
    assert batch == 1, 'canonical-layout patch assumes batch 1 (the study operating point)'
    x_c = x[0].contiguous()                     # (T, F, D) canonical; x is
    #                                             band_split's stack output —
    #                                             already contiguous; .contiguous()
    #                                             is a no-op guard, not a copy
    for transformer_block in self.layers:
        time_transformer, freq_transformer = transformer_block
        x_c = _axial_sublayer(time_transformer, x_c, 'time')
        x_c = _axial_sublayer(freq_transformer, x_c, 'freq')
    x = x_c.unsqueeze(0)
    # >>> PATCH END <<<

    if active_stem_ids is None:
        heads = self.mask_estimators
        stem_ids = list(range(len(self.mask_estimators)))
    else:
        heads = [self.mask_estimators[i] for i in active_stem_ids]
        stem_ids = active_stem_ids

    num_stems = len(heads)

    if self.use_torch_checkpoint:
        masks = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in heads], dim=1)
    else:
        masks = torch.stack([fn(x) for fn in heads], dim=1)
    masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

    # modulate frequency representation

    stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

    # complex number multiplication

    stft_repr = torch.view_as_complex(stft_repr)
    masks = torch.view_as_complex(masks)

    masks = masks.type(stft_repr.dtype)

    # need to average the estimated mask for the overlapped frequencies

    scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=num_stems, t=stft_repr.shape[-1])

    stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=num_stems)
    masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

    denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)

    masks_averaged = masks_summed / denom.clamp(min=1e-8)

    # modulate stft repr with estimated mask

    stft_repr = stft_repr * masks_averaged

    # istft

    stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

    if self.zero_dc:
        # whether to dc filter
        stft_repr = stft_repr.index_fill(1, tensor(0, device = device), 0.)

    recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False,
                              length=istft_length)

    recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=num_stems)

    if num_stems == 1:
        recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

    # if a target is passed in, calculate loss for learning

    if not exists(target):
        return recon_audio

    if self.num_stems > 1:
        assert target.ndim == 4 and target.shape[1] == self.num_stems

    if target.ndim == 2:
        target = rearrange(target, '... t -> ... 1 t')

    target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

    target_sel = target[:, stem_ids]

    loss = F.l1_loss(recon_audio, target_sel)

    multi_stft_resolution_loss = 0.

    for window_size in self.multi_stft_resolutions_window_sizes:
        res_stft_kwargs = dict(
            n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
            win_length=window_size,
            return_complex=True,
            window=self.multi_stft_window_fn(window_size, device=device),
            **self.multi_stft_kwargs,
        )

        recon_Y = torch.stft(
            rearrange(recon_audio, 'b n s t -> (b n s) t'),
            **res_stft_kwargs
        )
        target_Y = torch.stft(
            rearrange(target_sel, 'b n s t -> (b n s) t'),
            **res_stft_kwargs
        )

        multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

    weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

    total_loss = loss + weighted_multi_resolution_loss

    if not return_loss_breakdown:
        return total_loss

    return total_loss, (loss, multi_stft_resolution_loss)


# ------------------------------------------------------------------- swap ---

def swap_axial_forward(model):
    """Fail-closed swap: assert every structural assumption, THEN commit.
    Any exception leaves the stock model untouched. Returns an unswap()."""
    # drift guard: the installed clone's forward must be the transcribed one
    src = inspect.getsource(type(model).forward)
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    if sha != PINNED_FORWARD_SHA16:
        raise PatchUnsupported(
            f'pinned-forward drift: sha {sha} != {PINNED_FORWARD_SHA16}')
    if getattr(model, 'skip_connection', False):
        raise PatchUnsupported('skip_connection=True not supported')
    if getattr(model, 'use_torch_checkpoint', False):
        raise PatchUnsupported('use_torch_checkpoint=True not supported')
    for block in model.layers:
        if len(block) != 2:
            raise PatchUnsupported('linear_transformer block not supported')
        for tx in block:
            if len(tx.layers) != 1:
                raise PatchUnsupported('transformer depth != 1 not supported')
            attn, ff = tx.layers[0]
            if type(attn).__name__ != 'Attention':
                raise PatchUnsupported(f'unexpected attention: {type(attn)}')
            if getattr(attn, 'pope_embed', None) is not None:
                raise PatchUnsupported('pope_embed not supported')
            if attn.to_qkv.out_features % (3 * attn.heads) != 0:
                raise PatchUnsupported('qkv width not divisible by 3*heads')
    stock = model.forward
    model.forward = types.MethodType(patched_forward, model)
    def unswap():
        model.forward = stock
    return unswap
