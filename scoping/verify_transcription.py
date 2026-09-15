"""Mechanical proof that axial_patch.patched_forward is the pinned
MelBandRoformer.forward text modulo its three DECLARED deviations (evidence
for handoff rule 2 — the clone is never edited, the intervention is external
and auditable):
  D1  uniform 4-space dedent (module-level function vs class method)
  D2  the marked ADAPTATION import block (module-level names in the pin)
  D3  the marked PATCH region (the axial stack loop)

Checks, line by line:
  pinned[def .. "# axial / hierarchical attention") == patched[def .. PATCH BEGIN)
  pinned["if active_stem_ids is None:" .. end]      == patched[(PATCH END .. end]
under exactly these declared normalizations (the checker proves equality
MODULO them, and nothing else): (n1) the pinned text is dedented by 4;
(n2) the def line is renamed forward -> patched_forward; (n3) the ADAPTATION
block is removed — but ONLY after its content is verified to equal the
canonical import block below, so arbitrary code cannot hide there (gate-03
f31); (n4) blank lines at the segment seams are stripped. Marker uniqueness
and ordering (ADAPTATION BEGIN < END < PATCH BEGIN < END, one each) are
asserted. Also asserts sha256[:16] of the exact pinned source ==
axial_patch's drift-guard constant. Exit 0 + VERBATIM-OK only if every line
matches.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_model as bm  # noqa: E402
import axial_patch  # noqa: E402

# The ONLY lines permitted inside the ADAPTATION block (D2), stripped of
# comment lines; anything else fails the check.
ALLOWED_ADAPTATION = [
    'from models.bs_roformer.mel_band_roformer import (',
    '    pack_one, unpack_one, exists)',
    'import torch.nn.functional as F',
    'from torch import tensor',
    'from torch.utils.checkpoint import checkpoint',
]


def dedent4(lines):
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            out.append('')
        else:
            assert ln.startswith('    '), f'pinned line {i+1} not 4-indented: {ln!r}'
            out.append(ln[4:])
    return out


def strip_seam(seg, head=False, tail=False):
    a, b = 0, len(seg)
    if head:
        while a < b and not seg[a].strip():
            a += 1
    if tail:
        while b > a and not seg[b - 1].strip():
            b -= 1
    return seg[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msst-dir', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    msst = args.msst_dir or bm.paths_from_env()['msst']
    sys.path.insert(0, msst)
    from models.bs_roformer.mel_band_roformer import MelBandRoformer

    src = inspect.getsource(MelBandRoformer.forward)
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    assert sha == axial_patch.PINNED_FORWARD_SHA16, (
        f'sha {sha} != drift-guard {axial_patch.PINNED_FORWARD_SHA16}')

    pinned = dedent4(src.splitlines())
    assert pinned[0].startswith('def forward('), pinned[0]
    pinned[0] = pinned[0].replace('def forward(', 'def patched_forward(', 1)
    ax = next(i for i, l in enumerate(pinned)
              if l.strip() == '# axial / hierarchical attention')
    ac = next(i for i, l in enumerate(pinned)
              if l.strip() == 'if active_stem_ids is None:')
    pin_pre = strip_seam(pinned[:ax], tail=True)
    pin_post = strip_seam(pinned[ac:], tail=True)

    mine = inspect.getsource(axial_patch.patched_forward).splitlines()

    def marker_positions(lines, token):
        return [i for i, l in enumerate(lines) if token in l]

    pos = {t: marker_positions(mine, t) for t in
           ('>>> ADAPTATION BEGIN', '>>> ADAPTATION END',
            '>>> PATCH BEGIN', '>>> PATCH END')}
    for t, p in pos.items():
        assert len(p) == 1, f'marker {t!r} must appear exactly once, got {p}'
    ab = pos['>>> ADAPTATION BEGIN'][0]
    ae = pos['>>> ADAPTATION END'][0]
    assert ab < ae < pos['>>> PATCH BEGIN'][0] < pos['>>> PATCH END'][0], \
        f'marker order violated: {pos}'
    # ADAPTATION content must be EXACTLY the permitted import block (f31)
    block = [l[4:] if l.startswith('    ') else l
             for l in mine[ab + 1:ae] if l.strip()
             and not l.strip().startswith('#')]
    assert block == ALLOWED_ADAPTATION, (
        'ADAPTATION block contains lines outside the permitted import block:\n'
        + '\n'.join(f'  {l!r}' for l in block))
    mine = mine[:ab] + mine[ae + 1:]
    pb = marker_positions(mine, '>>> PATCH BEGIN')[0]
    pe = marker_positions(mine, '>>> PATCH END')[0]
    my_pre = strip_seam(mine[:pb], tail=True)
    my_post = strip_seam(mine[pe + 1:], head=True, tail=True)

    report = {'sha16': sha, 'pre_lines': len(pin_pre), 'post_lines': len(pin_post),
              'patch_region_pinned_lines': ac - ax, 'mismatches': []}
    for label, a, b in (('pre', pin_pre, my_pre), ('post', pin_post, my_post)):
        if len(a) != len(b):
            report['mismatches'].append(
                f'{label}: length {len(a)} (pinned) vs {len(b)} (patched)')
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                report['mismatches'].append(
                    f'{label}[{i}]: pinned {x!r} != patched {y!r}')
    report['ok'] = not report['mismatches']
    if args.out:
        from evidence_io import write_json
        write_json(args.out, report)
    if report['ok']:
        print(f'VERBATIM-OK sha16={sha} pre={len(pin_pre)} post={len(pin_post)} '
              f'lines match; patch region spans {ac - ax} pinned lines')
    else:
        for m in report['mismatches'][:20]:
            print('MISMATCH', m, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
