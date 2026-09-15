"""Assemble the three headline numbers from run artifacts (pure JSON/CSV math
— no GPU). Emits summary.json with every number RESULT.md needs plus the
evidence path each one came from. The GO/TICKET/NO-GO decision is the
OWNER's; this script only computes the pre-registered formulas and REFUSES a
headline number whose acceptance gates fail (codex gates 02 f11/f12/f15 and
03 f24/f25).

  R_layout    : copied from r_layout_baseline.json with its status/flags —
                a FLAGGED map is reported as a lower bound, never silently.
  R_recovered : numeric ONLY IF every gate passes (baseline evidence accepted,
                both maps status 'ok' with B complete, parity, settle, tight
                validation, headline timing, in-stack (A) fell, patched heads
                split+merge are views, zero SDPA-adjacent copies, and clock
                comparability available and within 5%). Otherwise
                FAILED(reasons) with the raw delta in *_DIAGNOSTIC fields.
  flag_recovers : True only from a changed-inventory knob whose follow-up map
                (status ok, B complete) shows >= 60% (A)+(B) reduction WITH
                tight validation passing; False only when EVERY expected knob
                resolved (no-change screens, sub-60% traced reductions, or
                validation-disqualified knobs — the rule is recorded);
                anything unresolved => UNTESTED(reasons).

Run on the SAME HOST as the samplers/drivers: clock-window matching compares
local nvidia-smi timestamps with time.time() stamps from the same machine.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import time
from evidence_io import read_json, write_json

EXPECTED_KNOBS = ['layout_opt_true', 'layout_opt_false',
                  'keep_output_stride_false', 'permute_fusion_true',
                  'persistent_reductions_false', 'max_autotune',
                  'max_autotune_no_cudagraphs']


def j(path):
    return read_json(path)


def validation_state(record, tier):
    """None means missing/inconsistent evidence; False means explicit failure."""
    if not isinstance(record, dict):
        return None
    seeds = record.get('seeds', {})
    if set(seeds) != {'4242', '1337', '90210'}:
        return None
    expected = (1e-2, 1e-3) if tier == 'loose' else (1e-4, 1e-5)
    if any((v.get('rtol'), v.get('atol')) != expected or
           not isinstance(v.get('ok'), bool) for v in seeds.values()):
        return None
    all_pass = all(v['ok'] for v in seeds.values())
    stated = record.get('tight_all_pass', record.get('all_pass'))
    return all_pass if stated is all_pass else None


def maybe(path):
    return j(path) if path and os.path.exists(path) else None


def clock_window_median(clock_csv, blocks):
    """Median clocks.sm (MHz) over samples inside the timed-block windows.
    Sampler rows: timestamp, clocks.sm, clocks.mem, temp, power, util."""
    if not (clock_csv and os.path.exists(clock_csv) and blocks):
        return None
    wins = [(b['unix_start'], b['unix_end']) for b in blocks
            if 'unix_start' in b]
    if not wins:
        return None
    vals = []
    with open(clock_csv, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 2 or 'MHz' not in row[1]:
                continue
            try:
                ts = time.mktime(time.strptime(row[0].strip().split('.')[0],
                                               '%Y/%m/%d %H:%M:%S'))
                mhz = int(row[1].strip().split()[0])
            except (ValueError, IndexError):
                continue
            if any(a - 1 <= ts <= b + 1 for a, b in wins):
                vals.append(mhz)
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase0-dir', required=True)
    ap.add_argument('--phase1-dir', required=True)
    ap.add_argument('--phase2-dir', default=None)
    ap.add_argument('--clocks-baseline', default=None)
    ap.add_argument('--clocks-patched', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--failed-step', default=None)
    args = ap.parse_args()

    base = maybe(os.path.join(args.phase0_dir, 'baseline_timing.json')) or {}
    vbase = maybe(os.path.join(args.phase0_dir, 'validation_baseline.json'))
    rl = maybe(os.path.join(args.phase1_dir, 'r_layout_baseline.json'))
    if not base.get('min_of_medians_ms') or not rl:
        reason = args.failed_step or 'baseline timing or attribution unavailable'
        summary = {'R_layout': 'UNRESOLVED (' + reason + ')',
                   'R_recovered': 'FAILED (baseline prerequisites unavailable)',
                   'flag_recovers': 'UNTESTED (baseline prerequisites unavailable)',
                   'baseline_validation': vbase, 'failed_step': reason,
                   'use_amp': (vbase or base).get('use_amp')}
        write_json(args.out, summary)
        return
    lat_base = base['min_of_medians_ms']
    R_layout = rl['R_layout_vs_clean']
    AB_base_ms = rl['A_ms_per_pass'] + (rl['B_ms_per_pass'] or 0.0)
    baseline_evidence_ok = (
        not base.get('settle', {}).get('settle_capped', True)
        and validation_state(vbase, 'loose') is True)
    baseline_map_ok = rl.get('status') == 'ok'
    baseline_B_ok = rl.get('B_status') == 'ok'

    summary = {
        'latency_baseline_ms': lat_base,
        'baseline_median_spread_ms': base.get('median_spread_ms'),
        'baseline_throughput_ms_best': base.get('throughput_ms_per_pass_best'),
        'baseline_evidence_ok': baseline_evidence_ok,
        'baseline_validation': None if not vbase else {
            'all_pass': vbase['all_pass'],
            'max_abs_diff_per_seed': {k: v.get('max_abs_diff')
                                      for k, v in vbase['seeds'].items()}},
        'R_layout': R_layout,
        'R_layout_status': rl.get('status'),
        'R_layout_status_flags': rl.get('status_flags', []),
        'R_layout_is_lower_bound': not (baseline_map_ok and baseline_B_ok),
        'A_ms': rl['A_ms_per_pass'],
        'B_ms': rl['B_ms_per_pass'],
        'B_status': rl['B_status'],
        'AB_baseline_ms': AB_base_ms,
        'heads_split_are_views': rl.get('heads_split_are_views'),
        'heads_merge_are_views': rl.get('heads_merge_are_views'),
        'per_layer': rl['per_layer'],
        'phase2_threshold_met': R_layout >= 0.03,
        'threshold_note': 'Phase 2 gated on R_layout >= 0.03 (handoff §4)',
        'use_amp': base.get('use_amp'),
        'failed_step': args.failed_step,
    }

    p2 = args.phase2_dir if (args.phase2_dir and os.path.isdir(args.phase2_dir)) else None

    # ---------------- R_recovered with acceptance gates ----------------
    if not baseline_map_ok and not summary['phase2_threshold_met']:
        summary['R_recovered'] = 'FAILED (A-only lower bound cannot establish the skip threshold)'
    elif not summary['phase2_threshold_met']:
        summary['R_recovered'] = 'SKIPPED (R_layout<0.03)'
    elif not p2 or not os.path.exists(os.path.join(p2, 'patched_timing.json')):
        summary['R_recovered'] = 'FAILED (patched run artifacts missing)'
    else:
        pt = j(os.path.join(p2, 'patched_timing.json'))
        vp = maybe(os.path.join(p2, 'validation_patched.json')) or {}
        rlp = maybe(os.path.join(p2, 'r_layout_patched.json'))
        parity = maybe(os.path.join(p2, 'parity_cpu.json'))
        lat_p = pt.get('min_of_medians_ms')
        # clocks are computed BEFORE acceptance (gate-03 f24)
        cb = clock_window_median(args.clocks_baseline, base.get('blocks'))
        cp = clock_window_median(args.clocks_patched, pt.get('blocks'))
        clocks_ok = bool(cb and cp and abs(cb - cp) / cb <= 0.05)
        summary['clock_comparability'] = (
            {'baseline_sm_mhz_median': cb, 'patched_sm_mhz_median': cp,
             'within_5pct': clocks_ok} if cb and cp else
            'UNAVAILABLE (sampler csv or block timestamps missing)')
        gates = {
            'attempt_completed': args.failed_step is None,
            'baseline_evidence_ok': baseline_evidence_ok,
            'baseline_map_status_ok': baseline_map_ok,
            'baseline_B_complete': baseline_B_ok,
            'cpu_parity_pass': bool(parity and parity.get('all_pass')),
            'patched_settle_ok': not pt.get('settle', {}).get('settle_capped', True),
            'tight_validation_pass': validation_state(vp, 'tight') is True,
            'timing_is_headline': pt.get('status') != 'NOT_HEADLINE' and bool(lat_p),
            'patched_map_status_ok': bool(rlp and rlp.get('status') == 'ok'),
            'patched_B_complete': bool(rlp and rlp.get('B_status') == 'ok'),
            'in_stack_A_fell': bool(rlp and rlp['A_ms_per_pass'] < rl['A_ms_per_pass']),
            'patched_heads_split_views': bool(
                rlp and rlp.get('heads_split_are_views') == 'true'),
            'patched_heads_merge_views': bool(
                rlp and rlp.get('heads_merge_are_views') == 'true'),
            'patched_no_sdpa_adjacent_copies': bool(
                rlp and rlp.get('sdpa_adjacent_copy_events_rep_pass') == 0),
            'clocks_comparable_within_5pct': clocks_ok,
        }
        failed = [k for k, v in gates.items() if not v]
        rec = {
            'latency_patched_ms': lat_p,
            'patched_median_spread_ms': pt.get('median_spread_ms'),
            'patched_throughput_ms_best': pt.get('throughput_ms_per_pass_best'),
            'gates': gates,
            'raw_delta_ms_DIAGNOSTIC': (lat_base - lat_p) if lat_p else None,
            'raw_delta_pct_of_forward_DIAGNOSTIC':
                ((lat_base - lat_p) / lat_base) if lat_p else None,
            'R_recovered_DIAGNOSTIC':
                ((lat_base - lat_p) / (R_layout * lat_base))
                if (lat_p and R_layout > 0) else None,
            'A_patched_ms': rlp['A_ms_per_pass'] if rlp else None,
            'B_patched_ms': rlp['B_ms_per_pass'] if rlp else None,
            'patched_status_flags': rlp.get('status_flags', []) if rlp else None,
        }
        if failed:
            summary['R_recovered'] = f"FAILED ({'; '.join(failed)})"
        elif R_layout <= 0:
            summary['R_recovered'] = 'FAILED (R_layout <= 0)'
        else:
            summary['R_recovered'] = (lat_base - lat_p) / (R_layout * lat_base)
        summary['patched'] = rec

    # ---------------- flag_recovers ----------------
    knobs = {}
    if not summary['phase2_threshold_met']:
        verdict = 'UNTESTED (Phase 2 skipped: R_layout < 0.03)'
    elif not p2:
        verdict = 'UNTESTED (no phase-2 artifacts)'
    else:
        base_inv = maybe(os.path.join(args.phase1_dir, 'inventory_baseline.json'))
        any_true = False
        unresolved = []
        false_basis = []
        for name in EXPECTED_KNOBS:
            kdir = os.path.join(p2, f'knob_{name}')
            key = f'knob_{name}'
            r = maybe(os.path.join(kdir, 'knob_result.json'))
            inv = maybe(os.path.join(kdir, 'inventory.json'))
            rlk = maybe(os.path.join(kdir, f'r_layout_knob_{name}.json'))
            entry = {'status': (r or {}).get('status', 'MISSING'),
                     'tight_validation_pass':
                         (r or {}).get('validation', {}).get('tight_all_pass'),
                     'latency_median_ms': (r or {}).get('latency_median_ms'),
                     'latency_ratio_vs_baseline':
                         ((r or {}).get('latency_median_ms') / lat_base)
                         if (r or {}).get('latency_median_ms') else None}
            state = validation_state((r or {}).get('validation'), 'tight')
            if r is None or entry['status'] != 'ok':
                unresolved.append(f"{key}: {entry['status']}")
            elif state is None:
                unresolved.append(f'{key}: tight-validation evidence missing or inconsistent')
            elif not (baseline_evidence_ok and baseline_map_ok and baseline_B_ok):
                unresolved.append(f'{key}: baseline prerequisites not acceptance-grade')
            elif state is False:
                false_basis.append(f'{key}: explicitly failed tight validation')
                entry['flag_verdict'] = 'validation-disqualified'
            elif not (base_inv and inv and base_inv.get('status') == 'ok'
                      and inv.get('status') == 'ok'):
                unresolved.append(f'{key}: inventory coverage incomplete')
            elif rlk is None:
                # Counts alone cannot prove unchanged cost: a flag can improve
                # coalescing/bandwidth without deleting a materialization.
                unresolved.append(f'{key}: follow-up attribution map missing')
            elif rlk.get('status') != 'ok' or rlk.get('B_status') != 'ok':
                unresolved.append(f'{key}: follow-up map not acceptance-grade')
            else:
                ab_k = rlk['A_ms_per_pass'] + rlk['B_ms_per_pass']
                red = 1 - ab_k / AB_base_ms if AB_base_ms > 0 else None
                entry['AB_knob_ms'] = ab_k
                entry['AB_reduction_vs_baseline'] = red
                if red is None:
                    unresolved.append(f'{key}: baseline (A)+(B) is zero')
                elif red >= 0.60:
                    any_true = True
                    entry['flag_verdict'] = 'RECOVERS'
                else:
                    entry['flag_verdict'] = f'reduction {red:.6f} < 0.60'
                    false_basis.append(f'{key}: {entry["flag_verdict"]}')
            knobs[key] = entry
        if any_true:
            verdict = True
        elif unresolved:
            verdict = f"UNTESTED ({'; '.join(unresolved[:5])})"
        else:
            verdict = False
            summary['flag_recovers_false_basis'] = false_basis
        if base_inv:
            summary['inventory_baseline'] = {
                k: base_inv.get(k) for k in
                ('pure_copy_count', 'pure_copy_bytes', 'permuted_read_sites')}
    summary['knobs'] = knobs
    summary['flag_recovers'] = verdict

    write_json(args.out, summary)
    print(json.dumps({k: summary.get(k) for k in
                      ('latency_baseline_ms', 'R_layout', 'R_layout_status',
                       'R_layout_is_lower_bound', 'A_ms', 'B_ms',
                       'heads_split_are_views', 'R_recovered',
                       'flag_recovers')}, indent=1))
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
