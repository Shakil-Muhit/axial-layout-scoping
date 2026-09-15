"""Regressions for the evidence acceptance failures found in gate 04."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCOPING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPING))
from evidence_io import read_json, write_json
from finalize import EXPECTED_KNOBS, validation_state
from kernel_map import access_expressions, read_permuted_state, refined_access, merge_path


def validation(tier='tight', ok=True):
    rtol, atol = (1e-4, 1e-5) if tier == 'tight' else (1e-2, 1e-3)
    return {'all_pass': ok, 'use_amp': True, 'seeds': {
        str(seed): {'ok': ok, 'rtol': rtol, 'atol': atol, 'max_abs_diff': 0.0}
        for seed in (4242, 1337, 90210)}}


class AccessTests(unittest.TestCase):
    def test_store_only_inout_and_unparsed_load(self):
        k = access_expressions('def f(in_out_ptr0):\n tl.store(in_out_ptr0 + (x0), v)\n', ['in_out_ptr0'])
        self.assertEqual(refined_access(k, 'in_out_ptr0'), (False, True, False))
        k['loads_present'] = ['in_out_ptr0']
        self.assertEqual(refined_access(k, 'in_out_ptr0'), (True, True, False))
        self.assertEqual(read_permuted_state(k, 'in_out_ptr0',
                         {'shape': (10,), 'stride': (1,)}), 'unresolved')

    def test_multiple_loads_and_outputs(self):
        source = '''def f(in_ptr0, out_ptr0, out_ptr1):
 a = tl.load(in_ptr0 + (x0 + (2*x1)))
 b = tl.load(in_ptr0 + (x2))
 tl.store(out_ptr0 + (x0), a)
 tl.store(out_ptr1 + (x1), b)
'''
        k = access_expressions(source, ['in_ptr0', 'out_ptr0', 'out_ptr1'])
        self.assertNotIn('in_ptr0', k['load_exprs'])
        k['load_exprs']['in_ptr0'] = 'x0'
        self.assertEqual(read_permuted_state(k, 'in_ptr0',
                         {'shape': (10,), 'stride': (1,)}), 'unresolved')

    def test_head_merge_follows_gate_to_clone(self):
        g = {'aliases': {}, 'allocs': {x: {'shape': (10,), 'stride': (1,),
              'dtype': 'torch.float16'} for x in ('sdpa', 'gated', 'copied')},
             'kernels': {'gate': {'ops': ['aten.mul'], 'load_exprs': {'in_ptr0': 'x0'},
                                 'store_exprs': {'out_ptr0': 'x0'}},
                         'clone': {'ops': ['aten.clone']}}}
        def arg(buf, write=False):
            return {'buf': buf, 'read': not write, 'write': write,
                    'role': 'out_ptr0' if write else 'in_ptr0'}
        wrapper = [
            {'g': 0, 'gr': g, 'name': 'sdpa', 'assigned': 'sdpa'},
            {'g': 0, 'gr': g, 'name': 'gate', 'kind': 'triton', 'align_ok': True,
             'args': [arg('sdpa'), arg('gated', True)]},
            {'g': 0, 'gr': g, 'name': 'clone', 'kind': 'triton', 'align_ok': True,
             'args': [arg('gated'), arg('copied', True)]},
            {'g': 0, 'gr': g, 'name': 'extern.mm', 'kind': 'extern', 'args': [arg('copied')]}]
        writers = {'sdpa': 0, 'gated': 1, 'copied': 2}
        self.assertEqual(merge_path(wrapper, 0, lambda g, b, p: writers.get(b))[0], 'false')


class AcceptanceTests(unittest.TestCase):
    def test_missing_seed_is_unresolved(self):
        v = validation()
        del v['seeds']['1337']
        self.assertIsNone(validation_state(v, 'tight'))
        self.assertIs(validation_state(validation(ok=False), 'tight'), False)

    def test_generation_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            m, artifact = Path(td) / 'manifest.json', Path(td) / 'artifact.json'
            m.write_text(json.dumps({'generation_id': 'new', 'use_amp': True}))
            artifact.write_text(json.dumps({'generation_id': 'old'}))
            with patch.dict(os.environ, {'AXIAL_MANIFEST': str(m)}):
                with self.assertRaisesRegex(ValueError, 'different or missing generation'):
                    read_json(artifact)
                write_json(artifact, {'status': 'ok'})
                self.assertEqual(read_json(artifact)['generation_id'], 'new')

    def finalize(self, flagged=False, failed_step=None, missing_validation=False):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def put(rel, obj):
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(obj))
            put('p0/baseline_timing.json', {'min_of_medians_ms': 10.0, 'use_amp': True,
                                         'settle': {'settle_capped': False}, 'blocks': []})
            put('p0/validation_baseline.json', validation('loose'))
            rm = {'R_layout_vs_clean': 0.1, 'A_ms_per_pass': 0.8, 'B_ms_per_pass': 0.2,
                  'B_status': 'ok', 'status': 'FLAGGED' if flagged else 'ok',
                  'per_layer': {}, 'heads_split_are_views': 'true',
                  'heads_merge_are_views': 'true', 'sdpa_adjacent_copy_events_rep_pass': 0}
            put('p1/r_layout_baseline.json', rm)
            inv = {'status': 'ok', 'pure_copy_count': 12, 'pure_copy_bytes': 100,
                   'permuted_read_sites': 12}
            put('p1/inventory_baseline.json', inv)
            for name in EXPECTED_KNOBS:
                r = {'status': 'ok'}
                if not missing_validation:
                    r['validation'] = validation()
                put(f'p2/knob_{name}/knob_result.json', r)
                put(f'p2/knob_{name}/inventory.json', inv)
            put('p2/patched_timing.json', {'min_of_medians_ms': 9.3,
                                         'settle': {'settle_capped': False}, 'blocks': []})
            put('p2/validation_patched.json', validation())
            put('p2/parity_cpu.json', {'all_pass': True})
            put('p2/r_layout_patched.json', {**rm, 'status': 'ok', 'A_ms_per_pass': 0.1})
            command = [sys.executable, str(SCOPING / 'finalize.py'),
                       '--phase0-dir', str(root / 'p0'), '--phase1-dir', str(root / 'p1'),
                       '--phase2-dir', str(root / 'p2'), '--out', str(root / 'summary.json')]
            if failed_step:
                command += ['--failed-step', failed_step]
            env = {k: v for k, v in os.environ.items() if k != 'AXIAL_MANIFEST'}
            subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            return json.loads((root / 'summary.json').read_text())

    def test_unchanged_inventory_needs_measured_cost(self):
        self.assertIsInstance(self.finalize()['flag_recovers'], str)

    def test_flagged_baseline_cannot_produce_false(self):
        self.assertTrue(self.finalize(flagged=True)['flag_recovers'].startswith('UNTESTED'))

    def test_missing_validation_is_unresolved(self):
        self.assertIn('validation', self.finalize(missing_validation=True)['flag_recovers'])

    def test_failed_attempt_rejects_stale_map(self):
        r = self.finalize(failed_step='patched trace')
        self.assertIn('attempt_completed', r['R_recovered'])
        self.assertFalse(r['patched']['gates']['attempt_completed'])


if __name__ == '__main__':
    unittest.main()
