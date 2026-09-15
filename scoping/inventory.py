"""Materialization inventory from an output_code dump alone (no trace) —
used to diff Phase-2 knob runs against the baseline cheaply (handoff §5:
"its own output_code dump and materialization count"). Reuses kernel_map's
role-aware parser (gate-02 f2: count permuted READS, not outputs) and its
two-signal permuted detection (gate-03 f20: call-site view meta OR kernel-
source load-vs-store index expressions).

Usage: inventory.py <dump_dir> <out_json>
"""

import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_map import (parse_output_code, arg_meta, is_permuted, nbytes,  # noqa: E402
                        PURE_COPY_OPS, B_SITE_MIN_READ_BYTES,
                        refined_access, read_permuted_state)
from evidence_io import write_json


def build_inventory(dump_dir):
    graphs = parse_output_code(dump_dir)
    inv = {'graphs': len(graphs), 'pure_copy_sites': [],
           'permuted_read_sites': 0, 'triton_calls': 0, 'extern_calls': 0,
           'fallback_calls': 0, 'ops_unknown_sites': 0, 'coverage_flags': []}
    if not graphs:
        inv['coverage_flags'].append('no generated graphs')
    for g in graphs:
        for c in g['calls']:
            inv[f"{c['kind']}_calls"] = inv.get(f"{c['kind']}_calls", 0) + 1
            if c['kind'] != 'triton':
                continue
            k = g['kernels'].get(c['name'], {})
            ops = k.get('ops', [])
            raw = k.get('ops_raw_name', '')
            if k.get('ops_source', 'unknown') == 'unknown':
                inv['ops_unknown_sites'] += 1
                inv['coverage_flags'].append(f"unknown ops: {c['name']}")
            pure = (bool(ops) and all(o.strip() in PURE_COPY_OPS for o in ops)) or \
                (k.get('ops_source') == 'name' and
                 re.fullmatch(r'(clone|copy|_to_copy|contiguous)', raw or '')
                 is not None)
            read_pairs, writes = [], []
            if not c.get('align_ok', False):
                inv['coverage_flags'].append(f"unaligned arguments: {c['name']}")
            for a in c['args']:
                r, w, unres = refined_access(k, a.get('role', ''))
                meta = arg_meta(g, a)
                if unres or not meta['dtype'] or not meta['shape']:
                    inv['coverage_flags'].append(f"unresolved access/bytes: {c['name']}")
                if r:
                    read_pairs.append((a, meta))
                if w:
                    writes.append(meta)
            reads = [m for _, m in read_pairs]
            if pure:
                inv['pure_copy_sites'].append(
                    {'kernel': c['name'],
                     'bytes': sum(nbytes(m['shape'], m['dtype'])
                                  for m in reads + writes),
                     'ops': ';'.join(ops) if ops else raw})
            perm = False
            for a, m in read_pairs:
                if nbytes(m['shape'], m['dtype']) < B_SITE_MIN_READ_BYTES:
                    continue
                state = read_permuted_state(k, a.get('role', ''), m)
                perm |= state == 'yes'
                if state == 'unresolved':
                    inv['coverage_flags'].append(f"unresolved permutation: {c['name']}")
            if perm:
                inv['permuted_read_sites'] += 1
    inv['pure_copy_count'] = len(inv['pure_copy_sites'])
    inv['pure_copy_bytes'] = sum(s['bytes'] for s in inv['pure_copy_sites'])
    inv['status'] = 'ok' if not inv['coverage_flags'] else 'FLAGGED'
    return inv


if __name__ == '__main__':
    inv = build_inventory(sys.argv[1])
    write_json(sys.argv[2], inv)
    print(f"[inventory] pure_copy={inv['pure_copy_count']} "
          f"({inv['pure_copy_bytes']:,} B) permuted_reads="
          f"{inv['permuted_read_sites']} unknown_ops={inv['ops_unknown_sites']}"
          f" -> {sys.argv[2]}")
