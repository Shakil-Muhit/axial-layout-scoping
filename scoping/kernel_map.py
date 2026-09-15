"""Phase 1c/1d/1e — Inductor kernel map, materialization inventory, R_layout.
v3 after codex gates 02+03 (docs/codex/02_code.md, 03_delta.md). Key
properties the owner can audit:

  * kernel metadata keyed by the WRAPPER symbol (assignment LHS); the inner
    def name is recorded separately (gate-03 f19); the def and its load/store
    index expressions are read from that kernel's own source slice.
  * ordered, role-tagged args from the def's in_ptr/out_ptr/in_out_ptr
    params; plain buffer-reuse lines (`bufA = bufB; del bufB`) are aliases;
    strict arg/param alignment — misaligned or dtype-unresolved sites are
    EXCLUDED from (B) and flagged, never silently priced at fp16 (f20).
  * permuted-read detection has two signals: the call-site view meta AND the
    kernel-source load-vs-store index expressions (Inductor bakes permuted
    indexing into the kernel and passes a bare base pointer, so wrapper
    strides alone undercount). Over-detection is safe: a coalesced kernel's
    measured duration ~= bytes/BW_ref, so its clamped penalty ~= 0; only
    UNDER-detection loses (B) — which the coverage counters flag (f20).
  * pass splitting: NVTX pass_<i> projected windows (primary; gap heuristic
    is a FLAGGED fallback); n_pass must equal the meta iteration count and
    every pass must hold exactly the wrapper's SDPA anchor count, else FATAL
    without --allow-pass-mismatch; per-name durations come only from rows
    inside the validated windows (f23).
  * name->trace matching: exact, else a UNIQUE prefix candidate that does not
    extend a trailing ordinal (no foo_1 borrowing foo_10); per-name
    launches-per-pass must equal wrapper multiplicity or the row is flagged;
    kern_sum aggregate cross-check recorded (f23).
  * stack membership: interior SDPA window + kernel-NAME reuse for the L0
    prologue / L5 epilogue. The dedup assumption is CHECKED post-hoc: an
    L0-vs-interior-layers A/B asymmetry raises a flag (f22).
  * producer map is position-aware (latest write BEFORE the consumer);
    head-split and head-merge verdicts are tri-state (true/false/unresolved)
    with unknown ancestry never counting as a view (f26); COPY-class launches
    adjacent to SDPA in the trace are counted (silent-failure signal).
  * the CSV is the ledger; every JSON total is summed from it. status='ok'
    ONLY when no flag was raised — downstream acceptance requires that.
"""

import argparse
import ast
import bisect
import csv
import glob
import json
import os
import re
from evidence_io import read_json, write_json

DTYPE_BYTES = {'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4,
               'torch.float64': 8, 'torch.int64': 8, 'torch.int32': 4,
               'torch.int8': 1, 'torch.uint8': 1, 'torch.bool': 1,
               'torch.complex64': 8}

PURE_COPY_OPS = {'aten.clone', 'aten._to_copy', 'aten.copy', 'aten.copy_',
                 'aten.contiguous', 'prims.convert_element_type',
                 'aten.constant_pad_nd'}  # pad included: layout-only data move
NORM_OP_HINTS = ('norm', 'var_mean', 'welford', 'rsqrt', 'mean', 'pow')
B_SITE_MIN_READ_BYTES = 1_000_000   # ignore broadcast/tiny reads for (B)

ASSIGN_TRITON_RE = re.compile(r"^(\w+)\s*=\s*async_compile\.triton\('(\w+)'", re.M)
ATEN_COMMENT_RE = re.compile(r'Original ATen:\s*\[([^\]]*)\]')
ALLOC_RE = re.compile(
    r'^\s*(\w+)\s*=\s*empty_strided_(?:cuda|xpu)\(\((.*?)\),\s*\((.*?)\),\s*(torch\.\w+)', re.M)
ALIAS_RE = re.compile(
    r'^\s*(\w+)\s*=\s*reinterpret_tensor\((\w+),\s*\((.*?)\),\s*\((.*?)\)[,)]', re.M)
RENAME_RE = re.compile(r'^\s*(buf\d+)\s*=\s*(buf\d+)\s*(?:;|#|$)', re.M)
GETITEM_RE = re.compile(r'^\s*(\w+)\s*=\s*(\w+)\[(\d+)\]\s*$', re.M)
ASSERT_RE = re.compile(
    r'^\s*assert_size_stride\((\w+),\s*\((.*?)\),\s*\((.*?)\)', re.M)
TRITON_CALL_RE = re.compile(r'^\s*(triton_\w+)\.run\((.*)\)', re.M)
EXTERN_CALL_RE = re.compile(
    r'^\s*(?:(\w+)\s*=\s*)?extern_kernels\.(\w+)\((.*)\)', re.M)
FALLBACK_RE = re.compile(
    r'^\s*(\w+)\s*=\s*torch\.ops\.(?:aten|prims)\.(\w+)[\.\w]*\((.*)\)', re.M)
ARG_TOKEN_RE = re.compile(
    r'reinterpret_tensor\((\w+),\s*\(([^)]*)\),\s*\(([^)]*)\)(?:,\s*\d+)?\)'
    r'|(?<![\w.])(buf\d+|arg\d+_\d+|primals_\d+)(?![\w])')


def _ints(s):
    return tuple(int(t) for t in re.findall(r'-?\d+', s))


def ops_from_name(name):
    m = re.match(r'triton_\w+?_fused_(.+?)(?:_\d+)?$', name)
    return m.group(1) if m else ''


def _canon_expr(e):
    return re.sub(r'\s+', '', e or '')


def access_expressions(source, params):
    """Parse complete pointer expressions, including multiple accesses.
    A regex ending at the first ')' loses nested arithmetic and later loads.
    """
    exprs = {'load': {}, 'store': {}}
    present = {'load': set(), 'store': set()}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        op = node.func.attr
        if op not in exprs or not node.args:
            continue
        pointer = node.args[0]
        names = {n.id for n in ast.walk(pointer) if isinstance(n, ast.Name)}
        for p in set(params) & names:
            present[op].add(p)
            offset = None
            if isinstance(pointer, ast.Name) and pointer.id == p:
                offset = '0'
            elif isinstance(pointer, ast.BinOp) and isinstance(pointer.op, ast.Add):
                if isinstance(pointer.left, ast.Name) and pointer.left.id == p:
                    offset = _canon_expr(ast.unparse(pointer.right))
            exprs[op].setdefault(p, set()).add(offset)
    result = {}
    for op in exprs:
        result[op + '_exprs'] = {p: next(iter(values)) for p, values in exprs[op].items()
                                if len(values) == 1 and None not in values}
        result[op + 's_present'] = sorted(present[op])
    return result


def parse_output_code(dump_dir):
    """Parse every output_code.py: kernel defs (ops, param roles, load/store
    index exprs) keyed by wrapper symbol, and the call() body in execution
    order with ORDERED, role-tagged args."""
    files = sorted(glob.glob(os.path.join(dump_dir, '**', 'output_code.py'),
                             recursive=True))
    graphs = []
    for path in files:
        src = open(path).read()
        source_literals = {}
        for node in ast.parse(src).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'triton'
                    and len(node.value.args) >= 2
                    and isinstance(node.value.args[1], ast.Constant)
                    and isinstance(node.value.args[1].value, str)):
                source_literals[node.targets[0].id] = node.value.args[1].value
        assigns = list(ASSIGN_TRITON_RE.finditer(src))
        kernels = {}
        for idx, am in enumerate(assigns):
            lhs, inner = am.group(1), am.group(2)
            lo = assigns[idx - 1].end() if idx else 0
            hi = assigns[idx + 1].start() if idx + 1 < len(assigns) else len(src)
            window = src[lo:am.start()]
            cm = None
            for cm in ATEN_COMMENT_RE.finditer(window):
                pass                      # LAST comment before this assignment
            k = {'inner_name': inner, 'inner_mismatch': inner != lhs}
            if cm:
                k['ops'] = [o.strip() for o in cm.group(1).split(',') if o.strip()]
                k['ops_source'] = 'aten_comment'
            else:
                raw = ops_from_name(lhs)
                k['ops'] = []
                k['ops_source'] = 'name' if raw else 'unknown'
                k['ops_raw_name'] = raw
            body = src[am.end():hi]       # this kernel's own source slice
            pm = re.search(r'def\s+' + re.escape(inner) + r'\s*\(([^)]*)\)', body)
            params = []
            if pm:
                for tok in pm.group(1).split(','):
                    tok = tok.split(':')[0].strip()
                    if tok.startswith(('in_ptr', 'out_ptr', 'in_out_ptr')):
                        params.append(tok)
            k['params'] = params
            k['load_exprs'] = {}
            k['store_exprs'] = {}
            k['loads_present'] = []
            k['stores_present'] = []
            for p in params:
                if re.search(r'tl\.load\(\s*' + re.escape(p) + r'\b', body):
                    k['loads_present'].append(p)
                if re.search(r'tl\.store\(\s*' + re.escape(p) + r'\b', body):
                    k['stores_present'].append(p)
                lm = re.search(r'tl\.load\(' + re.escape(p) + r'\s*\+\s*\(([^)]*)\)', body)
                sm2 = re.search(r'tl\.store\(' + re.escape(p) + r'\s*\+\s*\(([^)]*)\)', body)
                if lm:
                    k['load_exprs'][p] = _canon_expr(lm.group(1))
                if sm2:
                    k['store_exprs'][p] = _canon_expr(sm2.group(1))
            if lhs in source_literals:
                k.update(access_expressions(source_literals[lhs], params))
            kernels[lhs] = k

        try:
            body = src.split('def call(')[1]
        except IndexError:
            continue
        allocs, aliases, calls = {}, {}, []
        body_start = src[:src.index('def call(')].count('\n') + 1
        for body_line, line in enumerate(body.splitlines(), body_start):
            am = ALLOC_RE.match(line)
            if am:
                allocs[am.group(1)] = {'shape': _ints(am.group(2)),
                                       'stride': _ints(am.group(3)),
                                       'dtype': am.group(4)}
                continue
            xm = ALIAS_RE.match(line)
            if xm:
                aliases[xm.group(1)] = {'base': xm.group(2),
                                        'shape': _ints(xm.group(3)),
                                        'stride': _ints(xm.group(4))}
                continue
            rm = RENAME_RE.match(line)
            if rm and rm.group(1) != rm.group(2):
                aliases[rm.group(1)] = {'base': rm.group(2)}   # plain reuse
                continue
            gm = GETITEM_RE.match(line)
            if gm:
                aliases[gm.group(1)] = {'base': gm.group(2),
                                        'item': int(gm.group(3))}
                continue
            sm = ASSERT_RE.match(line)
            if sm:
                allocs.setdefault(sm.group(1), {'shape': _ints(sm.group(2)),
                                                'stride': _ints(sm.group(3)),
                                                'dtype': None})
                continue
            tm = TRITON_CALL_RE.match(line)
            em = EXTERN_CALL_RE.match(line)
            fm = FALLBACK_RE.match(line)
            if not (tm or em or fm):
                continue
            if tm:
                kind, name, argstr, assigned = 'triton', tm.group(1), tm.group(2), None
            elif em:
                kind, name, argstr, assigned = ('extern', f'extern.{em.group(2)}',
                                                em.group(3), em.group(1))
            else:
                kind, name, argstr, assigned = ('fallback', f'aten.{fm.group(2)}',
                                                fm.group(3), fm.group(1))
            out_kw = re.search(r'\bout\s*=', argstr)
            args = []
            for mt in ARG_TOKEN_RE.finditer(argstr):
                if mt.group(1):
                    a = {'buf': mt.group(1), 'shape': _ints(mt.group(2)),
                         'stride': _ints(mt.group(3)), 'reinterpret': True}
                else:
                    a = {'buf': mt.group(4), 'shape': None, 'stride': None,
                         'reinterpret': False}
                a['in_out_kw'] = bool(out_kw and mt.start() > out_kw.start())
                args.append(a)
            if kind == 'triton':
                params = kernels.get(name, {}).get('params', [])
                align_ok = len(params) == len(args)
                for i, a in enumerate(args):
                    role = params[i] if i < len(params) else ''
                    a['read'] = role.startswith(('in_ptr', 'in_out_ptr'))
                    a['write'] = role.startswith(('out_ptr', 'in_out_ptr'))
                    a['role'] = role or 'unaligned'
                    a['read'], a['write'], a['access_unresolved'] = refined_access(
                        kernels.get(name, {}), a['role'])
                calls.append({'kind': kind, 'name': name, 'args': args,
                              'align_ok': align_ok, 'assigned': None,
                              'line': line.strip()[:400]})
            elif kind == 'extern':
                for a in args:
                    a['write'] = a['in_out_kw']
                    a['read'] = not a['in_out_kw']
                    a['role'] = 'out' if a['write'] else 'in'
                calls.append({'kind': kind, 'name': name, 'args': args,
                              'align_ok': True, 'assigned': assigned,
                              'line': line.strip()[:400]})
            else:
                for a in args:
                    a['read'], a['write'], a['role'] = True, False, 'in'
                calls.append({'kind': kind, 'name': name, 'args': args,
                              'align_ok': True, 'assigned': assigned,
                              'line': line.strip()[:400]})
            calls[-1]['source_line'] = body_line
        graphs.append({'path': path, 'calls': calls, 'allocs': allocs,
                       'aliases': aliases, 'kernels': kernels})
    return graphs


# ------------------------------------------------------------- resolution ---

def resolve(gr, buf):
    """Follow alias chain to the base buffer; return (base, view_meta|None)."""
    meta = None
    seen = set()
    while buf in gr['aliases'] and buf not in seen:
        seen.add(buf)
        a = gr['aliases'][buf]
        if meta is None and 'shape' in a:
            meta = {'shape': a['shape'], 'stride': a['stride']}
        buf = a['base']
    return buf, meta


def arg_meta(gr, a):
    """Effective (shape, stride, dtype, base) the kernel sees for arg a."""
    base, alias_meta = resolve(gr, a['buf'])
    alloc = gr['allocs'].get(base, {})
    if a.get('reinterpret') and a.get('shape'):
        shape, stride = a['shape'], a['stride']
    elif alias_meta:
        shape, stride = alias_meta['shape'], alias_meta['stride']
    else:
        shape, stride = alloc.get('shape', ()), alloc.get('stride', ())
    return {'shape': shape or (), 'stride': stride or (),
            'dtype': alloc.get('dtype'), 'base': base}


def refined_access(k, role):
    """(reads, writes, unresolved) for a param role, derived from the kernel
    SOURCE (gate-04 f32): in_out_ptr reads iff a tl.load on it exists and
    writes iff a tl.store exists; neither found -> unresolved (charged
    conservatively as read+write, and the site is flagged)."""
    if role.startswith('in_out_ptr'):
        r = role in k.get('loads_present', k.get('load_exprs', {}))
        w = role in k.get('stores_present', k.get('store_exprs', {}))
        return (r, w, not (r or w))
    if role.startswith('in_ptr'):
        return (True, False, role not in k.get('loads_present', k.get('load_exprs', {})))
    if role.startswith('out_ptr'):
        return (False, True, role not in k.get('stores_present', k.get('store_exprs', {})))
    return (False, False, True)


def read_permuted_state(k, role, meta):
    """Tri-state permuted classification for one READING param (gate-04 f32):
    'yes'  — view meta is permuted, or the load index expr differs from the
             kernel's single store expr;
    'no'   — load expr equals the single store expr;
    'unresolved' — load expr missing, or zero / multiple distinct store exprs
             (an unrelated output with the same expr could hide a permuted
             read feeding another output)."""
    stores = set(k.get('store_exprs', {}).values())
    le = k.get('load_exprs', {}).get(role)
    expected_stores = set(k.get('stores_present', k.get('store_exprs', {})))
    if le is None or expected_stores != set(k.get('store_exprs', {})):
        return 'unresolved'
    if is_permuted(meta['shape'], meta['stride']):
        return 'yes'
    if len(stores) == 1:
        return 'yes' if le not in stores else 'no'
    return 'unresolved'


def is_permuted(shape, stride):
    """Permuted = the stride ORDER differs from row-major: fastest dim not
    unit-stride OR strides (over size>1 dims) not non-increasing."""
    if not shape or len(stride) != len(shape):
        return False
    nt = [(s, x) for s, x in zip(shape, stride) if s > 1]
    if not nt:
        return False
    if nt[-1][1] != 1:
        return True
    return any(nt[i][1] < nt[i + 1][1] for i in range(len(nt) - 1))


def nbytes(shape, dtype):
    n = 1
    for s in shape or ():
        n *= s
    return n * DTYPE_BYTES.get(dtype or 'torch.float16', 2)


def merge_path(wrapper, pos, producer_before):
    """Follow the SDPA output through gating to its projection, not just its
    first consumer. Unknown ops/accesses keep the verdict unresolved.
    Storage reuse is qualified by the reaching writer's position.
    """
    start = wrapper[pos]
    if not start.get('assigned'):
        return 'unresolved', ['missing SDPA output']
    live = {start['assigned']: pos}
    path = []
    for i in range(pos + 1, len(wrapper)):
        c = wrapper[i]
        if c['g'] != start['g'] or 'scaled_dot_product' in c['name']:
            break
        hits = []
        for a in c['args']:
            base, _ = resolve(c['gr'], a['buf'])
            if (a.get('read') and base in live and
                    producer_before(c['g'], base, i) == live[base]):
                hits.append(a)
        if not hits:
            continue
        path.append(c['name'])
        if c['kind'] == 'extern' and c['name'] in ('extern.mm', 'extern.addmm', 'extern.bmm'):
            return 'true', path
        k = c['gr']['kernels'].get(c['name'], {})
        ops = k.get('ops', [])
        if any(o in ('aten.clone', 'aten.copy', 'aten.copy_', 'aten.contiguous') for o in ops):
            return 'false', path
        if c['kind'] != 'triton' or not ops or not c.get('align_ok', False):
            return 'unresolved', path
        # Gating is arithmetic, but it must not conceal a fused layout copy.
        if any(o not in ('aten.mul', 'aten.sigmoid', 'aten._to_copy',
                         'prims.convert_element_type') for o in ops):
            return 'unresolved', path
        for a in hits:
            state = read_permuted_state(k, a.get('role', ''), arg_meta(c['gr'], a))
            if state != 'no':
                return ('false' if state == 'yes' else 'unresolved'), path
        outputs = [a for a in c['args'] if a.get('write')]
        if not outputs:
            return 'unresolved', path
        for a in outputs:
            live[resolve(c['gr'], a['buf'])[0]] = i
    return 'unresolved', path


def chain_class(ops, raw_name):
    text = ' '.join(ops).lower() if ops else (raw_name or '').lower()
    if any(h in text for h in NORM_OP_HINTS):
        return 'norm'
    if ops and all(o.strip() in PURE_COPY_OPS for o in ops):
        return 'copy'
    return 'pointwise'


def sdpa_axis(q_shape):
    if len(q_shape or ()) == 4:
        if q_shape[2] == 801:
            return 'time'
        if q_shape[2] == 60:
            return 'freq'
    return 'unknown'


# ------------------------------------------------------------------ trace ---

def read_gpu_trace(path):
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        def col(*names):
            for name in names:
                for i, h in enumerate(head):
                    if h.strip().strip('"') == name:
                        return i
            return next(i for i, h in enumerate(head)
                        if names[0].split(' ')[0] in h)
        c_start = col('Start (ns)', 'Start(ns)', 'Start')
        c_dur = col('Duration (ns)', 'Duration(ns)', 'Duration')
        c_name = col('Name')
        rows = []
        for r in rd:
            if len(r) <= max(c_start, c_dur, c_name) or not r[c_dur].strip():
                continue
            try:
                rows.append({'start': int(float(r[c_start].replace(',', ''))),
                             'dur': int(float(r[c_dur].replace(',', ''))),
                             'name': r[c_name]})
            except ValueError:
                continue
    rows.sort(key=lambda x: x['start'])
    return rows


def read_kern_sum(path):
    """name -> total ns from the aggregate export (cross-check only)."""
    out = {}
    try:
        with open(path, newline='') as f:
            rd = csv.reader(f)
            head = next(rd)
            c_t = next(i for i, h in enumerate(head) if 'Total Time' in h)
            c_n = next(i for i, h in enumerate(head)
                       if h.strip().strip('"') == 'Name')
            for r in rd:
                if len(r) > max(c_t, c_n) and r[c_t].strip():
                    try:
                        out[r[c_n]] = out.get(r[c_n], 0) + int(
                            float(r[c_t].replace(',', '')))
                    except ValueError:
                        continue
    except Exception as e:
        print(f'[map] WARN: kern_sum parse failed: {e}')
    return out


def nvtx_pass_windows(path):
    """pass_<i> projected ranges; require unique ids and non-overlap."""
    try:
        with open(path, newline='') as f:
            rd = csv.reader(f)
            head = next(rd)
            c_name = next(i for i, h in enumerate(head)
                          if h.strip().strip('"') in ('Name', 'Range'))
            c_start = next(i for i, h in enumerate(head) if 'Start' in h)
            c_dur = next(i for i, h in enumerate(head) if 'Duration' in h)
            wins = {}
            for r in rd:
                nm = r[c_name].strip().strip('"').split(':')[-1]
                if re.fullmatch(r'pass_\d+', nm):
                    try:
                        s = int(float(r[c_start].replace(',', '')))
                        d = int(float(r[c_dur].replace(',', '')))
                    except ValueError:
                        continue
                    i = int(nm.split('_')[1])
                    if i in wins:
                        return [], f'duplicate nvtx pass id {i}'
                    wins[i] = (s, s + d)
            ids = sorted(wins)
            if ids != list(range(len(ids))):
                return [], f'non-contiguous nvtx pass ids ({len(ids)} found)'
            ordered = [wins[i] for i in ids]
            for a, b in zip(ordered, ordered[1:]):
                if b[0] < a[1]:
                    return [], 'overlapping nvtx pass windows'
            return ordered, None
    except Exception as e:
        return [], f'nvtx parse failed: {e}'


def split_passes(rows, windows, classify):
    if windows:
        return [[r for r in rows if s <= r['start'] < e]
                for s, e in windows], 'nvtx-windows'
    sdpa_idx = [i for i, r in enumerate(rows) if classify(r['name']) == 'SDPA']
    n_pass = max(1, round(len(sdpa_idx) / 12)) if sdpa_idx else 1
    gaps = [(rows[i + 1]['start'] - (rows[i]['start'] + rows[i]['dur']), i)
            for i in range(len(rows) - 1)]
    cuts = sorted(i for _, i in sorted(gaps, key=lambda g: -g[0])[:n_pass - 1])
    passes, prev = [], 0
    for c in cuts:
        passes.append(rows[prev:c + 1])
        prev = c + 1
    passes.append(rows[prev:])
    return passes, 'gap-heuristic'


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-code-dir', required=True)
    ap.add_argument('--gpu-trace-csv', required=True)
    ap.add_argument('--kern-sum-csv', required=True)
    ap.add_argument('--nvtx-csv', default=None)
    ap.add_argument('--export-status', default=None)
    ap.add_argument('--trace-meta', required=True)
    ap.add_argument('--baseline-timing', required=True)
    ap.add_argument('--bwref', default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--allow-pass-mismatch', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from classify import classify

    status_flags = []
    export_status = read_json(args.export_status) if args.export_status else {}
    if export_status.get('status') != 'ok':
        status_flags.append('required-nsys-exports-incomplete-or-unverified')
    graphs = parse_output_code(args.output_code_dir)
    n_calls = sum(len(g['calls']) for g in graphs)
    assert graphs and n_calls > 0, \
        f'output_code parse produced nothing from {args.output_code_dir}'
    print(f'[map] parsed {len(graphs)} graphs, {n_calls} wrapper calls')

    wrapper = []
    for gi, g in enumerate(graphs):
        for c in g['calls']:
            wrapper.append({'g': gi, 'gr': g, **c})

    # position-aware producer map: (graph, base) -> sorted write positions
    producers = {}
    for i, c in enumerate(wrapper):
        for a in c['args']:
            if a.get('write'):
                base, _ = resolve(c['gr'], a['buf'])
                producers.setdefault((c['g'], base), []).append(i)
        if c.get('assigned'):
            producers.setdefault((c['g'], c['assigned']), []).append(i)

    def producer_before(g, base, pos):
        lst = producers.get((g, base))
        if not lst:
            return None
        k = bisect.bisect_left(lst, pos) - 1
        return lst[k] if k >= 0 else None

    sdpa_pos = [i for i, c in enumerate(wrapper)
                if c['kind'] == 'fallback' and 'scaled_dot_product' in c['name']]
    axes = []
    for pos in sdpa_pos:
        q = arg_meta(wrapper[pos]['gr'], wrapper[pos]['args'][0]) \
            if wrapper[pos]['args'] else {'shape': ()}
        axes.append(sdpa_axis(q['shape']))
    print(f'[map] SDPA wrapper calls: {len(sdpa_pos)} axes={axes}')
    if axes and axes != ['time', 'freq'] * (len(axes) // 2):
        status_flags.append(f'sdpa-axis-order-unexpected:{axes}')

    seg_of_call = {}
    if sdpa_pos:
        for k, pos in enumerate(sdpa_pos):
            lo = 0 if k == 0 else sdpa_pos[k - 1] + 1
            for i in range(lo, pos + 1):
                seg_of_call[i] = f'L{k // 2}_{axes[k]}'
        for i in range(sdpa_pos[-1] + 1, len(wrapper)):
            seg_of_call[i] = f'L{(len(sdpa_pos) - 1) // 2}_{axes[-1]}_post'

    interior = set(range(sdpa_pos[0] + 1, sdpa_pos[-1] + 1)) if sdpa_pos else set()
    names_interior = {wrapper[i]['name'] for i in interior}
    def in_stack(i):
        return i in interior or i in set(sdpa_pos) or (
            wrapper[i]['name'] in names_interior)

    # ---- 1d: head-split/merge tri-state verdicts (gate-03 f26)
    def ancestry(c, a, pos):
        meta = arg_meta(c['gr'], a)
        pi = producer_before(c['g'], meta['base'], pos)
        if pi is None:
            return 'graph-input-or-unknown', None
        pc = wrapper[pi]
        if pc['kind'] == 'extern':
            return ('view-of-gemm' if a.get('reinterpret') or
                    meta['base'] != a['buf'] else 'gemm-output'), pi
        if pc['kind'] == 'triton':
            k = pc['gr']['kernels'].get(pc['name'], {})
            ops = k.get('ops', [])
            if ops and all(o.strip() in PURE_COPY_OPS for o in ops):
                return f"materialized-copy({pc['name']})", pi
            if not ops:
                return f"unknown-ops({pc['name']})", pi
            return f"compute-output({pc['name']})", pi
        return f"{pc['kind']}-output({pc['name']})", pi

    def tri(verdicts, bad_prefix='materialized-copy',
            unknown_prefixes=('graph-input-or-unknown', 'unknown-ops')):
        if any(v.startswith(bad_prefix) for v in verdicts):
            return 'false'
        if any(v.startswith(p) for v in verdicts for p in unknown_prefixes):
            return 'unresolved'
        return 'true'

    head_verdicts = []
    split_states, merge_states = [], []
    for k, pos in enumerate(sdpa_pos):
        c = wrapper[pos]
        rec = {'wrapper_index': pos, 'segment': seg_of_call.get(pos),
               'axis': axes[k], 'inputs': [], 'line': c['line'][:200],
               'graph': c['gr']['path']}
        for a in c['args'][:3]:
            kind_, _ = ancestry(c, a, pos)
            rec['inputs'].append(kind_)
        rec['split_verdict'] = tri(rec['inputs'])
        split_states.append(rec['split_verdict'])
        rec['merge_verdict'], path = merge_path(wrapper, pos, producer_before)
        rec['merge_consumer'] = ' -> '.join(path)
        merge_states.append(rec['merge_verdict'])
        head_verdicts.append(rec)

    def overall(states):
        if not states:
            return 'unresolved'
        if 'false' in states:
            return 'false'
        if 'unresolved' in states:
            return 'unresolved'
        return 'true'
    heads_split = overall(split_states)
    heads_merge = overall(merge_states)
    if heads_split == 'unresolved':
        status_flags.append('head-split-ancestry-unresolved')
    if heads_merge == 'unresolved':
        status_flags.append('head-merge-ancestry-unresolved')

    # ---- trace: validated passes, per-name durations
    rows = read_gpu_trace(args.gpu_trace_csv)
    meta = read_json(args.trace_meta)
    iters = int(meta.get('iters_completed') or meta.get('iters') or 0)
    windows, nvtx_err = ([], None)
    if args.nvtx_csv:
        windows, nvtx_err = nvtx_pass_windows(args.nvtx_csv)
    passes, split_method = split_passes(rows, windows, classify)
    passes = [p for p in passes if p]
    n_pass = len(passes)
    print(f'[map] trace launches={len(rows)} passes={n_pass} '
          f'({split_method}); meta iters={iters}')
    if split_method != 'nvtx-windows':
        status_flags.append(
            f'pass-split-gap-heuristic ({nvtx_err or "no nvtx csv"})')
    if iters and n_pass != iters:
        msg = f'pass-count {n_pass} != iters {iters} ({split_method})'
        if args.allow_pass_mismatch:
            status_flags.append('PASS_MISMATCH: ' + msg)
        else:
            raise SystemExit(f'FATAL: {msg} — rerun or pass '
                             f'--allow-pass-mismatch to emit flagged output')
    sdpa_per_pass = [sum(1 for r in p if classify(r['name']) == 'SDPA')
                     for p in passes]
    bad = [i for i, n in enumerate(sdpa_per_pass) if n != len(sdpa_pos)]
    if bad:
        msg = (f'sdpa-anchor-mismatch in {len(bad)}/{n_pass} passes '
               f'(expected {len(sdpa_pos)}/pass)')
        if args.allow_pass_mismatch:
            status_flags.append(msg)
        else:
            raise SystemExit(f'FATAL: {msg} — rerun or pass '
                             f'--allow-pass-mismatch to emit flagged output')

    rows_in = [r for p in passes for r in p]   # validated union only (f23)
    per_name = {}
    for r in rows_in:
        d = per_name.setdefault(r['name'], {'total_ns': 0, 'launches': 0})
        d['total_ns'] += r['dur']
        d['launches'] += 1
    rep = max(passes, key=len)

    def match_trace(name):
        if name in per_name:
            return per_name[name], 'exact'
        cands = [k for k in per_name
                 if k.startswith(name) and not k[len(name):len(name) + 1].isdigit()]
        if len(cands) == 1:
            return per_name[cands[0]], 'prefix'
        return None, 'missing'

    calls_per_name = {}
    for c in wrapper:
        if c['kind'] == 'triton':
            calls_per_name[c['name']] = calls_per_name.get(c['name'], 0) + 1
    name_collisions = sum(
        1 for n in calls_per_name
        if sum(1 for g in graphs if n in g['kernels']) > 1)
    if name_collisions:
        status_flags.append(
            f'kernel-name-collisions-across-graphs:{name_collisions}')

    # kern_sum aggregate cross-check (f23)
    ksum = read_kern_sum(args.kern_sum_csv)
    relevant_names = [nm for nm in per_name if classify(nm) != 'COPY']
    missing_sum = [nm for nm in relevant_names if nm not in ksum]
    if not ksum or missing_sum:
        status_flags.append(f'kern-sum-coverage-missing:{len(missing_sum)}')
    ksum_mismatch = 0
    for nm, d in per_name.items():
        tot = ksum.get(nm)
        if tot and d['total_ns'] and abs(tot - d['total_ns']) > max(1, tot * 0.001):
            ksum_mismatch += 1
    if ksum_mismatch:
        status_flags.append(f'kern-sum-crosscheck-mismatch:{ksum_mismatch} names')

    # memcpy accounting: in-window vs boundary, averaged over ALL passes
    memcpy_in_ns, memcpy_out_ns = 0.0, 0.0
    for p in passes:
        sd = [r for r in p if classify(r['name']) == 'SDPA']
        w0, w1 = (sd[0]['start'], sd[-1]['start'] + sd[-1]['dur']) if sd else (0, -1)
        for r in p:
            if classify(r['name']) == 'COPY':
                if w0 <= r['start'] <= w1:
                    memcpy_in_ns += r['dur']
                else:
                    memcpy_out_ns += r['dur']
    memcpy_in_ns /= max(1, n_pass)
    memcpy_out_ns /= max(1, n_pass)

    # COPY-class launches adjacent to SDPA (silent-failure signal, f26)
    adjacent_copies = []
    sd_idx = [i for i, r in enumerate(rep) if classify(r['name']) == 'SDPA']
    for si in sd_idx:
        for j in range(max(0, si - 2), min(len(rep), si + 3)):
            if j != si and classify(rep[j]['name']) == 'COPY':
                adjacent_copies.append(rep[j]['name'])

    bwref = None
    bw_details = {}
    if args.bwref and os.path.exists(args.bwref):
        bw_record = read_json(args.bwref)
        bwref = bw_record.get('BW_ref_GBps', {})
        bw_details = bw_record.get('classes', {})
    b_excluded = {'bwref_class_missing': 0, 'bytes_unresolved': 0,
                  'args_unaligned': 0, 'duration_missing': 0,
                  'access_unresolved': 0, 'bwref_unresolved': 0}

    ledger = []
    cov = {'triton_sites': 0, 'ops_from_comment': 0, 'ops_from_name': 0,
           'ops_unknown': 0, 'trace_exact': 0, 'trace_prefix': 0,
           'trace_missing': 0, 'arg_align_fail': 0, 'dtype_unresolved': 0,
           'load_expr_unparsed_large_reads': 0,
           'access_unresolved_in_stack': 0,
           'multiplicity_mismatch': 0,
           'name_collisions_across_graphs': name_collisions,
           'inner_name_mismatch': 0}
    for i, c in enumerate(wrapper):
        if c['kind'] != 'triton':
            continue
        gr = c['gr']
        k = gr['kernels'].get(c['name'], {})
        ops, ops_src = k.get('ops', []), k.get('ops_source', 'unknown')
        raw = k.get('ops_raw_name', '')
        cov['triton_sites'] += 1
        cov['ops_from_comment' if ops_src == 'aten_comment' else
            'ops_from_name' if ops_src == 'name' else 'ops_unknown'] += 1
        if k.get('inner_mismatch'):
            cov['inner_name_mismatch'] += 1
        aligned = c.get('align_ok', True)
        if not aligned:
            cov['arg_align_fail'] += 1
        stack = in_stack(i)
        # refined per-arg access semantics from the kernel source (f32)
        reads, writes = [], []
        access_unresolved = False
        for a in c['args']:
            role = a.get('role', '')
            r, w, unres = refined_access(k, role)
            m = arg_meta(gr, a)
            if unres:
                access_unresolved = True
                r = w = True   # conservative traffic; the site is flagged
            if r:
                reads.append((role, m))
            if w:
                writes.append(m)
        dtype_ok = all(m['dtype'] is not None
                       for m in [x for _, x in reads] + writes)
        if not dtype_ok:
            cov['dtype_unresolved'] += 1
        b_read = sum(nbytes(m['shape'], m['dtype']) for _, m in reads)
        b_write = sum(nbytes(m['shape'], m['dtype']) for m in writes)
        pure = (bool(ops) and all(o.strip() in PURE_COPY_OPS for o in ops)) or \
            (ops_src == 'name' and
             re.fullmatch(r'(clone|copy|_to_copy|contiguous)', raw or '') is not None)
        # permuted-read: tri-state per large reading arg (f20/f32); a state of
        # 'unresolved' is a (B) CANDIDATE (the zero-clamp makes over-detection
        # safe) and, in-stack, raises a coverage flag
        perm_view = any(is_permuted(m['shape'], m['stride']) for _, m in reads)
        perm_source = False
        perm_unresolved = False
        for role, m in reads:
            if nbytes(m['shape'], m['dtype']) < B_SITE_MIN_READ_BYTES:
                continue
            state = read_permuted_state(k, role, m)
            if state == 'yes':
                perm_source = True
            elif state == 'unresolved':
                perm_unresolved = True
                if stack:
                    cov['load_expr_unparsed_large_reads'] += 1
        if access_unresolved and stack:
            cov['access_unresolved_in_stack'] += 1
        rperm = perm_view or perm_source or perm_unresolved
        tr, mk = match_trace(c['name'])
        cov['trace_' + mk] += 1
        mult = calls_per_name[c['name']]
        share = 1.0 / mult
        dur_ns = (tr['total_ns'] / max(1, n_pass) * share) if tr else 0.0
        mult_ok = bool(tr) and (tr['launches'] == mult * n_pass)
        if tr and not mult_ok:
            cov['multiplicity_mismatch'] += 1
        a_ns = b_ns = 0.0
        bw_class = bw_val = None
        excl = ''
        if stack and pure and (perm_view or perm_source) and not perm_unresolved:
            if tr:
                a_ns = dur_ns
            else:
                b_excluded['duration_missing'] += 1
                excl = 'A-site-missing-duration'
        elif stack and rperm and not pure and bwref is not None:
            bw_class = chain_class(ops, raw)
            bw_val = bwref.get(bw_class)
            if not aligned:
                b_excluded['args_unaligned'] += 1
                excl = 'B-excluded:args-unaligned'
            elif access_unresolved or perm_unresolved:
                b_excluded['access_unresolved'] += 1
                excl = 'B-excluded:access-unresolved'
            elif not dtype_ok:
                b_excluded['bytes_unresolved'] += 1
                excl = 'B-excluded:dtype-unresolved'
            elif not bw_val:
                b_excluded['bwref_class_missing'] += 1
                excl = f'B-excluded:no-BW_ref[{bw_class}]'
            elif not bw_details.get(bw_class, {}).get('reference_accepted', False):
                b_excluded['bwref_unresolved'] += 1
                excl = 'B-excluded:unverified-reference'
            elif (any(m['dtype'] != bw_details[bw_class]['input_dtype']
                      for _, m in reads if nbytes(m['shape'], m['dtype']) >= B_SITE_MIN_READ_BYTES)
                  or any(m['dtype'] != bw_details[bw_class]['out_dtype_contig_in']
                         for m in writes if nbytes(m['shape'], m['dtype']) >= B_SITE_MIN_READ_BYTES)):
                b_excluded['bwref_unresolved'] += 1
                excl = 'B-excluded:reference-boundary-dtype-mismatch'
            elif not tr:
                b_excluded['duration_missing'] += 1
                excl = 'B-excluded:missing-duration'
            else:
                b_ns = max(0.0, dur_ns - (b_read + b_write) / bw_val)
        ledger.append({
            'wrapper_index': i, 'kind': 'triton', 'kernel_name': c['name'],
            'source_file': gr['path'], 'source_line': c.get('source_line'),
            'nvtx_range': 'UNRESOLVED (ordinal SDPA segment)',
            'bytes_moved': b_read + b_write, 'duration_us': dur_ns / 1e3,
            'segment': seg_of_call.get(i, 'pre_stack'),
            'in_stack': stack, 'ops_source': ops_src,
            'fused_ops': ';'.join(ops) if ops else (raw or 'UNKNOWN'),
            'is_pure_copy': pure, 'reads_permuted': rperm,
            'permuted_signal': ('both' if perm_view and perm_source else
                                'view' if perm_view else
                                'source' if perm_source else ''),
            'bytes_read': b_read, 'bytes_written': b_write,
            'dtype_resolved': dtype_ok, 'args_aligned': aligned,
            'duration_ns_per_pass': dur_ns, 'trace_match': mk,
            'launches_expected_per_pass': mult,
            'multiplicity_ok': mult_ok,
            'achieved_GBps': ((b_read + b_write) / dur_ns) if dur_ns else 0.0,
            'bw_ref_class': bw_class or '', 'bw_ref_GBps': bw_val or '',
            'contrib_A_ns': a_ns, 'contrib_B_ns': b_ns, 'exclusion': excl,
        })
        if stack and mk == 'missing' and (pure or rperm):
            status_flags.append(f'unmatched-relevant-kernel:{c["name"]}')
    ledger.append({
        'wrapper_index': -1, 'kind': 'memcpy_window', 'kernel_name':
        'COPY-class rows between first/last SDPA per pass (mean over passes)',
        'segment': 'stack_window', 'in_stack': True, 'ops_source': 'trace',
        'fused_ops': 'memcpy/eager-copy', 'is_pure_copy': True,
        'reads_permuted': '', 'permuted_signal': '', 'bytes_read': '',
        'bytes_written': '', 'dtype_resolved': '', 'args_aligned': '',
        'duration_ns_per_pass': memcpy_in_ns, 'trace_match': 'trace',
        'launches_expected_per_pass': '', 'multiplicity_ok': '',
        'achieved_GBps': '', 'bw_ref_class': '', 'bw_ref_GBps': '',
        'contrib_A_ns': memcpy_in_ns, 'contrib_B_ns': 0.0, 'exclusion': ''})
    ledger.append({
        'wrapper_index': -2, 'kind': 'memcpy_boundary', 'kernel_name':
        'COPY-class rows outside the SDPA window per pass (mean) — NOT in A',
        'segment': 'outside_stack', 'in_stack': False, 'ops_source': 'trace',
        'fused_ops': 'memcpy/eager-copy', 'is_pure_copy': True,
        'reads_permuted': '', 'permuted_signal': '', 'bytes_read': '',
        'bytes_written': '', 'dtype_resolved': '', 'args_aligned': '',
        'duration_ns_per_pass': memcpy_out_ns, 'trace_match': 'trace',
        'launches_expected_per_pass': '', 'multiplicity_ok': '',
        'achieved_GBps': '', 'bw_ref_class': '', 'bw_ref_GBps': '',
        'contrib_A_ns': 0.0, 'contrib_B_ns': 0.0, 'exclusion': ''})

    cols = ['wrapper_index', 'kind', 'kernel_name', 'source_file', 'source_line',
            'nvtx_range', 'bytes_moved', 'duration_us', 'segment', 'in_stack',
            'ops_source', 'fused_ops', 'is_pure_copy', 'reads_permuted',
            'permuted_signal', 'bytes_read', 'bytes_written',
            'dtype_resolved', 'args_aligned', 'duration_ns_per_pass',
            'trace_match', 'launches_expected_per_pass', 'multiplicity_ok',
            'achieved_GBps', 'bw_ref_class', 'bw_ref_GBps',
            'contrib_A_ns', 'contrib_B_ns', 'exclusion']
    out_csv = os.path.join(args.out_dir, f'inductor_kernel_map_{args.tag}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ledger:
            w.writerow(r)
    print(f'-> {out_csv}')

    with open(os.path.join(args.out_dir,
                           f'kernel_order_per_pass_{args.tag}.csv'),
              'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pos', 'class', 'duration_ns', 'kernel_name'])
        for i, r in enumerate(rep):
            w.writerow([i, classify(r['name']), r['dur'], r['name']])

    # ---- totals summed FROM the ledger
    A_ns = sum(r['contrib_A_ns'] for r in ledger)
    B_ns = sum(r['contrib_B_ns'] for r in ledger)
    per_layer = {}
    for r in ledger:
        if r['contrib_A_ns'] or r['contrib_B_ns']:
            lay = str(r['segment']).split('_')[0]
            d = per_layer.setdefault(lay, {'A_ns': 0.0, 'B_ns': 0.0})
            d['A_ns'] += r['contrib_A_ns']
            d['B_ns'] += r['contrib_B_ns']

    # dedup-assumption check (f22): interior layers should carry comparable
    # (A)+(B); a near-empty L0 next to loaded L1..L4 suggests the L0 prologue
    # was NOT captured by name reuse -> undercount flag.
    lay_tot = {k: v['A_ns'] + v['B_ns'] for k, v in per_layer.items()
               if re.fullmatch(r'L\d+', k)}
    if len(lay_tot) >= 3:
        l0 = lay_tot.get('L0', 0.0)
        others = [v for k, v in lay_tot.items() if k != 'L0']
        med = sorted(others)[len(others) // 2]
        if med > 0 and l0 < 0.25 * med:
            status_flags.append(
                f'L0-contribution-anomalously-low ({l0/1e3:.1f}us vs median '
                f'{med/1e3:.1f}us): layer-0 prologue may not share kernel '
                'names — A/B undercount (lower bound)')

    if any(v for v in b_excluded.values()):
        status_flags.append(f'B-partial: excluded sites {b_excluded}')
    if cov['ops_unknown'] or cov['trace_missing'] or cov['arg_align_fail'] \
            or cov['load_expr_unparsed_large_reads'] \
            or cov['access_unresolved_in_stack'] \
            or cov['multiplicity_mismatch'] \
            or any(s.startswith('unmatched-relevant') for s in status_flags):
        status_flags.append('coverage-incomplete: R_layout is a lower bound')

    baseline = read_json(args.baseline_timing)
    if baseline.get('settle', {}).get('settle_capped'):
        raise SystemExit('FATAL: denominator timing has settle_capped=True')
    denom_ms = baseline['min_of_medians_ms']
    traced_med = meta.get('latency_median_ms')
    b_ok = (bwref is not None and not any(b_excluded.values())
            and not cov['load_expr_unparsed_large_reads']
            and not cov['access_unresolved_in_stack'])
    # Uncertain B can overestimate as well as underestimate. Only the
    # observed A contribution is eligible for the allowed lower bound.
    accepted_B_ns = B_ns if b_ok and not status_flags else 0.0
    r_layout = {
        'status': 'ok' if not status_flags else 'FLAGGED',
        'status_flags': status_flags,
        'A_ms_per_pass': A_ns / 1e6,
        'A_includes_memcpy_window_ms': memcpy_in_ns / 1e6,
        'boundary_copy_ms_not_in_A': memcpy_out_ns / 1e6,
        'B_ms_per_pass': B_ns / 1e6 if bwref is not None else None,
        'B_status': ('ok' if b_ok else
                     f'PARTIAL: excluded {b_excluded}' if bwref is not None else
                     'UNRESOLVED: no BW_ref (microbench missing) — R_layout '
                     'is the (A)-only lower bound'),
        'B_excluded_sites': b_excluded,
        'denominator_clean_min_of_medians_ms': denom_ms,
        'denominator_traced_median_ms': traced_med,
        'R_layout_basis': 'A+B' if b_ok and not status_flags else 'A-only lower bound',
        'R_layout_vs_clean': ((A_ns + accepted_B_ns) / 1e6) / denom_ms,
        'R_layout_vs_traced': (((A_ns + accepted_B_ns) / 1e6)
                               / traced_med) if traced_med else None,
        'per_layer': {k: {'A_ms': v['A_ns'] / 1e6, 'B_ms': v['B_ns'] / 1e6}
                      for k, v in sorted(per_layer.items())},
        'per_layer_note': 'segment = SDPA-anchored ordinal window; the '
            'per-layer split is heuristic at window boundaries; totals are '
            'summed from the committed CSV ledger',
        'sdpa_calls_per_pass_wrapper': len(sdpa_pos),
        'sdpa_axis_order': axes,
        'sdpa_adjacent_copy_events_rep_pass': len(adjacent_copies),
        'sdpa_adjacent_copy_names': sorted(set(adjacent_copies)),
        'passes_in_trace': n_pass, 'pass_split_method': split_method,
        'sdpa_anchor_mismatch_passes': len(bad),
        'coverage': cov,
        'heads_split_are_views': heads_split,
        'heads_merge_are_views': heads_merge,
        'heads_are_views': heads_split == 'true',
        'tag': args.tag,
    }
    ct_path = os.path.join(args.out_dir, f'r_layout_{args.tag}.json')
    write_json(ct_path, r_layout)
    print(f'-> {ct_path}')
    print(f"[map] A={r_layout['A_ms_per_pass']:.3f} ms  "
          f"B={r_layout['B_ms_per_pass']}  "
          f"R_layout(clean)={r_layout['R_layout_vs_clean']:.4f}  "
          f"status={r_layout['status']}")
    if status_flags:
        for s in status_flags:
            print(f'  FLAG: {s}')

    # ---- 1d materialization inventory (markdown, committed)
    inv = [f'# Materialization inventory — {args.tag}', '',
           f'head SPLIT reshapes into SDPA are views: **{heads_split}**',
           f'head MERGE after SDPA is a view: **{heads_merge}**',
           '(tri-state producer-ancestry verdicts; unknown never counts as a '
           'view — gate-03 f26)', '']
    for h in head_verdicts:
        inv.append(f"- SDPA @wrapper[{h['wrapper_index']}] ({h['segment']}, "
                   f"{h['axis']}): q/k/v = {h['inputs']} "
                   f"[split={h['split_verdict']}]; merge -> "
                   f"{h['merge_consumer']} [{h['merge_verdict']}]\n"
                   f"  `{h['line'][:160]}`\n  graph: `{h['graph']}`")
    inv.append('')
    inv.append('## Pure-copy materializations per segment (in-stack)')
    seg_tab = {}
    for r in ledger:
        if r['kind'] == 'triton' and r['is_pure_copy'] and r['in_stack']:
            seg_tab.setdefault(r['segment'], []).append(r)
    for seg in sorted(seg_tab):
        inv.append(f'\n### {seg}')
        for s in seg_tab[seg]:
            inv.append(f"- `{s['kernel_name']}` read={s['bytes_read']:,}B "
                       f"write={s['bytes_written']:,}B "
                       f"dur/pass={s['duration_ns_per_pass']/1e3:.1f}us "
                       f"ops={s['fused_ops']}")
    inv.append('')
    inv.append(f'memcpy window (in A): {memcpy_in_ns/1e3:.1f} us/pass; '
               f'boundary copies (not in A): {memcpy_out_ns/1e3:.1f} us/pass; '
               f'SDPA-adjacent COPY events (rep pass): {len(adjacent_copies)}')
    with open(os.path.join(args.out_dir,
                           f'materialization_inventory_{args.tag}.md'), 'w') as f:
        f.write('\n'.join(inv) + '\n')
    print(f"-> materialization_inventory_{args.tag}.md "
          f"({sum(len(v) for v in seg_tab.values())} pure-copy sites)")


if __name__ == '__main__':
    main()
