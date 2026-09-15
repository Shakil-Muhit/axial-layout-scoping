"""Temporal buffer metadata and source-proven C3 layout-copy attribution.

Buffer reuse must be resolved at each call, not from the final alias table.
Copy identity is checked from stored values, including layout copies whose
Inductor names and ATen comments say mm. B remains unpriced without a
matching fused-operation reference; this script never turns missing B into 0.
"""
import argparse
import ast
import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scoping'))
from classify import classify
from kernel_map import (parse_output_code, read_gpu_trace, nvtx_pass_windows,
                        split_passes, is_permuted, DTYPE_BYTES, access_expressions)


def literal(node):
    return ast.literal_eval(node)


def sources(tree):
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'triton'):
            text = node.value.args[1].value
            inner = ast.parse(text)
            fn = next(x for x in inner.body if isinstance(x, ast.FunctionDef))
            signature = None
            for decorator in fn.decorator_list:
                if isinstance(decorator, ast.Call):
                    for kw in decorator.keywords:
                        if kw.arg == 'triton_meta':
                            signature_node = next(v for k,v in zip(kw.value.keys,kw.value.values)
                                                  if literal(k) == 'signature')
                            signature = literal(signature_node)
            out[node.targets[0].id] = {'source': text, 'function': fn, 'signature': signature}
    return out


def copy_value(kernel, reads, writes):
    """All stored values must be unchanged loads, with exact widening casts.
    Index arithmetic is irrelevant to value identity and is retained as evidence.
    """
    assignments = {}
    for node in ast.walk(kernel['function']):
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            assignments.setdefault(node.targets[0].id, []).append(node.value)
    def origin(node, seen=()):
        if isinstance(node,ast.Name) and node.id not in seen:
            vals=assignments.get(node.id,[])
            return origin(vals[0],(*seen,node.id)) if len(vals)==1 else None
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='tl.load':
            names={x.id for x in ast.walk(node.args[0]) if isinstance(x,ast.Name)}
            hits=names & set(reads)
            return next(iter(hits)) if len(hits)==1 else None
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='to':
            # Only the lossless fp16->fp32 promotion emitted by these copies.
            if len(node.args)==1 and ast.unparse(node.args[0])=='tl.float32':
                return origin(node.func.value,seen)
        return None
    stores=[n for n in ast.walk(kernel['function']) if isinstance(n,ast.Call)
            and ast.unparse(n.func)=='tl.store']
    if not stores:return False
    for store in stores:
        names={n.id for n in ast.walk(store.args[0]) if isinstance(n,ast.Name)}
        dest=names & set(writes)
        src=origin(store.args[1])
        if len(dest)!=1 or src is None:return False
        target=next(iter(dest))
        if reads[src]['dtype'] != writes[target]['dtype']:return False
        if reads[src]['dtype'] not in ('torch.float16','torch.float32'):return False
    return True


def parse_temporal(path):
    text=path.read_text(); tree=ast.parse(text); kernels=sources(tree)
    stock=parse_output_code(str(path.parent))[0]
    call_by_line={c['source_line']:c for c in stock['calls']}
    env={}; records=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                and ast.unparse(node.value.func)=='rand_strided'):
            kws={k.arg:k.value for k in node.value.keywords}
            name=node.targets[0].id
            env[name]={'shape':literal(node.value.args[0]),'stride':literal(node.value.args[1]),
                       'dtype':ast.unparse(kws['dtype']),'storage':'parameter:'+name,'offset':0}
    def meta(node):
        if isinstance(node,ast.Name):return copy.deepcopy(env.get(node.id))
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='reinterpret_tensor':
            result=meta(node.args[0]); assert result is not None, ast.unparse(node)
            result.update(shape=literal(node.args[1]),stride=literal(node.args[2]),
                          offset=literal(node.args[3]) if len(node.args)>3 else 0)
            return result
        if isinstance(node,ast.Subscript):
            base=meta(node.value)
            return copy.deepcopy(base['tuple'][literal(node.slice)]) if base and 'tuple' in base else None
        return None
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='call')
    statements=sorted((n for n in ast.walk(fn) if isinstance(n,(ast.Assign,ast.Expr,ast.Delete))),
                      key=lambda n:(n.lineno,n.col_offset))
    for node in statements:
        value=node.value if isinstance(node,(ast.Assign,ast.Expr)) else None
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if isinstance(value,ast.Call) and ast.unparse(value.func).startswith('empty_strided_'):
                env[name]={'shape':literal(value.args[0]),'stride':literal(value.args[1]),
                           'dtype':ast.unparse(value.args[2]),'storage':f'alloc:{node.lineno}','offset':0}
            elif isinstance(value,ast.Call) and '_scaled_dot_product' in ast.unparse(value.func):
                q=meta(value.args[0]); assert q
                output={**q,'storage':f'sdpa:{node.lineno}','offset':0}
                env[name]={'tuple':[output]}
            else:
                result=meta(value)
                if result is not None:env[name]=result
                elif isinstance(value,ast.Call) and ast.unparse(value.func).startswith('torch.ops.'):
                    kws={k.arg:k.value for k in value.keywords}
                    env[name]={'shape':None,'stride':None,
                               'dtype':ast.unparse(kws['dtype']) if 'dtype' in kws else None,
                               'storage':f'fallback:{node.lineno}','offset':0}
        if isinstance(value,ast.Call) and ast.unparse(value.func)=='assert_size_stride':
            name=value.args[0].id
            env.setdefault(name,{'dtype':None,'storage':f'assert:{node.lineno}','offset':0})
            env[name].update(shape=literal(value.args[1]),stride=literal(value.args[2]))
        call=call_by_line.get(node.lineno)
        if call:
            rec={**call,'index':len(records),'metas':[]}
            for a in call['args']:
                m=copy.deepcopy(env.get(a['buf']))
                if m and a.get('reinterpret'):
                    m.update(shape=a['shape'],stride=a['stride'])
                rec['metas'].append(m)
            records.append(rec)
        if isinstance(node,ast.Delete):
            for target in node.targets:
                if isinstance(target,ast.Name):env.pop(target.id,None)
    assert len(records)==len(stock['calls'])
    return stock,kernels,records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root
    run=json.loads((root/'manifest.json').read_text())
    path=root/'phase1/active_dump/output_code.py'
    graph,kernels,calls=parse_temporal(path)
    trace=read_gpu_trace(str(root/'phase1/trace_c3_cuda_gpu_trace.csv'))
    windows,error=nvtx_pass_windows(str(root/'phase1/trace_c3_nvtx_gpu_proj_trace.csv'))
    assert not error and len(windows)==50
    passes,_=split_passes(trace,windows,classify)
    triton_calls=[c for c in calls if c['kind']=='triton']
    events=[[r for r in p if classify(r['name'])=='INDUCTOR_FUSED'] for p in passes]
    expected=[c['name'] for c in triton_calls]
    assert all([r['name'] for r in p]==expected for p in events), 'kernel order mismatch'
    anchors=[c for c in calls if c['kind']=='fallback' and 'scaled_dot_product' in c['name']]
    assert len(anchors)==12
    positions=[c['index'] for c in anchors]
    # Source landmarks: first axial RMSNorm and final axial norm reduction.
    start=next(c['index'] for c in calls if c['name'].endswith('norm_mul_81'))
    end=next(c['index'] for c in calls if c['name']=='triton_red_fused_add_linalg_vector_norm_98')
    rows=[]; details=[]; flags=[]; head=[]; reference_checks=[]
    reference=json.loads((root/'phase1/microbench_bwref.json').read_text())
    for tpos,c in enumerate(triton_calls):
        k=kernels[c['name']]; reads={};writes={};io=[]
        stack=start<=c['index']<=end
        for a,m in zip(c['args'],c['metas']):
            role=a['role']
            if m is None:m={'shape':None,'stride':None,'dtype':None,'storage':'UNRESOLVED','offset':None}
            if not m.get('dtype'):
                types={'*fp16':'torch.float16','*fp32':'torch.float32','*fp64':'torch.float64',
                       '*i64':'torch.int64','*i32':'torch.int32','*i8':'torch.int8','*u8':'torch.uint8'}
                m['dtype']=types.get(k['signature'].get(role))
                m['dtype_source']='generated kernel pointer signature'
            size=(math.prod(m['shape'])*DTYPE_BYTES[m['dtype']]
                  if m.get('shape') is not None and m.get('dtype') else None)
            if stack:assert size is not None,(c['name'],a,m)
            entry={**m,'bytes':size,'role':role}
            io.append(entry)
            if role.startswith(('in_ptr','in_out_ptr')):reads[role]=entry
            if role.startswith(('out_ptr','in_out_ptr')):writes[role]=entry
        accesses=access_expressions(k['source'],[a['role'] for a in c['args']])
        pure=copy_value(k,reads,writes)
        large=[m for m in reads.values() if m['bytes'] is not None and m['bytes']>=1_000_000]
        output_exprs=set(accesses['store_exprs'].values())
        indexed=any(accesses['load_exprs'].get(m['role']) not in output_exprs for m in large)
        permuted=any(is_permuted(m['shape'],m['stride']) for m in large) or indexed
        duration=sum(p[tpos]['dur'] for p in events)/len(events)
        a_ns=duration if stack and pure and permuted else 0.
        previous=[i for i,p in enumerate(positions) if p<c['index']]
        attention=previous[-1] if previous else 0
        segment=f'L{attention//2}_{"time" if attention%2==0 else "freq"}'
        byte_count=(sum(m['bytes'] for m in reads.values())+sum(m['bytes'] for m in writes.values())
                    if all(m['bytes'] is not None for m in [*reads.values(),*writes.values()]) else None)
        b_candidate=stack and permuted and not pure and bool(large)
        if b_candidate:
            ops=graph['kernels'][c['name']].get('ops',[])
            ref_class='norm' if 'aten.linalg_vector_norm' in ops else 'pointwise'
            ref=reference['classes'][ref_class]
            input_types=sorted({m['dtype'] for m in large})
            output_types=sorted({m['dtype'] for m in writes.values() if m['bytes']>=1_000_000})
            dtype_ok=(input_types==[ref['input_dtype']] and output_types==[ref['out_dtype_contig_in']])
            reason=('input/output dtype mismatch' if not dtype_ok else
                    'operation chain differs: reference uses gelu((x+bias)*1.5)')
            reference_checks.append({'wrapper_index':c['index'],'kernel_name':c['name'],
                'reference_class':ref_class,'actual_input_dtypes':';'.join(input_types),
                'actual_output_dtypes':';'.join(output_types),'reference_input_dtype':ref['input_dtype'],
                'reference_output_dtype':ref['out_dtype_contig_in'],'dtype_match':dtype_ok,
                'accepted':False,'reason':reason,'fused_ops':';'.join(ops)})
        row={'wrapper_index':c['index'],'source_line':c['source_line'],'kernel_name':c['name'],
             'nvtx_range':'pass_*; source/ordered-event mapping','segment':segment,'in_stack':stack,
             'fused_ops':';'.join(graph['kernels'][c['name']].get('ops',[])),
             'is_pure_copy':pure,'reads_permuted':permuted,'bytes_moved':byte_count,
             'duration_us':duration/1000,'achieved_GBps':byte_count/duration if duration and byte_count is not None else None,
             'A_ns':a_ns,'B_status':'UNRESOLVED: matching fusion reference required' if b_candidate else 'not priced',
             'B_candidate':b_candidate}
        rows.append(row)
        details.append({'wrapper_index':c['index'],'kernel':c['name'],'source_line':c['source_line'],
                        'arguments':io,'access_expressions':accesses,'source_proves_unchanged_values':pure})
        if a_ns:
            head.append({'layer':attention//2,'axis':'time' if attention%2==0 else 'freq',
                         'kernel':c['name'],'source_line':c['source_line'],
                         'bytes_read':sum(m['bytes'] for m in reads.values()),
                         'bytes_written':sum(m['bytes'] for m in writes.values()),
                         'duration_us':duration/1000,'next_consumer':'out-projection extern.mm'})
    assert len(head)==12 and all('fused_mm_' in x['kernel'] for x in head)
    out=root/'phase1'
    with open(out/'inductor_kernel_map_c3_audited.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with open(out/'bandwidth_reference_coverage.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(reference_checks[0]));writer.writeheader();writer.writerows(reference_checks)
    (out/'call_metadata_c3.json').write_text(json.dumps({'generation_id':run['generation_id'],
        'use_amp':True,'calls':details},indent=1)+'\n')
    A=sum(r['A_ns'] for r in rows)/1e6
    timing=json.loads((root/'qualification/timing.json').read_text())
    result={'status':'A_ONLY_LOWER_BOUND','generation_id':run['generation_id'],'use_amp':True,
            'A_ms_per_pass':A,'B_ms_per_pass':None,'R_layout':None,
            'A_share_of_c3':A/timing['min_of_medians_ms'],'denominator_ms':timing['min_of_medians_ms'],
            'R_layout_basis':'A-only; cannot establish NO-GO below 0.03',
            'B_reason':'Existing class references do not reproduce mixed-dtype, multi-output axial fusions; final axial norm also fuses into mask-head consumers.',
            'all_triton_calls_mapped_in_order':len(triton_calls),'passes':len(passes),
            'head_merge_copies':head,'head_split':'q/k are RoPE computation outputs; v is a QKV view; no separate pure split copy',
            'gating_layout':'SDPA outputs are read in BNHD order and gating writes BHND; the subsequent pure copy produces BNHD for the projection',
            'per_layer_A_ms':{f'L{i}':sum(x['duration_us'] for x in head if x['layer']==i)/1000 for i in range(6)},
            'B_candidate_sites':sum(r['B_candidate'] for r in rows),
            'B_reference_rejection_counts':dict(collections.Counter(r['reason'] for r in reference_checks)),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'layout_attribution_audited.json').write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='head_merge_copies'},indent=1))


if __name__=='__main__':main()
