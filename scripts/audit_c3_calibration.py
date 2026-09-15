"""Summarize the bounded reference refinement without accepting missing B."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scoping'))
from kernel_map import parse_output_code


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root
    run=json.loads((root/'manifest.json').read_text());rows=[]
    for folder in sorted((root/'matched_calibration').iterdir()):
        if not folder.is_dir():continue
        d=json.loads((folder/'result.json').read_text())
        assert d['generation_id']==run['generation_id'] and d['use_amp'] is True
        assert int((folder/'exit.txt').read_text())==0
        assert d['timing_methods']=={'contiguous':'cuda-graph-replay','permuted':'cuda-graph-replay'}
        graphs=parse_output_code(str(folder/'dump'))
        assert len(graphs)==2 and all(len(g['calls'])==1 for g in graphs)
        source_checks=[]
        for g in graphs:
            tree=ast.parse(Path(g['path']).read_text())
            calls=[]
            for node in tree.body:
                if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                        and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='triton'):
                    inner=ast.parse(node.value.args[1].value)
                    fn=next(n for n in inner.body if isinstance(n,ast.FunctionDef))
                    calls.extend(ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n,ast.Call))
            source_checks.append({'source':str(Path(g['path']).relative_to(root)),
                                  'sqrt_calls':calls.count('libdevice.sqrt'),
                                  'sigmoid_calls':calls.count('tl.sigmoid'),
                                  'cos_calls':calls.count('tl.cos'),
                                  'sin_calls':calls.count('tl.sin')})
        rows.append({'case':d['case'],'contiguous_ms':d['contiguous_ms'],
                     'permuted_ms':d['permuted_ms'],'ratio':d['permuted_over_contiguous'],
                     'tight_pass':d['all_tight_layout_checks_pass'],
                     'source_checks':source_checks,'result':str((folder/'result.json').relative_to(root))})
    assert len(rows)==10
    first=int((root/'extension_first_started_unix.txt').read_text())
    finished=int((root/'matched_calibration/finished_unix.txt').read_text())
    assert finished-first<=3600
    result={'status':'B_UNRESOLVED','generation_id':run['generation_id'],'use_amp':True,
            'cases':rows,'tight_failed_cases':[r['case'] for r in rows if not r['tight_pass']],
            'minimum_ratio':min(r['ratio'] for r in rows),'maximum_ratio':max(r['ratio'] for r in rows),
            'gpu_window_elapsed_seconds':finished-first,
            'reason':'Three single-norm cases fail the registered tight layout comparison; final axial normalization also fuses into mask-head consumers outside the simple attention window. No full B or NO-GO is established.',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'matched_calibration/audit.json').write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=1))


if __name__=='__main__':main()
