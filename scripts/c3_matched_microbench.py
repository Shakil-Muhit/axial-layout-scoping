"""Bounded Phase-1 calibration for actual C3 norm/rotary/gating chains.

No model intervention: standalone PyTorch op-chains compiled by Inductor.
These measurements are diagnostic until source, fusion, dtype, shape,
traffic, and boundary coverage are reviewed against the executed model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scoping'))
import torch
import torch.nn.functional as F
from microbench_p_l2 import bench_device_ms
from evidence_io import write_json


def tensor(shape,dtype=torch.float16):
    gen=torch.Generator(device='cpu').manual_seed(4242)
    return torch.randn(shape,generator=gen,dtype=dtype).cuda()


def build(case):
    family,axis,kind=case.split(':')
    batch,length=(60,801) if axis=='time' else (801,60)
    if family=='norm':
        dtype=torch.float16 if kind.endswith('half') else torch.float32
        base=tensor((batch,length,384),dtype)
        perm=base.permute(1,0,2).contiguous().permute(1,0,2)
        projection=tensor(base.shape); gamma=tensor((384,),torch.float32)
        if kind.startswith('single'):
            def chain(residual):
                return (F.normalize(projection+residual,dim=-1)*(384**0.5)*gamma).half().contiguous()
            parameter_bytes=projection.numel()*2+gamma.numel()*4
        else:
            ff=tensor(base.shape);bias=tensor((384,),torch.float32);gamma2=tensor((384,),torch.float32)
            def chain(residual):
                value=(ff.float()+bias)+(projection+residual)
                first=F.normalize(value,dim=-1)*(384**0.5)*gamma
                second=(F.normalize(first,dim=-1)*(384**0.5)*gamma2).half()
                return first.contiguous(),second.contiguous(),second.clone(memory_format=torch.contiguous_format)
            parameter_bytes=projection.numel()*2+ff.numel()*2+3*384*4
    elif family=='rotary':
        base=tensor((3,batch,length,8,64))
        perm=base.permute(1,2,0,3,4).contiguous().permute(2,0,1,3,4)
        frequencies=tensor((length,64),torch.float32).view(1,length,1,64)
        def chain(packed):
            q,k=packed[0],packed[1]
            def rotate(x):
                pairs=x.unflatten(-1,(-1,2))
                half=torch.stack((-pairs[...,1],pairs[...,0]),dim=-1).flatten(-2)
                return (x*frequencies.cos()+half*frequencies.sin()).half().contiguous()
            return rotate(q),rotate(k)
        parameter_bytes=frequencies.numel()*4
    elif family=='gate':
        base=tensor((batch,8,length,64))
        perm=base.permute(0,2,1,3).contiguous().permute(0,2,1,3)
        gates=tensor((batch,8,length)); bias=tensor((8,),torch.float32)
        perm_gates=gates.permute(0,2,1).contiguous().permute(0,2,1)
        def chain(x):
            gate=gates if x.is_contiguous() else perm_gates
            return (x*(gate+bias.view(1,8,1)).sigmoid().unsqueeze(-1)).half().contiguous()
        parameter_bytes=gates.numel()*2+bias.numel()*4
    else:raise ValueError(case)
    assert torch.equal(base,perm)
    return chain,base,perm,parameter_bytes


def outputs(value):return list(value) if isinstance(value,(tuple,list)) else [value]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    fn,base,perm,extra=build(args.case)
    compiled=torch.compile(fn,dynamic=False,fullgraph=True)
    with torch.no_grad(),torch.autocast('cuda',enabled=True):
        reference=outputs(fn(base));contiguous=outputs(compiled(base));permuted=outputs(compiled(perm))
        checks=[]
        for ref,c,p in zip(reference,contiguous,permuted):
            assert c.shape==p.shape==ref.shape and c.dtype==p.dtype==ref.dtype
            assert torch.isfinite(c).all() and torch.isfinite(p).all()
            assert torch.allclose(c,ref,rtol=1e-2,atol=1e-3)
            # Layout-only comparison retains the original tight tier.
            tight=bool(torch.allclose(c,p,rtol=1e-4,atol=1e-5))
            checks.append({'shape':list(c.shape),'dtype':str(c.dtype),'tight_layout_comparison':tight,
                           'max_abs_diff':float((c-p).abs().max()),
                           'contiguous_stride':list(c.stride()),'permuted_stride':list(p.stride())})
    c_ms,c_method=bench_device_ms(compiled,base)
    p_ms,p_method=bench_device_ms(compiled,perm)
    byte_count=base.numel()*base.element_size()+extra+sum(y.numel()*y.element_size() for y in contiguous)
    result={'case':args.case,'status':'DIAGNOSTIC_PENDING_SOURCE_AND_COVERAGE_REVIEW',
            'input_shape':list(base.shape),'input_dtype':str(base.dtype),
            'contiguous_stride':list(base.stride()),'permuted_stride':list(perm.stride()),
            'output_checks':checks,'contiguous_ms':c_ms,'permuted_ms':p_ms,
            'permuted_over_contiguous':p_ms/c_ms,'bytes_counted':byte_count,
            'contiguous_GBps':byte_count/c_ms/1e6,
            'timing_methods':{'contiguous':c_method,'permuted':p_method},
            'all_tight_layout_checks_pass':all(x['tight_layout_comparison'] for x in checks),
            'use_amp':True,'started_unix':start,'finished_unix':time.time(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.out/'result.json',result)


if __name__=='__main__':main()
