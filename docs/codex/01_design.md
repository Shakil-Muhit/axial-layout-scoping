# Codex review — 01_design

model: gpt-6-astra · effort: ultra · sandbox: read-only · 2026-09-15T16:51:45+06:00

**The review is blocked by repository access.** Every attempted read fails before the command runs:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

None of the five required files was read. I cannot honestly supply numbered-rule findings, repository `file:line` citations, or judgments of the documented justifications. The verdict below counts **one review-access blocker**, not an established design defect.

Independent upstream checks produced these provisional findings:

1. **The stride arithmetic is correct.** `T*F=48060`, `3*H*dh=1536`, and `F*1536=92160`. With Q/K/V offsets `0/512/1024`, the address is:
   ```text
   (60*t + f)*1536 + h*64 + d + offset
   ```
   This gives temporal strides `(1536,64,92160,1)` and frequency strides `(92160,64,1536,1)`. Both satisfy Flash’s contiguous-last-dimension requirement; its implementation accepts separate batch, sequence, and head strides. This establishes stride legality, subject to dtype/device and other SDPA requirements. [PyTorch 2.8 Flash implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp#L79-L97).

2. **Inductor does not necessarily rematerialize these views.** Its 2.8 SDPA constraint preserves already-realized, aligned inputs; both supplied stride tuples satisfy the alignment predicate. Inspect `output_code` for Flash arguments aliasing the original QKV allocation with the expected strides and offsets. A fresh buffer populated by a Triton kernel before Flash demonstrates materialization—even when that copy is fused with RoPE/casting and has no `clone` name. [SDPA constraint](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/lowering.py#L2300-L2346), [alignment predicate](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/ir.py#L396-L408).

3. **The retained head-merge copy needs scrutiny.** If the supplied temporal Q layout reaches Flash, its output allocation compacts that stride order. The resulting `(F,H,T,dh)` strides are `(512,64,30720,1)`; permuting to `(T,F,H,dh)` gives contiguous `(30720,512,64,1)`. Head merging and flattening can then be views. RoPE or earlier materialization could change this conclusion, so the actual design must be inspected. [Flash output metadata](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_meta_registrations.py#L5357-L5373), [allocation semantics](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/TensorFactories.cpp#L430-L450).

4. **Backend pinning is achievable in Dynamo 2.8.** It explicitly supports `torch.nn.attention.sdpa_kernel`. Module reuse alone does not establish which backend runs; the reused implementation and compiled Flash calls need inspection. [Version-pinned Dynamo handling](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/variables/torch.py#L388).

5. **Replay preserves kernels, but does not prove ordinal attribution.** Nsight’s node tracing can expose individual CUDA graph nodes, so twelve validated anchors can survive replay. That does not establish ownership of intervening fused or reordered kernels. Also, module hooks are generally supported by Dynamo; direct NVTX push/pop inside them are a separate issue. The 2.8 tracing path indicates those integer-returning calls should graph-break, but this was not runtime-tested here. [Nsight graph tracing](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-graph-trace), [module-hook support](https://docs.pytorch.org/docs/2.8/torch.compiler_nn_module.html), [NVTX tracing rules](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/trace_rules.py#L2630).

Please restore shell read access or provide the five files’ contents in the requested order so the binding review can be completed.

DESIGN-BLOCKERS: 1
---
exit: 0
