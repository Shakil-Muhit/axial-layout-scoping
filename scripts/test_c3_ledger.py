"""Regression cases for source-proven copies and temporal buffer reuse."""
import ast
from pathlib import Path
import tempfile
import unittest
from c3_layout_ledger import copy_value, parse_temporal


class LedgerTests(unittest.TestCase):
    def test_mm_named_copy_and_arithmetic_are_distinct(self):
        def kernel(expression):
            source = ('def k(in_ptr0, out_ptr0):\n'
                      ' v = tl.load(in_ptr0 + (x % 8 + 32 * (x // 8))).to(tl.float32)\n'
                      f' tl.store(out_ptr0 + x, {expression})\n')
            return {'function': ast.parse(source).body[0]}
        reads={'in_ptr0':{'dtype':'torch.float16'}}
        writes={'out_ptr0':{'dtype':'torch.float16'}}
        self.assertTrue(copy_value(kernel('v'),reads,writes))
        self.assertFalse(copy_value(kernel('v * 2'),reads,writes))
        self.assertFalse(copy_value(kernel('v'),{'in_ptr0':{'dtype':'torch.float32'}},writes))

    def test_buffer_alias_metadata_is_resolved_at_each_call(self):
        inner='def triton_poi_fused_mm_0(in_ptr0,out_ptr0,xnumel):\n v=tl.load(in_ptr0+x)\n tl.store(out_ptr0+x,v)\n'
        text=("# Original ATen: [aten.mm]\n"
              "triton_poi_fused_mm_0 = async_compile.triton('triton_poi_fused_mm_0', "
              + repr(inner) + ", device_str='cuda')\n"
              "def call(args):\n"
              " buf0 = empty_strided_cuda((2, 4), (4, 1), torch.float16)\n"
              " buf1 = reinterpret_tensor(buf0, (4, 2), (1, 4), 0)\n"
              " buf2 = empty_strided_cuda((4, 2), (2, 1), torch.float16)\n"
              " triton_poi_fused_mm_0.run(buf1, buf2, 8, stream=stream0)\n"
              " buf0 = empty_strided_cuda((3, 4), (4, 1), torch.float32)\n"
              " triton_poi_fused_mm_0.run(buf0, buf2, 8, stream=stream0)\n"
              " return (buf2,)\n")
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'output_code.py';path.write_text(text)
            _,_,calls=parse_temporal(path)
        self.assertEqual(calls[0]['metas'][0]['shape'],(4,2))
        self.assertEqual(calls[0]['metas'][0]['stride'],(1,4))
        self.assertEqual(calls[0]['metas'][0]['dtype'],'torch.float16')
        self.assertEqual(calls[1]['metas'][0]['shape'],(3,4))
        self.assertEqual(calls[1]['metas'][0]['dtype'],'torch.float32')
        self.assertNotEqual(calls[0]['metas'][0]['storage'],calls[1]['metas'][0]['storage'])


if __name__=='__main__':unittest.main()
