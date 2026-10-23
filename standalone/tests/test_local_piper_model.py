"""Actual pinned ORT CPU operator/RNG fixtures, without a native runtime."""
import base64
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

from standalone.local_piper_model import PiperModel,fields
from standalone.local_piper_random import NormalGenerator
from standalone.local_piper_tensor import Tensor,f32


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'numpy','onnx','onnxruntime','piper','torch','ctypes','subprocess'}:
            raise AssertionError('Piper model attempted native/process dependency '+fullname)


def _varint(value):
    result=bytearray()
    while value>127:result.append((value&127)|128);value>>=7
    result.append(value);return bytes(result)


def _field(number,wire,value):
    key=_varint(number<<3|wire)
    return key+(_varint(value)if wire==0 else _varint(len(value))+value if wire==2 else value)


class LocalPiperModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_piper_model.json').read_text())
        data=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        if hashlib.sha256(data).hexdigest()!=wrapper['data_sha256']:
            raise AssertionError('Piper native fixture digest mismatch')
        cls.oracle=json.loads(data)

    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)

    def test_original_onnx_137_tensor_and_neural_operator_cases(self):
        cases=self.oracle['operators']['cases'];self.assertEqual(len(cases),137)
        self.assertEqual(len({c['op']for c in cases}),51)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'native-golden.onnx'
            for case in cases:
                with self.subTest(op=case['op'],shapes=[v['shape']for v in case['inputs'].values()]):
                    path.write_bytes(base64.b64decode(case['model'],validate=True))
                    model=PiperModel(path)
                    inputs={name:Tensor.make(value['shape'],value['values'],value['dtype'])for name,value in case['inputs'].items()}
                    result=model.run(inputs)
                    self.assertEqual(len(result),len(case['expected']))
                    for actual,expected in zip(result,case['expected'],strict=True):
                        self.assertEqual(actual.shape,tuple(expected['shape']))
                        self.assertEqual(actual.dtype,expected['dtype'])
                        error=max((abs(a-b)for a,b in zip(actual.values,expected['values'],strict=True)),default=0.)
                        tolerance=1e-5+max(map(abs,expected['values']),default=0.)*1e-6 if actual.dtype==1 else 0
                        self.assertLessEqual(error,tolerance)

    def test_original_ort_normal_108_sequential_invocations_20736_values(self):
        cases=self.oracle['random']['cases'];self.assertEqual(len(cases),108)
        previous=None;total=0
        for case in cases:
            key=case['seed'],case['mean'],case['scale']
            if key!=previous:generator=NormalGenerator(f32(case['seed']));previous=key
            values=generator.generate(case['length'],f32(case['mean']),f32(case['scale']))
            with self.subTest(seed=case['seed'],length=case['length'],scale=case['scale']):
                self.assertEqual(len(values),len(case['native']))
                self.assertLessEqual(max(abs(a-b)for a,b in zip(values,case['native'],strict=True)),2e-6)
            total+=len(values)
        self.assertEqual(total,20736)

    def test_invalid_graph_custom_operator_and_external_storage_fail_explicitly(self):
        case=next(c for c in self.oracle['operators']['cases']if c['op']=='Identity')
        valid=base64.b64decode(case['model'])
        external=_field(1,0,1)+_field(2,0,1)+_field(8,2,b'external-weight')+_field(14,0,1)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'voice.onnx'
            invalid=[b'',b'\x80',b'\x0f',valid[:-1],valid.replace(b'Identity',b'NotAnOp!')]
            invalid.append(b''.join(_field(n,w,v+_field(5,2,external)if n==7 else v)for n,w,v in fields(valid)))
            for data in invalid:
                with self.subTest(data=data[:16]),self.assertRaises(ValueError):
                    path.write_bytes(data);PiperModel(path)

    def test_original_input_names_types_and_declared_shapes(self):
        case=next(c for c in self.oracle['operators']['cases']if c['op']=='Identity')
        valid=base64.b64decode(case['model'])
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'voice.onnx';path.write_bytes(valid);model=PiperModel(path)
            value=Tensor.make((2,3),range(6))
            for inputs in ({},{'x0':value,'unexpected':value},{'x0':Tensor.make((2,3),range(6),7)}):
                with self.subTest(inputs=list(inputs)),self.assertRaises(ValueError):model.run(inputs)
            dimensions=b''.join(_field(1,2,_field(1,0,n))for n in (2,3))
            tensor_type=_field(1,0,1)+_field(2,2,dimensions)
            declared=_field(1,2,b'x0')+_field(2,2,_field(1,2,tensor_type))
            parts=[]
            for n,w,v in fields(valid):
                if n==7:v=b''.join(_field(gn,gw,declared if gn==11 else gv)for gn,gw,gv in fields(v))
                parts.append(_field(n,w,v))
            path.write_bytes(b''.join(parts));model=PiperModel(path)
            with self.assertRaisesRegex(ValueError,'shape'):
                model.run({'x0':Tensor.make((1,6),range(6))})
            self.assertEqual(list(model.run({'x0':value})[0].values),list(range(6)))


if __name__=='__main__':unittest.main()
