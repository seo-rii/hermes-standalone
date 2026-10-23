"""Original Piper/ORT recurrent and Arabic/Hebrew backend contracts."""
from array import array
import base64
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

from standalone.local_piper_arabic import TashkeelDiacritizer,TashkeelError
from standalone.local_piper_hebrew import NakdimonDiacritizer,normalize
from standalone.local_piper_nakdimon import NakdimonModel
from standalone.local_piper_phonemize_hebrew import HebrewPhonemizer
from standalone.local_piper_recurrent import ArabicModel
from standalone.local_piper_tensor import Tensor


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'numpy','onnx','onnxruntime','piper','tensorflow','torch','ctypes','subprocess'}:
            raise AssertionError('Piper backend attempted native/process dependency '+fullname)


class RecordedHebrewModel:
    """Explicit test dependency returning independently captured native logits."""
    def __init__(self,test,events):self.test=test;self.events=list(events)

    def run(self,ids):
        event=self.events.pop(0);self.test.assertEqual(ids,event['inputs']['input_1'][0]);result=[]
        for output in event['outputs']:
            width=output['shape'][-1];values=output['values'];result.append([values[i:i+width]for i in range(0,len(values),width)])
        return result


class RecordedArabicModel:
    """Explicit test dependency; it is never installed in production modules."""
    def __init__(self,test,events):self.test=test;self.events=list(events)

    def run(self,inputs):
        event=self.events.pop(0);self.test.assertEqual(set(inputs),set(event['inputs']))
        for name,tensor in inputs.items():
            source=event['inputs'][name];values=[v for row in source for v in row]if isinstance(source[0],list)else source
            self.test.assertEqual(list(tensor.values),values);self.test.assertEqual(tensor.dtype,7)
        result=[]
        for out in event['outputs']:
            result.append(Tensor(tuple(out['shape']),array('B',out['values']),2)if out['dtype']=='uint8'else Tensor.make(out['shape'],out['values']))
        return result


class LocalPiperBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_piper_backends.json').read_text())
        data=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        if hashlib.sha256(data).hexdigest()!=wrapper['data_sha256']:raise AssertionError('Piper backend fixture digest mismatch')
        cls.oracle=json.loads(data)

    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)

    def test_original_ort_gru_and_attention_auxiliary_47_cases(self):
        cases=self.oracle['recurrent']['cases'];self.assertEqual(len(cases),47)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'original.onnx'
            for case in cases:
                with self.subTest(op=case['op'],inputs=[v['shape']for v in case['inputs'].values()]):
                    path.write_bytes(base64.b64decode(case['model'],validate=True));model=ArabicModel(path)
                    actual=model.run({name:Tensor.make(v['shape'],v['values'],v['dtype'])for name,v in case['inputs'].items()})
                    for result,expected in zip(actual,case['expected'],strict=True):
                        self.assertEqual(result.shape,tuple(expected['shape']));self.assertEqual(result.dtype,expected['dtype'])
                        error=max((abs(a-b)for a,b in zip(result.values,expected['values'],strict=True)),default=0.)
                        self.assertLessEqual(error,1e-5 if result.dtype==1 else 0.)

    def test_original_ort_lstm_24_cases_including_reverse_and_padding(self):
        cases=self.oracle['lstm']['cases'];self.assertEqual(len(cases),24)
        for case in cases:
            args=case['inputs'];steps,batch,width=args['x0']['shape'];directions,gatewidth,_=args['x1']['shape'];hidden=gatewidth//4
            expected=case['expected'][0]['values'];x=args['x0']['values'];w=args['x1']['values'];r=args['x2']['values'];bias=args['x3']['values'];lens=args['x4']['values']
            order=[gate*hidden+i for gate in (0,2,3,1)for i in range(hidden)]
            for d in range(directions):
                parameters=([w[(d*gatewidth+q)*width:(d*gatewidth+q+1)*width]for q in order],[r[(d*gatewidth+q)*hidden:(d*gatewidth+q+1)*hidden]for q in order],[bias[d*hidden*8+q]for q in order])
                reverse=(directions==2 and d==1)or(directions==1 and b'reverse'in base64.b64decode(case['model']))
                for b in range(batch):
                    values=[x[(t*batch+b)*width:(t*batch+b+1)*width]for t in range(steps)]
                    actual=NakdimonModel.recurrent(values,[t<lens[b]for t in range(steps)],parameters,reverse)
                    for t,row in enumerate(actual):
                        target=expected[((t*directions+d)*batch+b)*hidden:((t*directions+d)*batch+b+1)*hidden]
                        self.assertLessEqual(max(abs(a-z)for a,z in zip(row,target,strict=True)),1e-5)

    def test_original_hebrew_text_mapping_merge_and_ipa_34_cases(self):
        for case in self.oracle['hebrew']['cases']:
            with self.subTest(text=case['text']):
                diacritizer=NakdimonDiacritizer.__new__(NakdimonDiacritizer);diacritizer.model=RecordedHebrewModel(self,case['events'])
                self.assertEqual(diacritizer(case['text']),case['output']);self.assertFalse(diacritizer.model.events)
        for case in self.oracle['hebrew']['phonemes']:
            with self.subTest(ipa=case['text']):
                phonemizer=HebrewPhonemizer.__new__(HebrewPhonemizer);phonemizer.diacritizer=NakdimonDiacritizer.__new__(NakdimonDiacritizer)
                phonemizer.diacritizer.model=RecordedHebrewModel(self,case['events'])
                self.assertEqual(phonemizer.phonemize(case['text']),case['output']);self.assertFalse(phonemizer.diacritizer.model.events)
        self.assertEqual([normalize(c)for c in 'ךםןףץ'],list('ךםןףץ'))

    def test_original_arabic_hints_numerals_and_threshold_30_cases(self):
        diacritizer=TashkeelDiacritizer()
        for case in self.oracle['arabic']['cases']:
            with self.subTest(text=case['text'],threshold=case['threshold']):
                diacritizer.session=RecordedArabicModel(self,case['events'])
                self.assertEqual(diacritizer.diacritize(case['text'],case['threshold']),case['output']);self.assertFalse(diacritizer.session.events)
        for case in self.oracle['arabic']['cases']:
            if case['threshold']is None:
                diacritizer.session=RecordedArabicModel(self,case['events'])
                # Original __call__ accepts the argument but does not forward it.
                self.assertEqual(diacritizer(case['text'],taskeen_threshold=.8),case['output'])
        with self.assertRaises(TashkeelError):diacritizer('ب'*12001)

    def test_bundled_original_models_run_real_neural_inference(self):
        arabic=TashkeelDiacritizer();hebrew=NakdimonDiacritizer()
        for data,model in ((self.oracle['arabic']['cases'],arabic),(self.oracle['hebrew']['cases'],hebrew)):
            cases=[case for case in data if case['events']][:2]
            for case in cases:
                with self.subTest(text=case['text'],kind=type(model).__name__):
                    actual=model.diacritize(case['text'],case['threshold'])if model is arabic else model(case['text'])
                    self.assertEqual(actual,case['output'])
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'changed.onnx';path.write_bytes(b'unknown model data')
            with self.assertRaisesRegex(ValueError,'reviewed'):NakdimonModel(path)


if __name__=='__main__':unittest.main()
