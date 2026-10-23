"""Pinned native CT2/Faster-Whisper segment consumer traces, without native code.

The finite replay engine is a test dependency: all returned generation and
alignment values were recorded from the original trained model and SDK.
Production selects the qualified pure engine, never this fixture class.
"""
import base64
import copy
from dataclasses import asdict
import hashlib
import importlib.abc
import json
import io
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import zlib
import wave

from standalone import local_whisper as whisper
from standalone import local_whisper_segments as segments
from standalone.local_whisper_features import MelFeatures
from standalone.local_whisper_tokenizer import ByteBPETokenizer
from standalone.local_whisper_tokens import Tokenizer


def normalize(value):
    return json.loads(json.dumps(value))


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'numpy','av','tokenizers','ctranslate2','onnxruntime','torch'}:
            raise AssertionError('Whisper segment tests attempted to import '+fullname)


class RecordedEngine:
    def __init__(self, case):
        self.events = copy.deepcopy(case['events'])
        self.is_multilingual = case['model'] == 'tiny'
        self.device, self.device_index, self.n_mels = 'cpu', [0], 80

    def _next(self, operation):
        if not self.events:
            raise AssertionError('Unexpected inference call '+operation)
        event = self.events.pop(0)
        if event['operation'] != operation:
            raise AssertionError(('Wrong inference order', operation, event['operation']))
        return event

    def encode(self, features, to_cpu=False):
        event = self._next('encode')
        if normalize((1,)+features.shape) != event['shape'] or to_cpu != event['to_cpu']:
            raise AssertionError('Wrong feature shape or inference placement')
        return SimpleNamespace(shape=(1,1500,384))

    def generate(self, encoded, prompts, **kwargs):
        event = self._next('generate')
        if normalize(prompts) != event['prompts'] or normalize(kwargs) != event['kwargs']:
            raise AssertionError(('Changed prompt or decoding options',prompts,kwargs,event))
        return [SimpleNamespace(**result) for result in event['output']]

    def detect_language(self, encoded):
        return self._next('detect_language')['output']

    def align(self, encoded, sot, text, num_frames, **kwargs):
        event = self._next('align')
        arguments = dict(sot=sot,text=text,num_frames=num_frames,kwargs=kwargs)
        if normalize(arguments) != {name:event[name] for name in arguments}:
            raise AssertionError('Changed word alignment arguments')
        return [SimpleNamespace(**result) for result in event['output']]


class LocalWhisperSegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = Path(__file__).parent/'fixtures'
        wrapper = json.loads((fixtures/'local_whisper_segments.json').read_text())
        body = zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        if hashlib.sha256(body).hexdigest() != wrapper['data_sha256']:
            raise AssertionError('Native Whisper segment fixture digest mismatch')
        cls.cases = json.loads(body)
        cls.tokenizers = {name:value.encode() for name,value in cls.cases['tokenizers'].items()}

    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0,guard)
        self.addCleanup(sys.meta_path.remove,guard)

    def compare(self, actual, expected):
        if isinstance(expected,dict):
            self.assertEqual(actual.keys(),expected.keys())
            for key in expected:
                self.compare(actual[key],expected[key])
        elif isinstance(expected,list):
            self.assertEqual(len(actual),len(expected))
            for left,right in zip(actual,expected):
                self.compare(left,right)
        elif isinstance(expected,float):
            self.assertAlmostEqual(actual,expected,delta=1e-10+abs(expected)*1e-10)
        else:
            self.assertEqual(actual,expected)

    def test_native_transcription_seek_fallback_prompts_languages_and_words(self):
        self.assertGreaterEqual(len(self.cases['scenarios']),12)
        for case in self.cases['scenarios']:
            with self.subTest(scenario=case['name']):
                engine = RecordedEngine(case)
                files = {'tokenizer.json':self.tokenizers[case['model']],
                         'preprocessor_config.json':b'{}'}
                with mock.patch.object(whisper,'_engine',return_value=engine):
                    model = whisper.WhisperModel('fixture-'+case['model'],files=files,
                                                 device='cpu',compute_type='float32',cpu_threads=1)
                raw = base64.b64decode(case['audio_f32le_base64'],validate=True)
                audio = [value[0] for value in struct.iter_unpack('<f',raw)]
                output,info = model.transcribe(audio,**case['options'])
                if case['options'].get('language') is not None or not engine.is_multilingual:
                    self.assertEqual(engine.events,case['events'])
                self.compare(normalize(asdict(info)),case['info_before'])
                actual = [asdict(segment) for segment in output]
                self.compare(normalize(actual),case['output'])
                self.compare(normalize(asdict(info)),case['info'])
                self.assertEqual(engine.events,[])
                self.assertIn('en',model.supported_languages)
                if not engine.is_multilingual:
                    self.assertEqual(model.supported_languages,['en'])

    def test_original_timestamp_slices_prompt_truncation_and_suppression(self):
        helpers = self.cases['helpers']
        self.assertEqual({key:len(value) for key,value in helpers.items()},
                         {'split':48,'prompt':144,'suppression':8,'restore':3,'errors':2})
        model = whisper.WhisperModel.__new__(whisper.WhisperModel)
        model.time_precision,model.input_stride,model.max_length = .02,2,448
        tokenizers = {name:Tokenizer(ByteBPETokenizer(data),name=='tiny',task='transcribe',
                                     language='ko' if name=='tiny' else 'en')
                      for name,data in self.tokenizers.items()}
        for operation,method in [('split','_split_segments_by_timestamps'),('prompt','get_prompt')]:
            for number,case in enumerate(helpers[operation]):
                with self.subTest(operation=operation,case=number):
                    actual = getattr(model,method)(tokenizers[case['model']],**case['arguments'])
                    self.compare(normalize(actual),case['output'])
        for case in helpers['suppression']:
            self.assertEqual(normalize(segments.get_suppressed_tokens(tokenizers[case['model']],list(case['input']))),case['output'])

    def test_original_restored_word_boundaries_and_invalid_length_errors(self):
        for case in self.cases['helpers']['restore']:
            values = copy.deepcopy(case['input'])
            for value in values:
                if value['words']:
                    value['words'] = [whisper.Word(**word) for word in value['words']]
            actual = list(whisper.restore_speech_timestamps([whisper.Segment(**value) for value in values],case['chunks'],16000))
            self.compare(normalize([asdict(value) for value in actual]),case['output'])
        model = whisper.WhisperModel.__new__(whisper.WhisperModel)
        model.max_length,model.time_precision = 448,.02
        for case in self.cases['helpers']['errors']:
            with self.assertRaises(ValueError) as caught:
                model.generate_with_fallback(None,case['prompt'],None,whisper.TranscriptionOptions(**case['options']))
            self.assertEqual(type(caught.exception).__name__,case['type'])
            if case['options']['temperatures']:
                self.assertEqual(str(caught.exception),case['message'])
            else:
                # Python 3.14 changed the built-in max() diagnostic. The
                # original SDK raises the same ValueError on each interpreter.
                self.assertRegex(str(caught.exception),r'^max\(\).*empty')

    def test_native_multilingual_threshold_and_multiple_window_vote(self):
        self.assertEqual(len(self.cases['language_helpers']),3)
        for case in self.cases['language_helpers']:
            model = whisper.WhisperModel.__new__(whisper.WhisperModel)
            model.feature_extractor = SimpleNamespace(nb_max_frames=3000)
            model.model = RecordedEngine(dict(model='tiny',events=case['events']))
            features = MelFeatures(tuple((case['fill'],)*case['shape'][1] for _ in range(case['shape'][0])))
            actual = model.detect_language(features=features,**case['options'])
            self.compare(normalize(actual),case['output'])
            self.assertEqual(model.model.events,[])

    def test_model_data_alias_cache_and_prepared_wav_boundary(self):
        for name,expected in [('tiny','Systran/faster-whisper-tiny'),('base.en','Systran/faster-whisper-base.en'),
                              ('large','Systran/faster-whisper-large-v3'),('custom/model','custom/model')]:
            with mock.patch.object(whisper,'_snapshot_download',return_value='/model-data') as download:
                actual = whisper.download_model(name,output_dir='/model-data',cache_dir='/model-cache',
                                                 revision='a'*40,local_files_only=True,use_auth_token=True)
            self.assertEqual(actual,'/model-data')
            download.assert_called_once_with(expected,local_files_only=True,
                                             allow_patterns=['config.json','preprocessor_config.json','model.bin','tokenizer.json','vocabulary.*'],
                                             revision='a'*40,local_dir='/model-data',cache_dir='/model-cache',token=True)
        with self.assertRaisesRegex(ValueError,'Invalid model size'):
            whisper.download_model('not-a-model')
        data = io.BytesIO()
        with wave.open(data,'wb') as output:
            output.setparams((1,2,16000,0,'NONE','not compressed'))
            output.writeframes(struct.pack('<5h',-32768,-1,0,1,32767))
        data.seek(0)
        self.assertEqual(list(whisper.decode_audio(data)),[-1,-1/32768,0,1/32768,32767/32768])


if __name__=='__main__':
    unittest.main()
