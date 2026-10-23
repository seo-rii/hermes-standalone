"""Original PiperVoice observations with actual pure graph and text inference."""
import base64
from dataclasses import asdict
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave
import zlib

from standalone.local_piper_voice import PiperVoice,AudioChunk,PhonemeAlignment
from standalone.local_piper_config import PiperConfig,PhonemeType,SynthesisConfig
from standalone.local_piper_phoneme import ESPEAK_DATA_DIR
from standalone.local_piper_phoneme_ids import DEFAULT_PHONEME_ID_MAP,phonemes_to_ids

ROOT=Path(__file__).resolve().parent

class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'ctypes','subprocess','numpy','onnxruntime','piper','espeakbridge','torch'}:
            raise AssertionError('Native dependency requested by actual PiperVoice '+fullname)

class RecordModel:
    """Observe inputs without replacing graph execution."""
    def __init__(self,model):self.model=model;self.events=[]
    def run(self,inputs):
        self.events.append({key:dict(shape=list(tensor.shape),dtype={1:'float32',7:'int64'}[tensor.dtype],values=list(tensor.values))for key,tensor in inputs.items()})
        return self.model.run(inputs)

class PiperVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper=json.loads((ROOT/'fixtures/local_piper_voice.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64']))
        assert hashlib.sha256(raw).hexdigest()==wrapper['data_sha256']
        cls.fixture=json.loads(raw)
        from standalone import vendor
        cls.thai_library_available=False
        root=Path(vendor.__file__).resolve().parents[1]
        packages=vendor.load_dependencies(root/'standalone/dependencies.json')['packages']
        selected=[row for row in packages if row['name']in {'tltk','unicode-rbnf'}]
        cache=root/'standalone/.cache/downloads'
        if len(selected)==2 and all((cache/row['filename']).is_file()for row in selected):
            temporary=tempfile.TemporaryDirectory(prefix='hermes-piper-voice-thai-')
            cls.addClassCleanup(temporary.cleanup)
            library=Path(temporary.name)/'vendor'
            for package in selected:
                for name,content in vendor.read_wheel(package,cache)[0].items():
                    path=library/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            sys.path.insert(0,str(library));cls.addClassCleanup(sys.path.remove,str(library))
            cls.thai_library_available=True

    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)

    def test_original_default_map_and_12_multi_id_conversions(self):
        reference=self.fixture['contracts']
        self.assertEqual(DEFAULT_PHONEME_ID_MAP,reference['default_map'])
        self.assertEqual(len(reference['phoneme_ids']),12)
        for case in reference['phoneme_ids']:
            self.assertEqual(phonemes_to_ids(case['phonemes'],case['mapping']),case['ids'])

    def _compare_chunk(self,actual,wanted,name):
        self.assertEqual((actual.sample_rate,actual.sample_width,actual.sample_channels),(wanted['rate'],wanted['width'],wanted['channels']),name)
        self.assertEqual(actual.phonemes,wanted['phonemes'],name);self.assertEqual(actual.phoneme_ids,wanted['ids'],name)
        expected=[part[0]for part in struct.iter_unpack('<f',base64.b64decode(wanted['audio_f32le']))]
        self.assertEqual(len(actual.audio_float_array),len(expected),name)
        self.assertLessEqual(max((abs(left-right)for left,right in zip(actual.audio_float_array,expected,strict=True)),default=0.),1e-6,name)
        self.assertEqual(actual.audio_int16_bytes,base64.b64decode(wanted['pcm_s16le']),name)
        self.assertEqual(actual.phoneme_id_samples,wanted['samples'],name)
        alignment=None if actual.phoneme_alignments is None else[asdict(value)for value in actual.phoneme_alignments]
        self.assertEqual(alignment,wanted['alignments'],name)

    def _compare_wav(self,voice,settings,case):
        if'wav'not in case:return
        with io.BytesIO()as output:
            with wave.open(output,'wb')as writer:
                result=voice.synthesize_wav(case['text'],writer,settings,include_alignments=True)
            self.assertEqual(output.getvalue(),base64.b64decode(case['wav']),case['name'])
            self.assertEqual([asdict(value)for value in result],case['wav_alignments'],case['name'])

    def test_original_44_text_graph_speaker_normalization_alignment_and_wav_cases(self):
        cases=self.fixture['contracts']['cases'];self.assertEqual(len(cases),44)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'voice.onnx'
            for case in cases:
                path.write_bytes(base64.b64decode(case['model']))
                path.with_suffix('.onnx.json').write_text(json.dumps(case['config']))
                voice=PiperVoice.load(path);self.assertEqual(voice.espeak_data_dir,ESPEAK_DATA_DIR)
                voice.session=RecordModel(voice.session);settings=SynthesisConfig(**case['settings'])
                chunks=list(voice.synthesize(case['text'],settings,case['include_alignments']))
                self.assertEqual(len(chunks),len(case['chunks']),case['name'])
                for actual,wanted in zip(chunks,case['chunks'],strict=True):self._compare_chunk(actual,wanted,case['name'])
                self._compare_wav(voice,settings,case)
                self.assertEqual(voice.session.events,case['events'],case['name'])

    def test_original_54_real_espeak_raw_blocks_clusters_arabic_and_hebrew_cases(self):
        cases=self.fixture['phonemize']['cases'];self.assertEqual(len(cases),54);voices={}
        for case in cases:
            key=(case['kind'],case['voice'],tuple(tuple(value)for value in case['clusters'])if case['clusters']else None)
            if key not in voices:
                config=dict(num_symbols=256,num_speakers=1,audio={'sample_rate':22050},espeak={'voice':case['voice']},phoneme_id_map=DEFAULT_PHONEME_ID_MAP,phoneme_type=case['kind'])
                if case['clusters']:config['vowel_clusters']=case['clusters']
                voices[key]=PiperVoice(session=None,config=PiperConfig.from_dict(config))
            voice=voices[key];voice.use_tashkeel=case['tashkeel'];voice.taskeen_threshold=case['threshold']
            phones=voice.phonemize(case['text']);self.assertEqual(phones,case['phonemes'],(case['voice'],case['text']))
            self.assertEqual([voice.phonemes_to_ids(value)for value in phones],case['ids'],(case['voice'],case['text']))

    def test_unavailable_cuda_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError,'CUDA'):PiperVoice.load('unused.onnx',use_cuda=True)

    def test_original_40_thai_facade_consumers_and_per_voice_backend_cache(self):
        if not self.thai_library_available:self.skipTest('Explicitly fetch the reviewed Thai source dependencies')
        from standalone.local_piper_thai import ThaiPhonemizer
        cases=self.fixture['thai']['cases'];self.assertEqual(len(cases),40)
        config=PiperConfig(256,1,22050,'th',DEFAULT_PHONEME_ID_MAP,PhonemeType.THAI)
        voice=PiperVoice(None,config);backend=None
        for case in cases:
            actual=voice.phonemize(case['text'])
            self.assertEqual(actual,case['phonemes'],case['text'])
            self.assertEqual([voice.phonemes_to_ids(row)for row in actual],case['ids'],case['text'])
            self.assertIsInstance(voice._thai_phonemizer,ThaiPhonemizer)
            if backend is None:backend=voice._thai_phonemizer
            else:self.assertIs(voice._thai_phonemizer,backend)
        other=PiperVoice(None,config)
        self.assertEqual(other.phonemize(cases[0]['text']),cases[0]['phonemes'])
        self.assertIsNot(other._thai_phonemizer,backend)

if __name__=='__main__':unittest.main(verbosity=2)
