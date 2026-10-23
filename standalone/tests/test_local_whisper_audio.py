"""Original native PyAV/SWR and libopus file-input golden vectors."""
import base64
from dataclasses import asdict
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

from standalone import local_whisper_audio as audio
from standalone.local_whisper_audio_resample import convert_audio, resample, to_s16
from standalone.local_whisper_audio_wave import decode_wave
from standalone import local_whisper as whisper
from standalone.tests import test_local_whisper_segments as segment_tests


def _wave(values, rate=48000, bits=16, channels=1, floating=False):
    if floating:
        raw=b''.join(struct.pack('<'+('f' if bits==32 else 'd'),v)for v in values)
    elif bits==8:
        raw=bytes(values)
    else:
        raw=b''.join(int(v).to_bytes(bits//8,'little',signed=True)for v in values)
    fmt=struct.pack('<HHIIHH',3 if floating else 1,channels,rate,rate*channels*(bits//8),channels*(bits//8),bits)
    body=b'WAVEfmt '+struct.pack('<I',len(fmt))+fmt+b'data'+struct.pack('<I',len(raw))+raw
    return b'RIFF'+struct.pack('<I',len(body))+body+(b'\0'if len(raw)&1 else b'')


def _pcm(text):
    return [v[0]for v in struct.iter_unpack('<h',base64.b64decode(text,validate=True))]


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'av','numpy','ctranslate2','tokenizers','onnxruntime','torch','ctypes','subprocess','pydub','soundfile'}:
            raise AssertionError('Audio port attempted native or process import '+fullname)


class LocalWhisperAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=Path(__file__).parent/'fixtures/local_whisper_audio.json'
        wrapper=json.loads(fixture.read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        if hashlib.sha256(raw).hexdigest()!=wrapper['data_sha256']:
            raise AssertionError('Native audio fixture digest mismatch')
        cls.oracle=json.loads(raw)
        wrapper=json.loads((fixture.parent/'local_whisper_audio_vorbis.json').read_text())
        metadata=zlib.decompress(base64.b64decode(wrapper['metadata_zlib_base64'],validate=True))
        if hashlib.sha256(metadata).hexdigest()!=wrapper['metadata_sha256']:
            raise AssertionError('Native Vorbis metadata digest mismatch')
        cls.vorbis=json.loads(metadata)['groups']

    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard)
        self.addCleanup(sys.meta_path.remove,guard)

    def compare_pcm(self,actual,expected,delta=1):
        self.assertEqual(len(actual),len(expected))
        for left,right in zip(actual,expected):
            self.assertEqual(len(left),len(right))
            self.assertLessEqual(max((abs(a-b)for a,b in zip(left,right)),default=0),delta)

    def test_original_swr_672_format_rate_and_channel_combinations(self):
        cases=iter(self.oracle['formats']['cases']);rng=random.Random(3981);count=0
        for kind,bits in ((1,8),(1,16),(1,24),(1,32),(3,32),(3,64)):
            for rate in (8000,12345,16000,22050,44100,48000,96000):
                for channels in range(1,9):
                    floating=kind==3
                    values=([rng.uniform(-1.25,1.25)for _ in range(1025*channels)]if floating else
                            [rng.randrange(256)for _ in range(1025*channels)]if bits==8 else
                            [rng.randrange(-(1<<(bits-1)),1<<(bits-1))for _ in range(1025*channels)])
                    source=_wave(values,rate,bits,channels,floating)
                    parsed=decode_wave(source)
                    for split in (False,True):
                        case=next(cases);count+=1
                        with self.subTest(kind=kind,bits=bits,rate=rate,channels=channels,split=split):
                            self.assertEqual([kind,bits,rate,channels,split],[case[k]for k in ('kind','bits','rate','channels','split')])
                            self.assertEqual(hashlib.sha256(source).hexdigest(),case['sha256'])
                            actual=convert_audio(parsed[1],parsed[0],16000,parsed[2],parsed[3],split)
                            self.compare_pcm(actual,[_pcm(ch)for ch in case['pcm']])
        self.assertEqual(count,672)

    def test_short_edge_reflection_and_nonstandard_phase_ratios(self):
        self.assertEqual(len(self.oracle['short']['cases']),180)
        for case in self.oracle['short']['cases']:
            with self.subTest(rate=case['rate'],length=len(case['values'])):
                actual=[to_s16(v)for v in resample([v/32768 for v in case['values']],case['rate'],16000)]
                self.compare_pcm([actual],[_pcm(case['pcm'])])
        for case in self.oracle['boundaries']['cases']:
            with self.subTest(rates=(case['input_rate'],case['output_rate'])):
                actual=[to_s16(v)for v in resample([v/32768 for v in case['values']],case['input_rate'],case['output_rate'])]
                self.compare_pcm([actual],[_pcm(case['pcm'])])

    def test_extensible_rifx_rf64_wave64_g711_and_64bit_pcm(self):
        self.assertEqual(len(self.oracle['headers']['cases']),212)
        for case in self.oracle['headers']['cases']:
            with self.subTest(case=case['name'],rate=case['rate'],split=case['split']):
                output=audio.decode_audio(io.BytesIO(base64.b64decode(case['input'])),case['rate'],case['split'])
                output=output if case['split']else[output]
                self.compare_pcm([[round(v*32768)for v in ch]for ch in output],[_pcm(ch)for ch in case['pcm']])
        for case in self.oracle['pcm64']:
            values=[(i*187138183469973723)%2**64-2**63 for i in range(257*case['channels'])]
            parsed=decode_wave(_wave(values,case['rate'],64,case['channels']))
            actual=convert_audio(parsed[1],parsed[0],16000,parsed[2],parsed[3],case['split'])
            self.compare_pcm(actual,[_pcm(ch)for ch in case['pcm']])

    def test_native_multistream_opus_gain_preskip_granules_and_chaining(self):
        self.assertEqual(len(self.oracle['ogg']['cases']),32)
        self.assertEqual(len(self.oracle['ogg_metadata']['cases']),38)
        for group in ('ogg','ogg_metadata'):
            for case in self.oracle[group]['cases']:
                with self.subTest(group=group,case=case.get('name',case.get('file')),split=case['split']):
                    output=audio.decode_audio(io.BytesIO(base64.b64decode(case['input'])),split_stereo=case['split'])
                    output=output if case['split']else[output]
                    self.compare_pcm([[round(v*32768)for v in ch]for ch in output],[_pcm(ch)for ch in case['pcm']])

    def test_mp3_native_swr_all_rates_downmix_and_stereo(self):
        self.assertEqual(len(self.oracle['mp3']['cases']),72)
        for case in self.oracle['mp3']['cases']:
            with self.subTest(case=case['name'],rate=case['rate'],split=case['split']):
                output=audio.decode_audio(io.BytesIO(base64.b64decode(case['input'])),case['rate'],case['split'])
                output=output if case['split']else[output]
                self.compare_pcm([[round(v*32768)for v in ch]for ch in output],[_pcm(ch)for ch in case['pcm']])

    def test_aac_native_swr_868_file_rate_and_downmix_cases(self):
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_whisper_audio_aac.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['metadata_zlib_base64'],validate=True))
        self.assertEqual(hashlib.sha256(raw).hexdigest(),wrapper['metadata_sha256'])
        inputs={case['name']:base64.b64decode(case['data'],validate=True)for case in json.loads(raw)['files']}
        cases=self.oracle['aac']['cases']
        self.assertEqual(len(inputs),217)
        self.assertEqual({(c['name'],c['rate'],c['split'])for c in cases},
                         {(name,rate,split)for name in inputs for rate in (16000,22050)for split in (False,True)})
        for case in cases:
            with self.subTest(case=case['name'],rate=case['rate'],split=case['split']):
                output=audio.decode_audio(io.BytesIO(inputs[case['name']]),case['rate'],case['split'])
                output=output if case['split']else[output]
                self.compare_pcm([[round(v*32768)for v in ch]for ch in output],[_pcm(ch)for ch in case['pcm']])

    def test_aac_adts_and_m4a_magic_selection_path_and_owned_cursor(self):
        cases=[case for case in self.oracle['consumers']if case['file'].endswith(('.aac','.m4a'))]
        self.assertEqual(len(cases),4)
        for case in cases:
            with self.subTest(file=case['file']):
                data=base64.b64decode(case['input'],validate=True)
                expected=_pcm(case['pcm'])
                with tempfile.TemporaryDirectory()as directory:
                    path=Path(directory)/'audio.without_codec_extension';path.write_bytes(data)
                    self.compare_pcm([[round(v*32768)for v in audio.decode_audio(path)]],[expected])
                source=io.BytesIO(b'prefix'+data);source.seek(6)
                self.compare_pcm([[round(v*32768)for v in audio.decode_audio(source)]],[expected])
                self.assertFalse(source.closed)

    def test_he_aac_native_swr_all_1444_file_rate_and_downmix_cases(self):
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_whisper_audio_he_aac.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['metadata_zlib_base64'],validate=True))
        self.assertEqual(hashlib.sha256(raw).hexdigest(),wrapper['metadata_sha256'])
        inputs={case['name']:base64.b64decode(case['data'],validate=True)for case in json.loads(raw)['files']}
        cases=self.oracle['he_aac']['cases'];self.assertEqual(len(inputs),361);self.assertEqual(len(cases),1444)
        self.assertEqual({(case['name'],case['rate'],case['split'])for case in cases},
                         {(name,rate,split)for name in inputs for rate in (16000,22050)for split in (False,True)})
        for case in cases:
            with self.subTest(file=case['name'],rate=case['rate'],split=case['split']):
                output=audio.decode_audio(io.BytesIO(inputs[case['name']]),case['rate'],case['split'])
                output=output if case['split']else[output]
                self.assertEqual([hashlib.sha256(base64.b64decode(value,validate=True)).hexdigest()for value in case['pcm']],case['pcm_sha256'])
                self.compare_pcm([[round(v*32768)for v in channel]for channel in output],[_pcm(value)for value in case['pcm']])

    def test_he_aac_all_27_original_changing_format_refusals(self):
        from standalone.local_whisper_audio_aac_config import AacError
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_whisper_audio_he_aac.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['metadata_zlib_base64'],validate=True))
        self.assertEqual(hashlib.sha256(raw).hexdigest(),wrapper['metadata_sha256'])
        inputs={case['name']:base64.b64decode(case['data'],validate=True)for case in json.loads(raw)['changing_format']}
        expected=self.oracle['he_aac']['changing_format'];self.assertEqual(len(expected),27)
        self.assertEqual(set(inputs),{case['name']for case in expected})
        for case in expected:
            with self.subTest(file=case['name']):
                self.assertEqual(case['original']['type'],'ValueError')
                self.assertEqual(case['candidate']['type'],'AacError')
                with self.assertRaises(AacError)as caught:
                    audio.decode_audio(io.BytesIO(inputs[case['name']]))
                self.assertIsInstance(caught.exception,ValueError)
                self.assertEqual(str(caught.exception),case['candidate']['message'])

    def test_vorbis_native_swr_all_1040_file_rate_and_downmix_cases(self):
        inputs={(group,case['name']):base64.b64decode(case['data'],validate=True)
                for group in ('xiph','ffmpeg','container')for case in self.vorbis[group]['cases']}
        cases=self.oracle['vorbis']['cases'];self.assertEqual(len(inputs),260);self.assertEqual(len(cases),1040)
        self.assertEqual({(case['group'],case['name'],case['rate'],case['split'])for case in cases},
                         {(group,name,rate,split)for group,name in inputs
                          for rate in (16000,22050)for split in (False,True)})
        for case in cases:
            with self.subTest(file=case['name'],group=case['group'],rate=case['rate'],split=case['split']):
                output=audio.decode_audio(io.BytesIO(inputs[case['group'],case['name']]),case['rate'],case['split'])
                output=output if case['split']else[output]
                self.assertEqual([hashlib.sha256(base64.b64decode(value,validate=True)).hexdigest()
                                  for value in case['pcm']],case['pcm_sha256'])
                self.compare_pcm([[round(v*32768)for v in channel]for channel in output],
                                 [_pcm(value)for value in case['pcm']])

    def test_vorbis_original_demux_errors_and_filtered_decoder_errors(self):
        from standalone.local_whisper_audio_vorbis_bits import VorbisError
        inputs={case['name']:base64.b64decode(case['data'],validate=True)
                for case in self.vorbis['headers']['errors']}
        cases=self.oracle['vorbis']['original_header_boundaries'];self.assertEqual(len(cases),23)
        self.assertEqual(set(inputs),{case['name']for case in cases})
        self.assertEqual(sum('samples'in case['generic']for case in cases),10)
        for case in cases:
            self.assertEqual(case['host'],case['generic'])
            for rate in (16000,22050):
                for split in (False,True):
                    with self.subTest(file=case['name'],rate=rate,split=split):
                        if 'samples'in case['generic']:
                            self.assertEqual(case['candidate'],{'samples':0,'dtype':'float32','invalid_frame':True})
                            output=audio.decode_audio(io.BytesIO(inputs[case['name']]),rate,split)
                            output=output if split else[output]
                            self.assertEqual(len(output),2 if split else 1)
                            self.assertEqual([(channel.typecode,len(channel))for channel in output],
                                             [('f',0)]*(2 if split else 1))
                        else:
                            self.assertEqual(case['candidate']['type'],'VorbisError')
                            with self.assertRaises(VorbisError)as caught:
                                audio.decode_audio(io.BytesIO(inputs[case['name']]),rate,split)
                            self.assertEqual(str(caught.exception),case['candidate']['message'])

    def test_vorbis_magic_path_and_owned_file_cursor(self):
        case=next(case for case in self.oracle['vorbis']['cases']
                  if case['group']=='xiph'and case['name']=='xiph-48000-2-tone'
                  and case['rate']==16000 and not case['split'])
        encoded=next(case for case in self.vorbis['xiph']['cases']if case['name']=='xiph-48000-2-tone')
        data=base64.b64decode(encoded['data'],validate=True);expected=[_pcm(value)for value in case['pcm']]
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'audio.without_ogg_extension';path.write_bytes(data)
            self.compare_pcm([[round(value*32768)for value in audio.decode_audio(path)]],expected)
        source=io.BytesIO(b'prefix'+data);source.seek(6)
        self.compare_pcm([[round(value*32768)for value in audio.decode_audio(source)]],expected)
        self.assertFalse(source.closed)

    def test_mp3_magic_selection_file_cursor_and_container_rejection(self):
        case=next(c for c in self.oracle['consumers']if c['file'].endswith('.mp3'))
        data=base64.b64decode(case['input'])
        expected=_pcm(case['pcm'])
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'audio.without_mp3_extension';path.write_bytes(data)
            self.compare_pcm([[round(v*32768)for v in audio.decode_audio(path)]],[expected])
        source=io.BytesIO(b'prefix'+data);source.seek(6)
        self.compare_pcm([[round(v*32768)for v in audio.decode_audio(source)]],[expected])
        self.assertFalse(source.closed)
        for prefix in (b'RIFF'+bytes(24),b'OggS'+bytes(24),b'fLaC'+bytes(24),
                       b'\x1aE\xdf\xa3'+bytes(24),b'\x00\x00\x00\x18ftypisom'+bytes(12)):
            with self.subTest(prefix=prefix),self.assertRaises(ValueError):
                audio.decode_audio(io.BytesIO(prefix+data))

    def test_actual_native_whisper_file_consumer_prompts_and_segments(self):
        wrapper=json.loads((Path(segment_tests.__file__).parent/'fixtures/local_whisper_segments.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64']))
        tokenizer=json.loads(raw)['tokenizers']['tiny.en'].encode()
        for case in self.oracle['consumers']:
            with self.subTest(file=case['file']):
                data=base64.b64decode(case['input'])
                waveform=audio.decode_audio(io.BytesIO(data))
                self.compare_pcm([[round(v*32768)for v in waveform]],[_pcm(case['pcm'])])
                engine=segment_tests.RecordedEngine(case)
                with mock.patch.object(whisper,'_engine',return_value=engine):
                    model=whisper.WhisperModel('fixture-tiny.en',device='cpu',compute_type='float32',cpu_threads=1,
                        files={'tokenizer.json':tokenizer,'preprocessor_config.json':b'{}'})
                    iterator,info=model.transcribe(io.BytesIO(data),**case['options'])
                    output=[asdict(segment)for segment in iterator]
                self.assertEqual(segment_tests.normalize(output),case['output'])
                self.assertEqual(segment_tests.normalize(asdict(info)),case['info'])
                self.assertEqual(engine.events,[])

    def test_owned_file_cursor_id3_prefix_and_explicit_invalid_inputs(self):
        data=_wave([-32768,-1,0,1,32767],16000)
        source=io.BytesIO(b'prefix'+data);source.seek(6)
        self.assertEqual(list(audio.decode_audio(source)),[-1,-1/32768,0,1/32768,32767/32768])
        self.assertFalse(source.closed)
        for version in (3,4):
            tagged=b'ID3'+bytes((version,0,0,0,0,0,3))+b'tag'+data
            self.assertEqual(list(audio.decode_audio(io.BytesIO(tagged))),[-1,-1/32768,0,1/32768,32767/32768])
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'audio.wav';path.write_bytes(data)
            self.assertEqual(list(audio.decode_audio(path)),[-1,-1/32768,0,1/32768,32767/32768])
        with self.assertRaises(TypeError):audio.decode_audio(io.StringIO('text'))
        for raw in (b'',b'bad',b'ID3',b'ID3\x04\0\0\x80\0\0\0',b'RIFF\0\0\0\0WAVE',data[:15]):
            with self.subTest(raw=raw),self.assertRaises(ValueError):audio.decode_audio(io.BytesIO(raw))
        with self.assertRaises(ValueError):decode_wave(data,max_seconds=0)
        with self.assertRaises(ValueError):audio.decode_audio(io.BytesIO(data),0)
        ogg=bytearray(base64.b64decode(self.oracle['ogg']['cases'][0]['input']));ogg[-1]^=1
        with self.assertRaisesRegex(ValueError,'checksum'):audio.decode_audio(io.BytesIO(ogg))


if __name__=='__main__':unittest.main()
