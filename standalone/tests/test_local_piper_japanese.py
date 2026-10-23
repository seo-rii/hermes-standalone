"""Original Japanese full-controller and PiperVoice observations, SDK free."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone.agent_probe_worker import fixture_environment


WORKER = r'''
import base64,hashlib,importlib.abc,io,json,pathlib,sys,wave,zlib
root,scratch,controller_path,voice_path=sys.argv[1:]
root=pathlib.Path(root);scratch=pathlib.Path(scratch)
sys.path.insert(0,str(root))
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(root,root/'vendor',scratch);sys.addaudithook(guard)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in {'numpy','onnxruntime','pyopenjtalk','sudachipy','ctypes','cffi','piper','torch'}:
            raise AssertionError('Japanese port attempted native or external SDK '+fullname)
sys.meta_path.insert(0,NoNative())
from standalone.local_piper_japanese import JapanesePhonemizer
from standalone.local_piper_voice import PiperVoice
from standalone.local_piper_config import SynthesisConfig
package=root/'standalone'
manifest=json.loads((package/'piper-japanese-resources-provenance.json').read_bytes())
for name,record in manifest['resources'].items():
    data=(package/name).read_bytes()
    assert len(data)==record['bytes']and hashlib.sha256(data).hexdigest()==record['sha256'],name
phonemizer=JapanesePhonemizer();controller=phonemizer.pyopenjtalk
raw=pathlib.Path(controller_path).read_bytes()
assert hashlib.sha256(raw).hexdigest()=='a59f5ab841b21c02b52f0123e47d3759b91dd5ecf50d00ceb6b83c8d00dbf22d'
fixture=json.loads(zlib.decompress(raw))
def encode(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def observe(function,*args):
    try:return function(*args)
    except(ValueError,TypeError,RuntimeError)as error:return {'error_type':type(error).__name__,'error':str(error)}
for index,case in enumerate(fixture['cases']):
    actual=observe(controller.run_frontend,case['input']);expected=case['output']
    if 'error'in expected:
        assert actual==expected,(index,case['input']);continue
    assert encode(actual)==encode(expected['features']),(index,case['input'])
    assert controller.make_label(actual)==expected['labels'],(index,case['input'])
for index,case in enumerate(fixture['detailed']):
    assert encode(observe(controller.run_frontend_detailed,case['input']))==encode(case['output']),(index,case['input'])
cases=fixture['cases']
indices=sorted(set(range(0,len(cases),max(1,len(cases)//400)))|set(range(max(0,len(cases)-150),len(cases))))
for index in indices:
    case=cases[index]
    if 'error'in case['output']:continue
    assert phonemizer.phonemize_openjtalk(case['input'])==case['output']['phones'],(index,case['input'])
    assert phonemizer.phonemize(case['input'])==case['output']['ipa'],(index,case['input'])
raw=pathlib.Path(voice_path).read_bytes()
assert hashlib.sha256(raw).hexdigest()=='4a13efe16550e71bd2a9f673f12bc912839e735e7228a9d3796ef2098a9933bf'
voice_fixture=json.loads(zlib.decompress(raw))
def prepare(label):
    profile=scratch/label;profile.mkdir()
    model=profile/'actual.onnx';model.write_bytes(base64.b64decode(voice_fixture['small_voice_model']))
    mapping=voice_fixture['id_mapping']
    config=dict(voice_fixture['small_voice_config'],phoneme_type='japanese',phoneme_id_map=mapping,
                num_symbols=max(number for values in mapping.values()for number in values)+1)
    pathlib.Path(str(model)+'.json').write_text(json.dumps(config))
    voice=PiperVoice.load(model,download_dir=profile)
    assert not hasattr(voice,'_japanese_phonemizer')
    assert voice.phonemize('')==[]
    return voice,voice._japanese_phonemizer
voice,backend=prepare('profile-a')
for case in voice_fixture['cases']:
    actual=voice.phonemize(case['text'])
    assert actual==case['phonemes'],case['text']
    assert[voice.phonemes_to_ids(row)for row in actual]==case['ids'],case['text']
    assert voice._japanese_phonemizer is backend
voice.config.phoneme_id_map=voice_fixture['custom_id_mapping']
for case in voice_fixture['custom_ids']:assert voice.phonemes_to_ids(case['phonemes'])==case['ids']
voice.config.phoneme_id_map=voice_fixture['id_mapping']
expected=voice_fixture['synthesis'];settings=SynthesisConfig(**expected['settings'])
chunks=list(voice.synthesize(expected['text'],settings,include_alignments=True))
assert len(chunks)==len(expected['chunks'])
for chunk,wanted in zip(chunks,expected['chunks'],strict=True):
    assert chunk.phonemes==wanted['phonemes']and chunk.phoneme_ids==wanted['ids']
    assert chunk.phoneme_id_samples==wanted['samples']
    assert chunk.audio_int16_bytes==base64.b64decode(wanted['pcm'])
output=io.BytesIO()
with wave.open(output,'wb')as writer:voice.synthesize_wav(expected['text'],writer,settings)
assert output.getvalue()==base64.b64decode(expected['wav'])
other,other_backend=prepare('profile-b')
assert other.download_dir!=voice.download_dir and other_backend is not backend
assert other_backend.pyopenjtalk is backend.pyopenjtalk
assert not guard.events,guard.events
print(json.dumps({'controller_cases':len(cases),'detailed_cases':len(fixture['detailed']),
                  'labels':sum(len(case['output'].get('labels',[]))for case in cases),
                  'voice_cases':len(voice_fixture['cases']),'resource_files':len(manifest['resources']),
                  'default_reading_corrections':True,'actual_owned_voice_graph_pcm_wav':True,'blocked_events':guard.events}))
'''


class JapanesePiperTests(unittest.TestCase):
    def test_original_default_controller_dictionaries_and_actual_voice_graph(self):
        root = Path(__file__).resolve().parents[2]
        fixtures = Path(__file__).parent / 'fixtures'
        with tempfile.TemporaryDirectory(prefix='hermes-public-japanese-') as temporary:
            scratch = Path(temporary)
            for name in ('home', 'work', 'tmp'):
                (scratch / name).mkdir()
            result = subprocess.run(
                [sys.executable, '-I', '-S', '-B', '-X', 'utf8', '-c', WORKER, str(root), str(scratch),
                 str(fixtures / 'local_piper_japanese.json.zlib'),
                 str(fixtures / 'local_piper_japanese_voice.json.zlib')],
                cwd=scratch / 'work', env=fixture_environment(scratch), capture_output=True,
                text=True, encoding='utf8', timeout=900,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual((report['controller_cases'], report['detailed_cases'], report['labels']),
                             (9001, 161, 126017))
            self.assertEqual((report['voice_cases'], report['resource_files']), (58, 86))
            self.assertTrue(report['default_reading_corrections'])
            self.assertTrue(report['actual_owned_voice_graph_pcm_wav'])
            self.assertEqual(report['blocked_events'], [])
