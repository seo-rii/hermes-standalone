"""Original Chinese text behavior and actual embedded G2PW/PiperVoice execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib

from standalone import vendor

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'standalone'


class ChinesePhonemizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='hermes-chinese-tests-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.library = cls.directory / 'vendor'
        packages = vendor.load_dependencies(ROOT / 'standalone/dependencies.json')['packages']
        package = next(row for row in packages if row['name'] == 'unicode-rbnf')
        cache = ROOT / 'standalone/.cache/downloads'
        if not (cache / package['filename']).is_file():
            raise unittest.SkipTest('Explicitly fetch the reviewed unicode-rbnf source archive first')
        for name, content in vendor.read_wheel(package, cache)[0].items():
            target = cls.library / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def run_isolated(self, body):
        home = self.directory / self._testMethodName
        home.mkdir()
        prefix = f'import sys;sys.path[:0]={[str(self.library), str(ROOT)]!r}\n'
        prefix += f'import standalone;standalone.__path__.insert(0,{str(PACKAGE)!r})\n'
        prefix += '''
import base64,hashlib,importlib.abc,io,json,re,struct,wave,zlib
from pathlib import Path
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0] in {'numpy','ctypes','subprocess','onnxruntime','torch','piper','g2pw','transformers','regex','sentence_stream'}:
            raise AssertionError('Chinese runtime requested native/original SDK '+fullname)
sys.meta_path.insert(0,NoNative())
def offline(event,args):
    if event.startswith('socket.') or event in {'subprocess.Popen','os.system','ctypes.dlopen','ctypes.dlsym'}:
        raise AssertionError('Chinese runtime requested network/native execution '+event)
sys.addaudithook(offline)
from standalone import local_piper_chinese as frontend
from standalone import local_piper_chinese_controller as controller
from standalone import local_piper_chinese_sentence as sentences
from standalone.local_piper_chinese_tokenizer import BertWordPiece
from standalone.local_piper_tensor import Tensor
'''
        prefix += f"package=Path({str(PACKAGE)!r});resources=package/'piper_chinese_data'\n"
        prefix += "fixture=json.loads(zlib.decompress((package/'tests/fixtures/local_piper_chinese.json.zlib').read_bytes()))\n"
        environment = {'HOME': str(home), 'USERPROFILE': str(home), 'PATH': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        for name in ('SYSTEMROOT', 'WINDIR'):
            if name in os.environ:
                environment[name] = os.environ[name]
        child = subprocess.run([sys.executable, '-I', '-S', '-B', '-X', 'utf8', '-c', prefix + body],
                               cwd=home, env=environment, capture_output=True, text=True, encoding='utf8', timeout=600)
        self.assertEqual(child.returncode, 0, child.stderr + child.stdout)
        return json.loads(child.stdout)

    def test_original_tokenizer_features_numbers_and_grouped_ids(self):
        report = self.run_isolated('''
source=fixture['text'];normalize=lambda value:json.loads(json.dumps(value,ensure_ascii=False))
tokenizer=BertWordPiece(resources/'bert-base-chinese');assert len(tokenizer.vocab)==21128
for case in source['tokens']:
    assert normalize(controller.wordize_and_map(case['text']))==case['words']
    actual=controller.tokenize_and_map(tokenizer,case['text'])
    assert normalize(actual)==case['tokens'] and tokenizer.convert_tokens_to_ids(actual[0])==case['ids']
for case in source['label_cases']:
    assert normalize(controller.get_phoneme_labels(case['rows']))==case['bare']
    assert normalize(controller.get_char_phoneme_labels(case['rows']))==case['with_character']
for case in source['features']:
    builder=controller._FeatureBuilder(tokenizer,source['labels'],source['char2phonemes'],source['chars'],[case['text']],[case['query']],**case['settings'])
    try:value=builder[0];error=None
    except Exception as exc:value=None;error={'type':type(exc).__name__,'message':str(exc)}
    assert value==case['value'] and error==case['failure']
for case in source['collation']:
    actual=controller._collate(case['samples']);assert set(actual)==set(case['result'])
    for name,value in actual.items():
        expected=case['result'][name]
        assert list(value.shape)==expected['shape'] and {1:'float32',7:'int64'}[value.dtype]==expected['dtype'] and list(value.values)==expected['values']
for case in source['syllables']:
    value=frontend._normalize_g2pw_syllable(case['value'])
    assert value==case['normalized'] and normalize(frontend._split_initial_final_tone(value))==case['split']
numbers=frontend.ChineseFrontend(None)
for case in source['numbers']:
    try:value=numbers._numbers_to_words(case['text']);error=None
    except Exception as exc:value=None;error={'type':type(exc).__name__,'message':str(exc)}
    assert value==case['value'] and error==case['failure']
frontend._LOGGER.disabled=True
for case in source['ids']:assert frontend.phonemes_to_ids(case['phonemes'],case['mapping'])==case['ids']
print(json.dumps({name:len(source[name])for name in ('tokens','label_cases','features','collation','syllables','numbers','ids')}))
''')
        self.assertEqual(report, {'tokens': 180, 'label_cases': 80, 'features': 192, 'collation': 48,
                                  'syllables': 2136, 'numbers': 114, 'ids': 256})

    def test_original_stream_states_and_all_consumed_unicode_predicates(self):
        report = self.run_isolated('''
source=fixture['text']
for case in source['sentence_cases']:
    instance=sentences.SentenceBoundaryDetector()
    for event in case['events']:
        assert list(instance.add_chunk(event['chunk']))==event['sentences']
        assert instance.remaining_text==event['remaining'] and instance.current_sentence==event['current']
    assert instance.finish()==case['tail'] and instance.remaining_text==case['final_remaining'] and instance.current_sentence==case['final_current']
observed=source['unicode_sentence_properties'];values=base64.b64decode(observed['bytes'])
assert hashlib.sha256(values).hexdigest()==observed['sha256']
bits={name:1<<index for index,name in enumerate(observed['properties'])}
for code,mask in enumerate(values):
    char=chr(code);union=bool(mask & (bits['Lu']|bits['Lt']|bits['Lo']))
    assert bool(sentences.SENTENCE_BOUNDARY_RE.search('.'+char+'A'))==bool(mask & bits['SPACE']),('space',code)
    assert bool(sentences.ABBREVIATION_RE.fullmatch(char+'.'))==bool(mask & bits['Lu']),('upper',code)
    assert bool(sentences.ABBREVIATION_RE.fullmatch('A'+char+'.'))==bool(mask & bits['L']),('letter',code)
    match=sentences.ABBREVIATION_RE.search(char+'A.')
    assert (match is None or match.start()!=1)==bool(mask & bits['WORD']),('word',code)
    assert bool(sentences.SENTENCE_BOUNDARY_RE.search('. '+char))==union,('sentence-initial',code)
    assert bool(sentences.SENTENCE_BOUNDARY_RE.search('. '+char+'. '))==(union or bool(mask & bits['DIGIT'])),('numbered-item',code)
print(json.dumps({'streamed_cases':len(source['sentence_cases']),'codepoints':len(values),'consumed_predicates':6}))
''')
        self.assertEqual(report, {'streamed_cases': 160, 'codepoints': 1114112, 'consumed_predicates': 6})

    def test_original_controller_feeds_with_explicit_recorded_nn_dependency(self):
        report = self.run_isolated('''
class RecordedSession:
    """Controller-only dependency; actual neural inference has a separate test."""
    def __init__(self):self.events=[];self.count=0
    def run(self,inputs):
        event=self.events.pop(0);assert set(inputs)==set(event['inputs'])
        for name,value in inputs.items():
            expected=event['inputs'][name]
            assert list(value.shape)==expected['shape'] and {1:'float32',7:'int64'}[value.dtype]==expected['dtype'] and list(value.values)==expected['values']
        self.count+=1
        return [Tensor.make(row['shape'],(value[0]for value in struct.iter_unpack('<f',base64.b64decode(row['bytes']))))for row in event['outputs']]
source=fixture['text'];session=RecordedSession()
converter=controller.G2PWTextController(resources/'model',resources/'bert-base-chinese',resources/'g2pw',session,style='pinyin',enable_non_tradional_chinese=True)
assert converter.labels==source['labels'] and converter.chars==source['chars'] and converter.char2phonemes==source['char2phonemes']
phonemizer=frontend.ChineseFrontend(converter)
for case in source['controller']:
    session.events=list(case['events'])
    actual=converter(case['text'])if case['kind']=='g2pw'else phonemizer.phonemize(case['text'])
    assert actual==case['result'] and not session.events
    if case['kind']=='piper':assert [frontend.phonemes_to_ids(row)for row in actual]==case['ids']
print(json.dumps({'cases':len(source['controller']),'observed_batches':session.count,'scope':'controller-only'}))
''')
        self.assertEqual(report, {'cases': 48, 'observed_batches': 44, 'scope': 'controller-only'})

    def test_real_embedded_model_cold_voice_profile_ids_synthesis_and_wav(self):
        report = self.run_isolated('''
from standalone.local_piper_voice import PiperVoice
from standalone.local_piper_config import SynthesisConfig
source=fixture['voice'];profile=Path('profile-a');profile.mkdir()
model=profile/'actual-voice.onnx';model.write_bytes(base64.b64decode(source['small_voice_model']))
config=dict(source['small_voice_config'],phoneme_type='pinyin',phoneme_id_map=source['id_mapping'],num_symbols=73)
Path(str(model)+'.json').write_text(json.dumps(config))
voice=PiperVoice.load(model,download_dir=profile);assert not hasattr(voice,'_chinese_phonemizer')
assert voice.phonemize('')==[];backend=voice._chinese_phonemizer
assert backend.g2p.model_source=='bert-base-chinese' and len(backend.g2p.tokenizer.vocab)==21128
expected=next(row for row in fixture['cold']['cases']if row['kind']=='missing')['files']
for name,row in expected.items():
    raw=(profile/'g2pW'/name).read_bytes()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
case=source['cases'][0];phonemes=voice.phonemize(case['text'])
assert voice._chinese_phonemizer is backend and phonemes==case['phonemes']
assert [voice.phonemes_to_ids(row)for row in phonemes]==case['ids']
expected=source['synthesis'];settings=SynthesisConfig(**expected['settings'])
chunks=list(voice.synthesize(expected['text'],settings,include_alignments=True));assert len(chunks)==len(expected['chunks'])
for actual,wanted in zip(chunks,expected['chunks']):
    assert actual.phonemes==wanted['phonemes'] and actual.phoneme_ids==wanted['ids']
    assert actual.phoneme_id_samples==wanted['samples'] and actual.audio_int16_bytes==base64.b64decode(wanted['pcm'])
output=io.BytesIO()
with wave.open(output,'wb')as target:voice.synthesize_wav(expected['text'],target,settings)
assert output.getvalue()==base64.b64decode(expected['wav'])
second=PiperVoice.load(model,download_dir=profile);assert second.phonemize('')==[]
assert second._chinese_phonemizer is not backend and second._chinese_phonemizer.g2p.session_g2pw is not backend.g2p.session_g2pw
import shutil
other_profile=Path('profile-b');other_model=other_profile/'g2pW';other_model.mkdir(parents=True)
for name in ('g2pw.onnx','config.py','POLYPHONIC_CHARS.txt','MONOPHONIC_CHARS.txt','version'):
    shutil.copyfile(profile/'g2pW'/name,other_model/name)
settings_path=other_model/'config.py';settings_path.write_text(settings_path.read_text()+'\\nbatch_size = 7\\n')
existing={path.name:hashlib.sha256(path.read_bytes()).hexdigest()for path in other_model.iterdir()}
third=PiperVoice.load(model,download_dir=other_profile);assert third.phonemize('')==[]
assert third._chinese_phonemizer.g2p.batch_size==7 and third.download_dir!=voice.download_dir
assert third._chinese_phonemizer is not backend and third._chinese_phonemizer is not second._chinese_phonemizer
assert {path.name:hashlib.sha256(path.read_bytes()).hexdigest()for path in other_model.iterdir()}==existing
print(json.dumps({'actual_g2pw_model':True,'actual_small_voice_graph':True,'cold_model_bytes_exact':True,'wav_exact':True,'backend_per_voice':True,'separate_profiles':True,'custom_existing_config_preserved':True}))
''')
        self.assertEqual(report, {'actual_g2pw_model': True, 'actual_small_voice_graph': True,
                                  'cold_model_bytes_exact': True, 'wav_exact': True, 'backend_per_voice': True,
                                  'separate_profiles': True, 'custom_existing_config_preserved': True})

    def test_original_partial_existing_and_occupied_model_directory_behavior(self):
        report = self.run_isolated('''
cases={case['kind']:case for case in fixture['cold']['cases']}
def contents(path):
    return {str(item.relative_to(path)):{'bytes':item.stat().st_size,'sha256':hashlib.sha256(item.read_bytes()).hexdigest()}for item in sorted(path.iterdir())if item.is_file()}
partial=Path('partial');partial.mkdir()
(partial/'config.py').write_text('batch_size = 7\\n')
(partial/'keep.txt').write_bytes(b'user supplied unrelated file')
frontend.download_model(partial);assert contents(partial)==cases['partial']['files']
existing=Path('existing');existing.mkdir()
(existing/'g2pw.onnx').write_bytes(b'user owned incomplete or custom graph')
(existing/'config.py').write_bytes(b'user owned settings')
frontend.download_model(existing);assert contents(existing)==cases['existing_model_fast_return']['files']
occupied=Path('occupied');occupied.write_bytes(b'file instead of directory')
try:frontend.download_model(occupied)
except Exception as error:assert type(error).__name__==cases['file_instead_of_directory']['error_type']
else:raise AssertionError('Expected original occupied-path error')
print(json.dumps({'partial_byte_exact':True,'existing_no_write':True,'original_occupied_error':True}))
''')
        self.assertEqual(report, {'partial_byte_exact': True, 'existing_no_write': True, 'original_occupied_error': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
