"""Copied-file acceptance using independent Chinese/Japanese voice observations."""
import json
from pathlib import Path
import zlib


def write_payload(directory):
    fixtures = Path(__file__).parent / 'fixtures'
    chinese = json.loads(zlib.decompress((fixtures / 'local_piper_chinese.json.zlib').read_bytes()))
    japanese = json.loads(zlib.decompress((fixtures / 'local_piper_japanese_voice.json.zlib').read_bytes()))
    voices = {'chinese': chinese['voice'], 'japanese': japanese}
    for name, voice in voices.items():
        texts = ('銀行', '0%和100%。') if name == 'chinese' else ('あわてる', '今日はいい天気です。何をしますか？')
        voice['cases'] = [next(case for case in voice['cases'] if case['text'] == text) for text in texts]
    payload = directory / 'original-language-observations.json'
    payload.write_text(json.dumps({'voices': voices, 'cold': chinese['cold']}, ensure_ascii=False))
    return payload


WORKER = r'''
import base64,hashlib,importlib.abc,io,json,pathlib,runpy,sys,wave
artifact,cache,scratch,payload=sys.argv[1:]
scratch=pathlib.Path(scratch)
bundle=runpy.run_path(artifact,run_name='owned_piper_language_acceptance')
archive,manifest,manifest_bytes,files=bundle['_read_archive']()
dictionary='app/standalone/local_piper_japanese/data/sudachi/system.dic'
assert files[dictionary]['encoding']=='sudachi-v1-blocks'
try:root=bundle['_extract'](archive,files,manifest_bytes,bundle['_cache_directory'](cache))
finally:archive.close()
source=root/'app';library=root/'vendor';sys.path[:0]=[str(source),str(library)]
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(source,library,scratch);sys.addaudithook(guard)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'numpy','onnxruntime','torch','ctypes','cffi','subprocess','piper','g2pw','transformers','pyopenjtalk','sudachipy','regex','sentence_stream'}:
            raise AssertionError('Unexpected Piper language native/SDK dependency '+fullname)
sys.meta_path.insert(0,NoNative())
def offline(event,args):
    if event.startswith('socket.')or event in {'subprocess.Popen','os.system','ctypes.dlopen','ctypes.dlsym'}:
        raise AssertionError('Language runtime requested external execution '+event)
sys.addaudithook(offline)
from standalone.local_piper_voice import PiperVoice
from standalone.local_piper_config import SynthesisConfig
package=source/'standalone'
fixture=json.loads(pathlib.Path(payload).read_text())

def resources(name):
    reviewed=json.loads((package/('piper-'+name+'-resources-provenance.json')).read_bytes())
    for path,row in reviewed['resources'].items():
        raw=(package/path).read_bytes()
        assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],path
    return len(reviewed['resources'])

def prepare(name,observed,label):
    profile=scratch/(name+'-'+label);profile.mkdir()
    model=profile/'actual-voice.onnx';model.write_bytes(base64.b64decode(observed['small_voice_model']))
    mapping=observed['id_mapping']
    config=dict(observed['small_voice_config'],phoneme_type='pinyin'if name=='chinese'else'japanese',
                phoneme_id_map=mapping,num_symbols=max(number for values in mapping.values()for number in values)+1)
    pathlib.Path(str(model)+'.json').write_text(json.dumps(config))
    voice=PiperVoice.load(model,download_dir=profile)
    attribute='_'+name+'_phonemizer'
    assert not hasattr(voice,attribute)
    assert voice.phonemize('')==[]
    return voice,getattr(voice,attribute)

def synthesis(voice,observed):
    expected=observed['synthesis'];settings=SynthesisConfig(**expected['settings'])
    chunks=list(voice.synthesize(expected['text'],settings,include_alignments=True))
    assert len(chunks)==len(expected['chunks'])
    for actual,wanted in zip(chunks,expected['chunks'],strict=True):
        assert actual.phonemes==wanted['phonemes']and actual.phoneme_ids==wanted['ids']
        assert actual.phoneme_id_samples==wanted['samples']
        assert actual.audio_int16_bytes==base64.b64decode(wanted['pcm'])
    output=io.BytesIO()
    with wave.open(output,'wb')as writer:voice.synthesize_wav(expected['text'],writer,settings)
    assert output.getvalue()==base64.b64decode(expected['wav'])

def check(name,observed):
    count=resources(name)
    voice,backend=prepare(name,observed,'a')
    for case in observed['cases']:
        actual=voice.phonemize(case['text'])
        assert actual==case['phonemes'],(name,case['text'])
        assert[voice.phonemes_to_ids(row)for row in actual]==case['ids'],(name,case['text'])
        assert getattr(voice,'_'+name+'_phonemizer')is backend
    voice.config.phoneme_id_map=observed['custom_id_mapping']
    for case in observed['custom_ids']:assert voice.phonemes_to_ids(case['phonemes'])==case['ids']
    voice.config.phoneme_id_map=observed['id_mapping']
    synthesis(voice,observed)
    other,other_backend=prepare(name,observed,'b')
    assert other_backend is not backend and other.download_dir!=voice.download_dir
    if name=='chinese':
        expected=next(row for row in fixture['cold']['cases']if row['kind']=='missing')['files']
        for path,row in expected.items():
            raw=(voice.download_dir/'g2pW'/path).read_bytes()
            assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],path
        assert other_backend.g2p.session_g2pw is not backend.g2p.session_g2pw
    else:assert other_backend.pyopenjtalk is backend.pyopenjtalk
    return {'resources':count,'voice_cases':len(observed['cases']),'custom_ids':len(observed['custom_ids']),
            'real_auxiliary_backend':True,'actual_small_voice_pcm_wav':True,'separate_profiles':True}

report={name:check(name,observed)for name,observed in fixture['voices'].items()}
assert not guard.events,guard.events
print(json.dumps({'languages':report,'dictionary_restored':True,'blocked_events':guard.events}))
'''
