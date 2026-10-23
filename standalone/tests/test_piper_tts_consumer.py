"""Original TTS registry, profile caches and lifecycle over actual owned Piper models."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib

from standalone.tests.test_entrypoints import BundledSourceTestCase, support


def write_piper_consumer_payload(scratch):
    """Provide original native graph/PCM goldens to a source or copied-artifact worker."""
    scratch = Path(scratch)
    wrapper = json.loads((Path(__file__).parent / 'fixtures/local_piper_voice.json').read_text())
    raw = zlib.decompress(base64.b64decode(wrapper['data_zlib_base64']))
    assert hashlib.sha256(raw).hexdigest() == wrapper['data_sha256']
    original = json.loads(raw)['contracts']['cases']
    cases = {case['name']: case for case in original}
    selected = []
    for index in (0, 2, 3, 4, 5, 6, 7, 8):
        case = cases[f'speakersTrue-alignmentsTrue-{index}']
        settings = {} if index == 0 else {
            'length_scale': 1.25, 'noise_scale': .4, 'noise_w_scale': .7,
            'normalize_audio': True, 'volume': 1., 'speaker_id': 42,
            **case['settings'],
        }
        selected.append({
            'name': case['name'], 'text': case['text'], 'settings': settings,
            'wav': case['wav'], 'events': case['events'][0],
        })
    reference = cases['speakersTrue-alignmentsTrue-0']
    payload = {
        'fixture_sha256': hashlib.sha256(raw).hexdigest(),
        'model': reference['model'], 'config': reference['config'],
        'cases': selected,
    }
    path = scratch / 'piper-native-contract.json'
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path


PIPER_CONSUMER_WORKER = r'''
import base64,ctypes,ctypes.util,hashlib,importlib.abc,json,os,pathlib,sys,wave
context=globals().get('PIPER_CONTEXT')
if context is None:context=json.loads(sys.argv[1])
scratch=pathlib.Path(PIPER_SCRATCH if 'PIPER_SCRATCH'in globals()else sys.argv[2])
payload_path=pathlib.Path(PIPER_FIXTURE if 'PIPER_FIXTURE'in globals()else sys.argv[3])
payload=json.loads(payload_path.read_text())
source=pathlib.Path(context['root'])/'app'
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
class PiperGuard(ProbeGuard):
    def check_path(self,event,value,*,write):
        # Original path protection reads only this interpreter's own venv metadata.
        if not write and isinstance(value,(str,bytes)):
            if pathlib.Path(os.fsdecode(value)).absolute()==pathlib.Path(sys.prefix)/'pyvenv.cfg':
                return
        super().check_path(event,value,write=write)
guard=PiperGuard(source,source.parent/'vendor',scratch)
sys.addaudithook(guard)
def deny_native_loading(event,args):
    if event in ('ctypes.dlopen','ctypes.dlsym'):
        raise AssertionError('Piper TTS consumer cannot load or call a native library')
sys.addaudithook(deny_native_loading)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0] in {'ctypes','numpy','onnxruntime','piper','espeakbridge','torch','soundfile'}:
            raise AssertionError('Native SDK requested by original Piper TTS consumer: '+fullname)
sys.meta_path.insert(0,NoNative())
from agent.sdk_boundary import bundled
from agent.secret_scope import set_secret_scope,reset_secret_scope
from hermes_constants import get_hermes_home,set_hermes_home_override,reset_hermes_home_override
from standalone.local_piper_voice import PiperVoice
from standalone.local_piper_model import PiperModel
from tools import tts_tool as tts,tts_tool_local as local,tts_tool_lifecycle as lifecycle
from tools.registry import registry
from pm.extras import available,extra_supported
assert bundled() and tts._import_piper() is PiperVoice and tts._check_piper_available()
assert available('piper') and extra_supported('piper')
assert all(name not in sys.modules for name in ('piper','numpy','onnxruntime','espeakbridge','torch','soundfile'))
profile=pathlib.Path(os.environ['HERMES_HOME'])
profile.mkdir(parents=True,exist_ok=True)
assert get_hermes_home()==profile
token=set_secret_scope({},profile_home=str(profile))
voice_name=local.DEFAULT_PIPER_VOICE
model_raw=base64.b64decode(payload['model'])

def write_model(directory,name=voice_name):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(name+'.onnx')
    path.write_bytes(model_raw)
    path.with_suffix('.onnx.json').write_text(json.dumps(payload['config']))
    return path

def configure(settings=None):
    config={'tts':{'provider':'piper','keep_warm_seconds':0,'piper':settings or {}}}
    temporary=profile/'config-next.json'
    temporary.write_text(json.dumps(config))
    temporary.replace(profile/'config.yaml')
    return config['tts']

try:
    configure()
    directory=local._get_piper_voices_dir()
    assert directory==profile/'cache/piper-voices'
    model=write_model(directory)
    assert local._resolve_piper_voice_path('',directory)==str(model)
    assert local._resolve_piper_voice_path(str(model),directory)==str(model)
    entry=registry.get_entry('text_to_speech')
    assert entry is not None and entry.check_fn()
    warm=lifecycle.warm_tts_provider()
    assert warm['warmed'] and warm['action']=='loaded',warm
    voice,config=local._load_piper_voice_for_config(tts._load_tts_config())
    assert isinstance(voice,PiperVoice) and isinstance(voice.session,PiperModel)
    assert len(local._piper_voice_cache)==1
    again=lifecycle.warm_tts_provider()
    assert again['warmed'] and again['action']=='cached',again
    assert local._load_piper_voice_for_config(tts._load_tts_config())[0] is voice
    observed=[]
    original_run=voice.session.run
    def observe(inputs):
        observed.append({name:{'shape':list(value.shape),'dtype':{1:'float32',7:'int64'}[value.dtype],'values':list(value.values)}for name,value in inputs.items()})
        return original_run(inputs)
    voice.session.run=observe
    outputs=[]
    for case in payload['cases']:
        configure(case['settings'])
        output=profile/'cache/audio'/(case['name']+'.wav')
        raw_result=registry.dispatch('text_to_speech',{'text':case['text'],'output_path':str(output)})
        result=json.loads(raw_result)
        assert result.get('success') and result['provider']=='piper',result
        path=pathlib.Path(result['file_path'])
        assert path==output and path.read_bytes()==base64.b64decode(case['wav']),case['name']
        assert observed[-1]==case['events'],(case['name'],observed[-1],case['events'])
        with wave.open(str(path),'rb')as reader:
            assert(reader.getframerate(),reader.getnchannels(),reader.getsampwidth(),reader.getnframes())==(22050,1,2,30)
        assert len(local._piper_voice_cache)==1 and next(iter(local._piper_voice_cache.values())) is voice
        outputs.append({'name':case['name'],'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for invalid in (True,'108',None):
        settings={'speaker_id':invalid,'length_scale':1.25,'noise_scale':.4,'noise_w_scale':.7}
        configure(settings)
        output=profile/'cache/audio'/('typed-speaker-'+type(invalid).__name__+'.wav')
        result=json.loads(registry.dispatch('text_to_speech',{'text':payload['cases'][0]['text'],'output_path':str(output)}))
        assert result.get('success'),result
        assert observed[-1]['sid']['values']==[0]
    voice.session.run=original_run
    configure()
    for index in range(3):
        alternate=write_model(directory,'en_US-fixture-'+str(index))
        item,_=local._load_piper_voice_for_config({'piper':{'voice':str(alternate)}})
        assert isinstance(item,PiperVoice) and isinstance(item.session,PiperModel)
    assert len(local._piper_voice_cache)==3
    assert all(not key.startswith(str(model)+'::')for key in local._piper_voice_cache)
    assert list(voice.synthesize(payload['cases'][0]['text'])) # Eviction releases only the cache reference.
    assert lifecycle.release_tts_provider('piper')=={'released':3}
    assert not local._piper_voice_cache
    first=lifecycle.acquire_tts_lease('fixture:desktop')
    second=lifecycle.acquire_tts_lease('fixture:gateway')
    repeated=lifecycle.acquire_tts_lease('fixture:gateway')
    assert first['warmed'] and first['action']=='loaded' and first['leases']==1,first
    assert second['action']=='cached' and second['leases']==2,second
    assert repeated['action']=='cached' and repeated['leases']==2,repeated
    assert lifecycle.release_tts_lease('not-held')=={'leases':2,'released':0}
    assert lifecycle.release_tts_lease('fixture:desktop')=={'leases':1,'released':0}
    assert len(local._piper_voice_cache)==1
    assert lifecycle.release_tts_lease('fixture:gateway')=={'leases':0,'released':1}
    assert not local._piper_voice_cache and not lifecycle.tts_lease_holders()
    other=profile.parent/'secondary'
    legacy=other/'piper_voices_cache'
    other_model=write_model(legacy)
    (other/'config.yaml').write_text(json.dumps({'tts':{'provider':'piper','piper':{},'keep_warm_seconds':0}}))
    home_token=set_hermes_home_override(other)
    other_secret=set_secret_scope({},profile_home=str(other))
    try:
        assert local._get_piper_voices_dir()==legacy
        assert tts._default_output_dir()==str(other/'cache/audio')
        other_warm=lifecycle.warm_tts_provider()
        assert other_warm['warmed'] and other_warm['action']=='loaded',other_warm
        other_voice,_=local._load_piper_voice_for_config(tts._load_tts_config())
        assert isinstance(other_voice,PiperVoice) and other_voice is not voice
        assert next(iter(local._piper_voice_cache))==str(other_model)+'::cuda=False'
        assert model.read_bytes()==model_raw and model.parent==directory
        explicit=other/'custom-voices'
        explicit_model=write_model(explicit)
        explicit_voice,_=local._load_piper_voice_for_config({'piper':{'voices_dir':str(explicit)}})
        assert explicit_voice is not other_voice
        assert str(explicit_model)+'::cuda=False' in local._piper_voice_cache
        try:local._load_piper_voice_for_config({'piper':{'voice':str(other_model),'use_cuda':True}})
        except ValueError as error:assert 'CUDA' in str(error)
        else:raise AssertionError('Explicit CUDA must fail')
        assert str(other_model)+'::cuda=True'not in local._piper_voice_cache
        assert lifecycle.release_tts_provider('piper')=={'released':2}
    finally:
        reset_secret_scope(other_secret)
        reset_hermes_home_override(home_token)
    assert get_hermes_home()==profile and local._get_piper_voices_dir()==directory
    assert not local._piper_voice_cache
    from standalone.runtime import bundle_module_argv
    cold_home=profile.parent/'cold-download-profile'
    cold_directory=cold_home/'models'
    cold_directory.mkdir(parents=True)
    home_token=set_hermes_home_override(cold_home)
    cold_secret=set_secret_scope({},profile_home=str(cold_home))
    previous_helper=local._run_helper
    previous_key=os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY']='private-fixture-launch-profile-key'
    cold_calls=[]
    cold_mode='success'
    def observe_child(command,timeout,**options):
        module,arguments=bundle_module_argv(command)
        assert module=='standalone.local_piper_download_voices'
        assert arguments[1:]==['--download-dir',str(cold_directory)]
        assert '--home'in command and command[command.index('--home')+1]==str(cold_home)
        assert command[0]==sys.executable and command[1:6]==['-I','-S','-B','-X','utf8']
        assert timeout==300
        environment=options['env']
        assert environment['HERMES_HOME']==str(cold_home)
        assert 'OPENAI_API_KEY'not in environment
        cold_calls.append({'module':module,'voice':arguments[0],'timeout':timeout,'profile_bound':True,'launch_credential_removed':True})
        if cold_mode=='timeout':
            (cold_directory/(arguments[0]+'.onnx')).write_bytes(b'original partial file before timeout')
            raise local.subprocess.TimeoutExpired(command,timeout)
        if cold_mode=='refusal':
            return local.subprocess.CompletedProcess(command,7,stdout='',stderr='x'*1000)
        if cold_mode=='success':
            write_model(cold_directory,arguments[0])
        return local.subprocess.CompletedProcess(command,0,stdout='',stderr='')
    local._run_helper=observe_child
    try:
        cold_voice='en_US-cold-medium'
        assert local._resolve_piper_voice_path(cold_voice,cold_directory)==str(cold_directory/(cold_voice+'.onnx'))
        assert local._resolve_piper_voice_path(cold_voice,cold_directory)==str(cold_directory/(cold_voice+'.onnx'))
        assert len(cold_calls)==1
        cold_mode='timeout';requested='en_US-timeout-medium'
        try:local._resolve_piper_voice_path(requested,cold_directory)
        except RuntimeError as error:assert str(error)==f"Piper voice download timed out after 300s for '{requested}'"
        else:raise AssertionError('Original total download timeout was lost')
        assert(cold_directory/(requested+'.onnx')).read_bytes()==b'original partial file before timeout'
        assert not(cold_directory/(requested+'.onnx.json')).exists()
        cold_mode='refusal';requested='en_US-refused-medium'
        try:local._resolve_piper_voice_path(requested,cold_directory)
        except RuntimeError as error:assert str(error)==f"Piper voice download failed for '{requested}': "+'x'*400
        else:raise AssertionError('Original download refusal was lost')
        cold_mode='missing';requested='en_US-missing-medium'
        try:local._resolve_piper_voice_path(requested,cold_directory)
        except RuntimeError as error:assert 'completed but'in str(error)and str(cold_directory/(requested+'.onnx'))in str(error)
        else:raise AssertionError('Successful helper without model was accepted')
        assert len(cold_calls)==4
    finally:
        local._run_helper=previous_helper
        if previous_key is None:os.environ.pop('OPENAI_API_KEY',None)
        else:os.environ['OPENAI_API_KEY']=previous_key
        reset_secret_scope(cold_secret)
        reset_hermes_home_override(home_token)
    assert all(name not in sys.modules for name in ('piper','numpy','onnxruntime','espeakbridge','torch','soundfile'))
    assert not[record for record in guard.events if not record.get('recoverable_capability_probe')],guard.events
    print(json.dumps({'registry':True,'native_pcm_cases':len(outputs),'outputs':outputs,'typed_speaker_cases':3,'cache_lru':True,'leases':True,'profile_directories':True,'cuda_refusal':True,'owned_downloader':True,'downloader_cases':cold_calls,'blocked_events':guard.events}),flush=True)
finally:
    reset_secret_scope(token)
'''


class PiperTTSConsumerTests(BundledSourceTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from standalone.build import INTERNAL_MODULES
        module = 'standalone.local_piper_download_voices'
        assert module in INTERNAL_MODULES
        cls.context['manifest']['internal_modules'].append(module)

    def test_original_registry_real_graph_native_pcm_profile_lru_warm_and_release(self):
        with tempfile.TemporaryDirectory(prefix='piper-consumer-', dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ('home', 'work', 'tmp'):
                (scratch / name).mkdir()
            payload = write_piper_consumer_payload(scratch)
            environment = support._load_helper('agent_probe_worker').fixture_environment(scratch)
            process = subprocess.run(
                [sys.executable, '-I', '-S', '-B', '-X', 'utf8', '-c', PIPER_CONSUMER_WORKER,
                 json.dumps(self.context), str(scratch), str(payload)],
                env=environment, cwd=scratch / 'work', capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            report = json.loads(process.stdout)
            self.assertEqual(report['native_pcm_cases'], 8)
            self.assertEqual(report['typed_speaker_cases'], 3)
            self.assertEqual(len(report['downloader_cases']), 4)
            for field in ('registry', 'cache_lru', 'leases', 'profile_directories', 'cuda_refusal', 'owned_downloader'):
                self.assertTrue(report[field], field)


if __name__ == '__main__':
    unittest.main(verbosity=2)
