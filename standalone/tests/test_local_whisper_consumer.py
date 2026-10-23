"""The original STT selector and model cache use the qualified CPU engine.

The small CT2 model fixture checks real weight loading and cache ownership;
trained speech inference has separate independent native reference vectors.
"""

import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib

from standalone.tests.test_entrypoints import BundledSourceTestCase, support


_WORKER = r'''
import gc, importlib.abc, json, os, pathlib, sys, weakref
context=json.loads(sys.argv[1]);scratch=pathlib.Path(sys.argv[2])
source=pathlib.Path(context['root'])/'app'
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0] in {'faster_whisper','numpy','av','tokenizers','ctranslate2','onnxruntime','torch'}:
            raise AssertionError('STT consumer attempted to import '+fullname)
sys.meta_path.insert(0,NoNative())
def guard(event,args):
    if event.startswith('socket.') and event!='socket.__new__':
        raise AssertionError('Cached STT consumer attempted network access')
    if event in ('subprocess.Popen','os.system'):
        raise AssertionError('Cached STT consumer attempted an installer or native process')
sys.addaudithook(guard)
import pm
from pm.extras import missing
from tools import transcription_tools as stt, transcription_local as local
from standalone.local_whisper import WhisperModel
from standalone.local_whisper_engine import WhisperEngine
assert stt._HAS_FASTER_WHISPER and stt._detect_local_backend()=='local'
assert stt._get_provider({'provider':'local'})=='local'
for extra in ('voice','stt-whisper'):
    assert pm.available(extra) and not missing(extra)
    pm.ensure_import(extra)
assert local._try_lazy_install_stt()
path=str(source/'standalone/test-fixtures/whisper-consumer-model')
model=stt._get_or_load_local_model(path,{'device':'cpu','compute_type':'float32'})
assert type(model) is WhisperModel and type(model.model) is WhisperEngine
assert model.model.device=='cpu' and model.model.compute_type=='float32'
assert stt._get_or_load_local_model(path,{'device':'cpu','compute_type':'float32'}) is model
weak=weakref.ref(model.model)
stt._unload_local_model()
assert stt._local_model is None and stt._local_model_name is None
assert weak() is model.model, 'An active transcription lost its strong model reference'
del model;gc.collect()
assert weak() is None, 'The idle unload retained the model'
assert not any(name in sys.modules for name in ('faster_whisper','numpy','av','tokenizers','ctranslate2','onnxruntime','torch'))
print(json.dumps({'available':True,'actual_factory':True,'cached':True,'unloaded':True}),flush=True)
os._exit(0)
'''


class OriginalWhisperConsumerTests(BundledSourceTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fixtures=Path(__file__).with_name('fixtures')
        native=json.loads((fixtures/'local_whisper_engine.json').read_text())['synthetic_model']
        wrapper=json.loads((fixtures/'local_whisper_segments.json').read_text())
        segments=json.loads(zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True)))
        target=cls.bundle/'app/standalone/test-fixtures/whisper-consumer-model'
        target.mkdir(parents=True)
        (target/'model.bin').write_bytes(zlib.decompress(base64.b64decode(native['model_zlib_base64'],validate=True)))
        (target/'config.json').write_text(json.dumps(native['config']))
        (target/'vocabulary.json').write_text(json.dumps(native['vocabulary']))
        (target/'tokenizer.json').write_text(segments['tokenizers']['tiny.en'])

    def test_original_selector_factory_and_cache_without_native_sdk_or_installation(self):
        with tempfile.TemporaryDirectory(prefix='whisper-consumer-',dir=self.directory) as temporary:
            scratch=Path(temporary)
            for name in ('home','work','tmp'):
                (scratch/name).mkdir()
            profile=scratch/'home/hermes';profile.mkdir()
            (profile/'config.yaml').write_text(json.dumps({'stt':{'provider':'local','local':{'unload_after_idle_seconds':0}}}))
            environment=support._load_helper('agent_probe_worker').fixture_environment(scratch)
            process=subprocess.run([sys.executable,'-I','-S','-B','-c',_WORKER,json.dumps(self.context),str(scratch)],
                                   env=environment,cwd=scratch/'work',capture_output=True,text=True,timeout=35)
            self.assertEqual(process.returncode,0,process.stderr+process.stdout)
            self.assertEqual(json.loads(process.stdout),dict(available=True,actual_factory=True,cached=True,unloaded=True))


if __name__=='__main__':
    unittest.main()
