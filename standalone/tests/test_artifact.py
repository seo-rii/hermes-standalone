"""Opt-in integration tests of the actual generated distribution."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ARTIFACT: Path | None = None
# Cold startup verifies the full embedded model/dictionary archive in the CLI
# and each real managed child. The bound includes this measured startup cost.
ARTIFACT_TIMEOUT = 300


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        if ARTIFACT is None:
            self.skipTest("Pass --artifact to standalone/check.py to exercise the generated file")
        self.temporary = tempfile.TemporaryDirectory(prefix="hermes standalone integration ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.artifact = self.directory / "relocated hermes.py"
        shutil.copyfile(ARTIFACT, self.artifact)
        self.environment = os.environ.copy()
        self.environment.pop("PYTHONPATH", None)
        self.environment.pop("PYTHONHOME", None)
        self.environment["HERMES_HOME"] = str(self.directory / "untouched-profile")
        self.cache = self.directory / "private cache"

    def run_artifact(self, args, *, isolated=True):
        command = [sys.executable]
        if isolated:
            command += ["-I", "-S", "-B"]
        command += [str(self.artifact), "--cache-dir", str(self.cache), *args]
        return subprocess.run(command, cwd=self.directory, env=self.environment,
                              capture_output=True, text=True, encoding="utf-8", timeout=ARTIFACT_TIMEOUT)

    def test_relocated_distribution_exercises_actual_upstream_components(self):
        result = self.run_artifact(["selftest", "--json"])
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"], report)
        self.assertFalse(report["agent_parity_verified"])
        self.assertTrue(any(item["name"] == "internal-worker" for item in report["checks"]))
        self.assertFalse(Path(self.environment["HERMES_HOME"]).exists())

    def test_normal_launch_reenters_isolated_python_and_reports_supported_runtime(self):
        result = self.run_artifact(["doctor", "--json", "--require-agent"], isolated=False)
        self.assertEqual(result.returncode, 0 if sys.version_info >= (3, 14) else 2, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["isolated"])
        self.assertTrue(report["site_disabled"])
        self.assertEqual(report["agent_ready"], sys.version_info >= (3, 14))
        self.assertFalse(report["full_parity_verified"])
        self.assertGreater(report["embedded_distributions"], 0)
        self.assertFalse(Path(self.environment["HERMES_HOME"]).exists())

    def test_dependency_metadata_and_license_survive_relocation(self):
        result = self.run_artifact(["licenses", "httpx"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Encode OSS Ltd", result.stdout)
        self.assertIn("Redistribution and use in source and binary forms", result.stdout)
        self.assertIn("httpx", result.stdout.lower())

    def test_original_agent_imports_without_pm_or_native_packages(self):
        result = self.run_artifact(["probe-agent", "--json"])
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "passed", report)
        self.assertTrue(report["agent_imported"])
        self.assertTrue(report["artifact_integrity_verified"])
        self.assertEqual(report["runtime_binding"], "bundle")
        self.assertFalse(report["agent_constructed"])
        self.assertFalse(report["live_provider_verified"])
        self.assertFalse(Path(self.environment["HERMES_HOME"]).exists())

    @unittest.skipIf(sys.version_info < (3, 14), "Original agent requires Python 3.14")
    def test_original_constructor_uses_qualified_transport_without_installed_sdk(self):
        result = self.run_artifact(["probe-agent", "--stage", "construct", "--json"])
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertTrue(report["agent_imported"], report)
        self.assertTrue(report["original_constructor_verified"])
        self.assertFalse(report["live_provider_verified"])
        self.assertNotIn("missing_module", report)
        self.assertFalse(Path(self.environment["HERMES_HOME"]).exists())

    def test_original_agent_turn_with_local_fixture_client(self):
        result = self.run_artifact(["probe-agent", "--stage", "fixture-turn", "--json"])
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertTrue(report["agent_imported"], report)
        self.assertTrue(report["fixture_turn_verified"], report)
        self.assertTrue(report["fixture_tool_result_verified"], report)
        self.assertEqual(report["fixture_model_calls"], 2)
        self.assertFalse(report["original_constructor_verified"])
        self.assertFalse(report["live_provider_verified"])
        self.assertFalse(Path(self.environment["HERMES_HOME"]).exists())

    def test_restored_original_piper_models_run_without_native_dependencies(self):
        driver = r'''
import hashlib, importlib.abc, json, pathlib, runpy, sys
artifact, cache = sys.argv[1:]
bundle = runpy.run_path(artifact, run_name='owned_model_acceptance')
archive, manifest, manifest_bytes, files = bundle['_read_archive']()
try:
    root = bundle['_extract'](archive, files, manifest_bytes, bundle['_cache_directory'](cache))
finally:
    archive.close()
expected = {
    'hebrew/nakdimon.onnx': '9ff491dcc7d66392019d427a98b97d5de10c0d721628ae740858174ae22b190e',
    'tashkeel/model.onnx': '52c40d42ca094ff9ff07a091d0a1eafd7baa5cf2388e60a9cc89bacd8e5e955f',
}
for name, digest in expected.items():
    path = root / 'app/standalone/assets/piper' / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'numpy', 'onnx', 'onnxruntime', 'piper', 'tensorflow', 'torch', 'ctypes', 'subprocess'}:
            raise AssertionError('Bundled model attempted native dependency ' + fullname)
sys.meta_path.insert(0, NoNative())
sys.path[:0] = [str(root / 'app'), str(root / 'vendor')]
from standalone import local_piper_arabic, local_piper_hebrew
for module in (local_piper_arabic, local_piper_hebrew):
    assert pathlib.Path(module.__file__).is_relative_to(root)
arabic = local_piper_arabic.TashkeelDiacritizer()('مرحبا')
hebrew = local_piper_hebrew.NakdimonDiacritizer()('שלום')
# Expected strings were independently captured from original Piper/ORT.
assert arabic == 'مُرَحِبًا', arabic
assert hebrew == 'שְׁלוֹם', hebrew
print(json.dumps({'original_models': len(expected), 'arabic': arabic, 'hebrew': hebrew}))
'''
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', driver,
                                 str(self.artifact), str(self.cache)], cwd=self.directory,
                                env=self.environment, capture_output=True, text=True,
                                encoding='utf-8', timeout=ARTIFACT_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)['original_models'], 2)
        self.assertFalse(Path(self.environment['HERMES_HOME']).exists())

    @unittest.skipIf(sys.version_info < (3, 14), "Original TTS tool requires Python 3.14")
    def test_copied_piper_tts_registry_cache_profiles_and_owned_model_downloader(self):
        from standalone.tests.test_piper_tts_consumer import (
            PIPER_CONSUMER_WORKER, write_piper_consumer_payload,
        )
        from standalone.tests.test_entrypoints import support
        scratch=self.directory/'piper consumer';scratch.mkdir()
        for name in ('home','work','tmp'):(scratch/name).mkdir()
        payload=write_piper_consumer_payload(scratch)
        worker=scratch/'consumer.py';worker.write_text(PIPER_CONSUMER_WORKER)
        driver=r'''
import json,pathlib,runpy,sys
artifact,cache,scratch,payload,worker=sys.argv[1:]
bundle=runpy.run_path(artifact,run_name='owned_piper_tts_acceptance')
archive,manifest,manifest_bytes,files=bundle['_read_archive']()
cache= bundle['_cache_directory'](cache)
try:root=bundle['_extract'](archive,files,manifest_bytes,cache)
finally:archive.close()
context={'artifact':artifact,'root':str(root),'cache_dir':str(cache),'manifest':manifest,'bundle_sha256':bundle['_BUNDLE_INFO']['sha256']}
assert 'standalone.local_piper_download_voices'in manifest['internal_modules']
code=pathlib.Path(worker).read_text()
exec(compile(code,worker,'exec'),{'PIPER_CONTEXT':context,'PIPER_SCRATCH':scratch,'PIPER_FIXTURE':payload})
'''
        environment=support._load_helper('agent_probe_worker').fixture_environment(scratch)
        process=subprocess.run([sys.executable,'-I','-S','-B','-X','utf8','-c',driver,
                                str(self.artifact),str(self.cache),str(scratch),str(payload),str(worker)],
                               cwd=scratch/'work',env=environment,capture_output=True,text=True,
                               encoding='utf8',timeout=ARTIFACT_TIMEOUT)
        self.assertEqual(process.returncode,0,process.stderr+process.stdout)
        report=json.loads(process.stdout);self.assertEqual(report['native_pcm_cases'],8)
        self.assertEqual(len(report['downloader_cases']),4)
        for field in ('registry','cache_lru','leases','profile_directories','cuda_refusal','owned_downloader'):
            self.assertTrue(report[field],field)
        rejected=self.run_artifact(['--internal-module','standalone.local_piper_download_voices',
                                    'invalid-hermes-fixture','--download-dir',str(scratch/'downloads')])
        self.assertNotEqual(rejected.returncode,0)
        self.assertIn('did not match pattern',rejected.stderr)
        self.assertFalse(Path(self.environment['HERMES_HOME']).exists())

    def test_copied_thai_original_resources_numbers_and_piper_voice(self):
        import zlib
        fixture=json.loads(zlib.decompress((Path(__file__).parent/'fixtures/local_piper_thai.json.zlib').read_bytes()))
        cases=[next(case for case in fixture['piper']['cases']
                    if case['text']==text and case['expand_numbers']==expand)
               for expand in (False,True)
               for text in ('สวัสดี','1234','๑๒๓๔','฿1,250.50','-3.14%')]
        scratch=self.directory/'thai consumer';scratch.mkdir()
        payload=scratch/'original-observations.json'
        payload.write_text(json.dumps({'cases':cases,'resources':fixture['consumed_resources']},ensure_ascii=False))
        driver=r'''
import hashlib,importlib.abc,json,pathlib,runpy,sys
artifact,cache,scratch,payload=sys.argv[1:]
bundle=runpy.run_path(artifact,run_name='owned_thai_acceptance')
archive,manifest,manifest_bytes,files=bundle['_read_archive']()
try:root=bundle['_extract'](archive,files,manifest_bytes,bundle['_cache_directory'](cache))
finally:archive.close()
source=root/'app';library=root/'vendor';sys.path[:0]=[str(source),str(library)]
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(source,library,pathlib.Path(scratch));sys.addaudithook(guard)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'numpy','onnxruntime','torch','ctypes','subprocess','piper','tltk','pandas','nltk','sklearn','sklearn_crfsuite','gensim'}:
            raise AssertionError('Unexpected Thai native/SDK dependency '+fullname)
sys.meta_path.insert(0,NoNative())
def no_external(event,args):
    if event.startswith('socket.')or event in {'subprocess.Popen','os.system','ctypes.dlopen','ctypes.dlsym'}:
        raise AssertionError('Thai consumer attempted external execution')
sys.addaudithook(no_external)
from standalone.local_piper_thai import ThaiPhonemizer
from standalone.local_piper_voice import PiperVoice
from standalone.local_piper_config import PiperConfig,PhonemeType
from standalone.local_piper_phoneme_ids import DEFAULT_PHONEME_ID_MAP
fixture=json.loads(pathlib.Path(payload).read_text())
for name,expected in fixture['resources'].items():
    raw=(library/'hermes_tltk'/name).read_bytes()
    assert len(raw)==expected['bytes']and hashlib.sha256(raw).hexdigest()==expected['sha256'],name
instances={expand:ThaiPhonemizer(expand_numbers=expand)for expand in (False,True)}
config=PiperConfig(256,1,22050,'th',DEFAULT_PHONEME_ID_MAP,PhonemeType.THAI)
voice=PiperVoice(None,config);cached=None
for case in fixture['cases']:
    instance=instances[case['expand_numbers']]
    assert instance._normalize(case['text'])==case['normalized']
    assert instance.phonemize(case['text'])==case['output']
    if case['expand_numbers']:
        assert voice.phonemize(case['text'])==case['output']
        if cached is not None:assert voice._thai_phonemizer is cached
        cached=voice._thai_phonemizer
assert isinstance(cached,ThaiPhonemizer)
assert not guard.events,guard.events
print(json.dumps({'cases':len(fixture['cases']),'resources':len(fixture['resources']),'cached':True}))
'''
        environment={'HOME':str(scratch/'home'),'USERPROFILE':str(scratch/'home'),'PATH':'',
                     'HERMES_HOME':str(scratch/'profile'),'PYTHONDONTWRITEBYTECODE':'1'}
        for name in ('SYSTEMROOT','WINDIR'):
            if name in os.environ:environment[name]=os.environ[name]
        result=subprocess.run([sys.executable,'-I','-S','-B','-X','utf8','-c',driver,
                               str(self.artifact),str(self.cache),str(scratch),str(payload)],
                              cwd=scratch,env=environment,capture_output=True,text=True,encoding='utf8',timeout=180)
        self.assertEqual(result.returncode,0,result.stderr+result.stdout)
        self.assertEqual(json.loads(result.stdout),{'cases':10,'resources':4,'cached':True})
        self.assertFalse(Path(self.environment['HERMES_HOME']).exists())

    def test_copied_chinese_japanese_original_resources_profiles_and_voice_pcm(self):
        from standalone.agent_probe_worker import fixture_environment
        from standalone.tests.artifact_piper_languages import WORKER, write_payload
        scratch = self.directory / 'piper languages'
        scratch.mkdir()
        for name in ('home', 'work', 'tmp'):
            (scratch / name).mkdir()
        payload = write_payload(scratch)
        result = subprocess.run(
            [sys.executable, '-I', '-S', '-B', '-X', 'utf8', '-c', WORKER,
             str(self.artifact), str(self.cache), str(scratch), str(payload)],
            cwd=scratch / 'work', env=fixture_environment(scratch), capture_output=True,
            text=True, encoding='utf8', timeout=ARTIFACT_TIMEOUT,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertTrue(report['dictionary_restored'])
        self.assertEqual(report['blocked_events'], [])
        for name, resources, custom in (('chinese', 30, 4), ('japanese', 86, 6)):
            self.assertEqual(report['languages'][name], {
                'resources': resources, 'voice_cases': 2, 'custom_ids': custom,
                'real_auxiliary_backend': True, 'actual_small_voice_pcm_wav': True,
                'separate_profiles': True,
            })
        self.assertFalse(Path(self.environment['HERMES_HOME']).exists())

    def test_copied_vorbis_matroska_speech_files_decode_to_original_frontend_pcm(self):
        import base64
        import hashlib
        import zlib
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_whisper_audio.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        self.assertEqual(hashlib.sha256(raw).hexdigest(),wrapper['data_sha256'])
        cases=[case for case in json.loads(raw)['consumers']if case['file'].endswith('.ogg')]
        self.assertEqual(len(cases),2)
        fixtures=Path(__file__).parent/'fixtures'
        remux=json.loads(zlib.decompress((fixtures/'local_whisper_matroska_remux.json.zlib').read_bytes()))
        profiles=json.loads(zlib.decompress((fixtures/'local_whisper_matroska_remux_profiles.json.zlib').read_bytes()))
        profiles={case['name']:case for case in profiles['cases']}
        for name in ('jfk-vorbis-16000-1ch.ogg-webm','speech48k.opus-webm'):
            case=next(case for case in remux['cases']if case['name']==name)
            expected=next(row for row in profiles[name]['outputs']if row['rate']==16000 and not row['split'])
            cases.append({'input':case['data'],'pcm':expected['codec_reference'][0]})
        payload=self.directory/'original-container-speech.json';payload.write_text(json.dumps(cases))
        driver=r'''
import base64,importlib.abc,io,json,pathlib,runpy,struct,sys
artifact,cache,payload=sys.argv[1:]
bundle=runpy.run_path(artifact,run_name='owned_audio_container_acceptance')
archive,manifest,manifest_bytes,files=bundle['_read_archive']()
try:root=bundle['_extract'](archive,files,manifest_bytes,bundle['_cache_directory'](cache))
finally:archive.close()
source=root/'app';library=root/'vendor';sys.path[:0]=[str(source),str(library)]
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(source,library,pathlib.Path(payload).parent);sys.addaudithook(guard)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'av','numpy','onnxruntime','torch','ctypes','subprocess','soundfile','audioread','pydub'}:
            raise AssertionError('Unexpected audio container native/process import '+fullname)
sys.meta_path.insert(0,NoNative())
from standalone.local_whisper_audio import decode_audio
cases=json.loads(pathlib.Path(payload).read_text())
for case in cases:
    data=base64.b64decode(case['input'],validate=True)
    original=base64.b64decode(case['pcm'],validate=True)
    expected=[row[0]for row in struct.iter_unpack('<h',original)]
    stream=io.BytesIO(data);actual=decode_audio(stream)
    assert not stream.closed and actual.typecode=='f'
    assert len(actual)==len(expected)==32000
    assert max(abs(round(value*32768)-target)for value,target in zip(actual,expected))<=1
assert not guard.events,guard.events
print(json.dumps({'speech_files':len(cases),'samples_each':32000,'max_pcm_error_lsb':1}))
'''
        result=subprocess.run([sys.executable,'-I','-S','-B','-X','utf8','-c',driver,
                               str(self.artifact),str(self.cache),str(payload)],
                              cwd=self.directory,env=self.environment,capture_output=True,text=True,
                              encoding='utf8',timeout=ARTIFACT_TIMEOUT)
        self.assertEqual(result.returncode,0,result.stderr+result.stdout)
        self.assertEqual(json.loads(result.stdout),{'speech_files':4,'samples_each':32000,'max_pcm_error_lsb':1})
        self.assertFalse(Path(self.environment['HERMES_HOME']).exists())


if __name__ == "__main__":
    unittest.main()
