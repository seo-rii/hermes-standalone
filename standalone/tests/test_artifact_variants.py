"""Opt-in checks of actual copied full and lite generated distributions."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from standalone.agent_probe_worker import fixture_environment
from standalone.tests import test_artifact


LITE_CONSUMER = r'''
import importlib.abc, json, pathlib, runpy, sys
artifact, cache, scratch = sys.argv[1:]
bundle = runpy.run_path(artifact, run_name='owned_lite_variant_acceptance')
archive, manifest, manifest_bytes, files = bundle['_read_archive']()
cache = bundle['_cache_directory'](cache)
try:
    root = bundle['_extract'](archive, files, manifest_bytes, cache)
finally:
    archive.close()
assert manifest['build_variant']['name'] == 'lite'
paths = set(files)
excluded_prefixes = (
    'app/standalone/assets/piper/',
    'app/standalone/local_piper_japanese/data/',
    'app/standalone/piper_chinese_data/',
)
assert not any(name.startswith(excluded_prefixes) for name in paths)
for prefix in excluded_prefixes:
    assert not (root / prefix).exists(), prefix
thai = {name for name in paths if name.startswith('vendor/hermes_tltk/')}
assert thai and all(name.endswith('.py') for name in thai), thai
assert 'vendor/hermes_tltk/__init__.py' in thai
discovery_prefix = 'vendor/googleapiclient/discovery_cache/documents/'
chat_path = discovery_prefix + 'chat.v1.json'
assert {name for name in paths if name.startswith(discovery_prefix)} == {chat_path}
source, library = root / 'app', root / 'vendor'
sys.path[:0] = [str(source), str(library)]
from standalone.runtime import initialize_bundle
initialize_bundle({
    'artifact': artifact, 'root': str(root), 'cache_dir': str(cache),
    'manifest': manifest, 'bundle_sha256': bundle['_BUNDLE_INFO']['sha256'],
})
from standalone.agent_probe_worker import ProbeGuard
guard = ProbeGuard(source, library, pathlib.Path(scratch))
sys.addaudithook(guard)
class NoExcludedPiper(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'standalone.local_piper_voice':
            raise AssertionError('Lite imported a local Piper model before checking availability')
        if fullname.split('.', 1)[0] in {'piper', 'numpy', 'onnxruntime', 'torch'}:
            raise AssertionError('Lite imported a native/model SDK: ' + fullname)
sys.meta_path.insert(0, NoExcludedPiper())
from googleapiclient.discovery_cache import get_static_doc
chat = json.loads(get_static_doc('chat', 'v1'))
assert chat['id'] == 'chat:v1' and 'spaces' in chat['resources']
assert get_static_doc('compute', 'v1') is None
assert not guard.events, guard.events
from standalone.variant import UnavailableFeatureError, current_variant
from tools import tts_tool as tts
assert current_variant()['name'] == 'lite'
assert not tts._check_piper_available()
try:
    tts._import_piper()
except UnavailableFeatureError as exc:
    assert 'lite build' in str(exc) and 'full build' in str(exc), str(exc)
else:
    raise AssertionError('Lite made the excluded local Piper engine available')
provider, error = tts._select_builtin_engine('piper')
error = json.loads(error)
assert provider == 'piper' and not error['success']
assert 'full build' in error['error'] and 'cloud TTS' in error['error']
assert tts._select_builtin_engine('openai') == ('openai', None)
assert tts._select_builtin_engine('gemini') == ('gemini', None)
assert not guard.events, guard.events
print(json.dumps({
    'variant': 'lite', 'verified_files': len(files),
    'local_piper_unavailable': True, 'google_chat_static_document': chat['id'],
    'cloud_tts_dispatch_available': True, 'blocked_events': guard.events,
}), flush=True)
'''


class ArtifactVariantTests(unittest.TestCase):
    def setUp(self):
        test_artifact.ArtifactTests.setUp(self)
        for name in ('home', 'work', 'tmp'):
            (self.directory / name).mkdir()
        self.environment = fixture_environment(self.directory)

    def test_copied_doctor_reports_expected_variant_and_capabilities(self):
        result = test_artifact.ArtifactTests.run_artifact(self, ['doctor', '--json'])
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        expected = test_artifact.ARTIFACT_VARIANT
        self.assertEqual(report['build_variant']['name'], expected)
        self.assertFalse(report['full_parity_verified'])
        features = {item['id']: item for item in report['features']}
        if expected == 'lite':
            self.assertEqual(set(report['build_variant']['excluded_features']), {
                'local-piper-model', 'google-static-discovery',
            })
            self.assertEqual(features['local-piper-model']['status'], 'excluded')
            self.assertEqual(features['google-static-discovery']['status'], 'partial')
            self.assertIn('full build', report['excluded_feature_details']['local-piper-model'])
            self.assertIn('Google Chat', report['excluded_feature_details']['google-static-discovery'])
        else:
            self.assertEqual(report['build_variant']['excluded_features'], [])
            self.assertNotEqual(features['local-piper-model']['status'], 'excluded')
        self.assertFalse(Path(self.environment['HERMES_HOME']).exists())

    def test_copied_lite_omits_resources_and_preserves_offline_google_chat_and_cloud_tts(self):
        if test_artifact.ARTIFACT_VARIANT != 'lite':
            self.skipTest('This acceptance case checks intentionally omitted lite resources')
        if sys.version_info < (3, 14):
            self.skipTest('The original TTS tool consumer requires Python 3.14')
        result = subprocess.run(
            [sys.executable, '-I', '-S', '-B', '-X', 'utf8', '-c', LITE_CONSUMER,
             str(self.artifact), str(self.cache), str(self.directory)],
            cwd=self.directory / 'work', env=self.environment, capture_output=True,
            text=True, encoding='utf8', timeout=test_artifact.ARTIFACT_TIMEOUT,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report['variant'], 'lite')
        self.assertGreater(report['verified_files'], 0)
        self.assertTrue(report['local_piper_unavailable'])
        self.assertEqual(report['google_chat_static_document'], 'chat:v1')
        self.assertTrue(report['cloud_tts_dispatch_available'])
        self.assertEqual(report['blocked_events'], [])
        self.assertFalse(Path(self.environment['HERMES_HOME']).exists())


if __name__ == '__main__':
    unittest.main()
