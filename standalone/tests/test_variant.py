"""Artifact variants cannot be changed by profiles or pretend missing SDKs are installable."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from standalone.cli import _doctor, _print_doctor
from standalone.variant import manifest_variant, normalize_variant
from standalone.tests.test_entrypoints import PROJECT, STANDALONE, support
from standalone.tests import test_runtime as runtime_tests


LITE = {
    "name": "lite",
    "excluded_features": ["local-piper-model", "google-static-discovery"],
    "generated_files": ["app/standalone/capabilities.json"],
}


class VariantMetadataTests(unittest.TestCase):
    def test_legacy_and_explicit_full_manifests_have_no_exclusions(self):
        for manifest in ({}, {"build_variant": {"name": "full", "excluded_features": []}}):
            variant = manifest_variant(manifest)
            self.assertEqual(variant["name"], "full")
            self.assertEqual(variant["excluded_features"], ())

    def test_normalization_copies_and_freezes_nested_metadata(self):
        original = json.loads(json.dumps(LITE))
        variant = normalize_variant(original)
        original["name"] = "full"
        original["excluded_features"].clear()
        original["generated_files"].clear()
        self.assertEqual(variant["name"], "lite")
        self.assertEqual(set(variant["excluded_features"]), set(LITE["excluded_features"]))
        self.assertEqual(variant["generated_files"], ("app/standalone/capabilities.json",))
        with self.assertRaises(TypeError):
            variant["name"] = "full"

    def test_invalid_variants_do_not_silently_enable_features(self):
        for variant in (
            None, [], {}, {"name": "lite"}, {"name": "unknown", "excluded_features": []},
            {"name": "full", "excluded_features": ["local-piper-model"]},
            {"name": "lite", "excluded_features": []},
            {**LITE, "excluded_features": ["local-piper-model", "local-piper-model"]},
            {**LITE, "excluded_features": ["local-piper-model", []]},
            {**LITE, "generated_files": ["../config.yaml"]},
            {**LITE, "unexpected": True},
        ):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                normalize_variant(variant)

    def test_doctor_reports_lite_constraints_and_retained_google_chat(self):
        capabilities = json.loads((STANDALONE / "capabilities.json").read_text())
        manifest = {
            "project": "hermes-standalone", "version": "fixture", "upstream": {},
            "runtime": {"hermes_python": "3.14"}, "capabilities": capabilities,
            "distributions": [], "files": [], "build_variant": LITE,
        }
        context = {"manifest": manifest, "bundle_sha256": "a" * 64}
        report = _doctor(context)
        self.assertEqual(report["build_variant"]["name"], "lite")
        self.assertIn("Google Chat", report["excluded_feature_details"]["google-static-discovery"])
        self.assertFalse(report["full_parity_verified"])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            _print_doctor(report)
        self.assertIn("Build: lite", stream.getvalue())
        self.assertIn("cloud TTS", stream.getvalue())
        self.assertIn("Google Chat", stream.getvalue())
        self.assertEqual(manifest["build_variant"], LITE)


class VariantBindingTests(runtime_tests.RuntimeBindingTests):
    """The base fixture copies the actual bootstrap startup seam into an empty source tree."""

    def setUp(self):
        super().setUp()
        shutil.copyfile(STANDALONE / "variant.py", self.app / "standalone/variant.py")

    def test_lite_binding_is_immutable_and_independent_of_profile_environment(self):
        result = self.invoke("""
from standalone.runtime import current_bundle
from standalone.variant import current_variant, require_feature, UnavailableFeatureError
context['manifest']['build_variant'] = {
    'name': 'lite', 'excluded_features': ['local-piper-model', 'google-static-discovery'],
    'generated_files': ['app/standalone/capabilities.json'],
}
initialize_bundle(context)
context['manifest']['build_variant']['excluded_features'].clear()
os.environ['HERMES_HOME'] = str(pathlib.Path(context['root']).parent / 'another-profile')
os.environ['HERMES_STANDALONE_VARIANT'] = 'full'
assert current_variant()['name'] == 'lite'
assert 'local-piper-model' in current_variant()['excluded_features']
try:
    current_bundle()['manifest']['build_variant']['name'] = 'full'
except TypeError:
    pass
else:
    raise AssertionError('Variant metadata can be changed')
try:
    require_feature('local-piper-model')
except UnavailableFeatureError as exc:
    assert 'full build' in str(exc) and 'cloud TTS' in str(exc)
else:
    raise AssertionError('Excluded Piper was available')
context['manifest']['build_variant'] = {'name': 'full', 'excluded_features': []}
try:
    initialize_bundle(context)
except ValueError:
    print('immutable lite binding passed')
else:
    raise AssertionError('An initialized artifact changed variants')
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("immutable lite binding passed", result.stdout)

    def test_unbound_and_legacy_processes_ignore_variant_environment_flags(self):
        result = self.invoke("""
from standalone.variant import current_variant, require_feature
os.environ['HERMES_STANDALONE_VARIANT'] = 'lite'
assert current_variant()['name'] == 'full'
require_feature('local-piper-model')
initialize_bundle(context)
assert current_variant()['name'] == 'full'
require_feature('local-piper-model')
print('legacy full binding passed')
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("legacy full binding passed", result.stdout)


_LITE_TTS_WORKER = r'''
import importlib.abc, json, os, pathlib, sys
context=json.loads(sys.argv[1])
scratch=pathlib.Path(sys.argv[2])
source=pathlib.Path(context['root'])/'app'
vendor=source.parent/'vendor'
sys.path[:0]=[str(source),str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(source,vendor,scratch)
sys.addaudithook(guard)
class NoExcludedPiper(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname == 'standalone.local_piper_voice':
            raise AssertionError('Piper code was imported before the lite feature guard')
sys.meta_path.insert(0,NoExcludedPiper())
from agent.sdk_boundary import bundled
from standalone.variant import UnavailableFeatureError
from tools import tts_tool as tts, tts_tool_local as local
from pm.extras import available, ensure_import
from pm.package import InstallError
assert bundled()
assert not (source/'standalone/assets/piper').exists()
assert not (source/'standalone/local_piper_japanese/data').exists()
assert not (source/'standalone/piper_chinese_data').exists()
assert not tts._check_piper_available()
assert not available('piper')
provider,error=tts._select_builtin_engine('piper')
error=json.loads(error)
assert provider == 'piper' and not error['success']
assert 'lite build' in error['error'] and 'full build' in error['error']
assert 'not installed' not in error['error']
for operation in (
    tts._import_piper,
    lambda: local._load_piper_voice_for_config({'piper': {'voices_dir': str(scratch/'no-download')}}),
):
    try: operation()
    except UnavailableFeatureError as exc: assert 'lite build' in str(exc)
    else: raise AssertionError('Excluded Piper reached a model or downloader')
assert not (scratch/'no-download').exists()
try: ensure_import('piper')
except InstallError as exc: assert 'lite build' in str(exc) and 'full build' in str(exc)
else: raise AssertionError('Excluded Piper attempted runtime installation')
assert tts._select_builtin_engine('openai') == ('openai',None)
assert tts._select_builtin_engine('gemini') == ('gemini',None)
from tools import transcription_tools as stt
assert stt._HAS_OPENAI and stt._HAS_MISTRAL
assert available('stt-whisper')
assert all(name not in sys.modules for name in ('piper','numpy','onnxruntime','torch'))
assert not guard.events, guard.events
print(json.dumps({'piper_unavailable':True,'cloud_tts_dispatch_available':True,
                  'cloud_stt_import_available':True,'blocked_events':guard.events}),flush=True)
os._exit(0)
'''


class LiteTTSConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dependencies = json.loads((STANDALONE / "dependencies.json").read_text())
        downloads = STANDALONE / ".cache/downloads"
        if any(not (downloads / package["filename"]).is_file() for package in dependencies["packages"]):
            raise unittest.SkipTest("Reviewed cached dependencies are needed for the actual TTS consumer")
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-lite-consumer-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        builder, vendor = support._load_helper("build"), support._load_helper("vendor")
        sources = builder.collect_sources(PROJECT, variant="lite")
        digest = hashlib.sha256()
        for name, content in sorted(sources.items()):
            digest.update(name.encode() + b"\0" + hashlib.sha256(content).digest())
        cls.bundle = cls.directory / digest.hexdigest()
        for name, content in sources.items():
            destination = cls.bundle / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        for package in dependencies["packages"]:
            files, _ = vendor.read_wheel(package, downloads)
            for name, content in files.items():
                path = "vendor/" + name
                if not builder._include_variant_path(path, "lite"):
                    continue
                destination = cls.bundle / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        cls.context = {
            "root": str(cls.bundle), "artifact": str(cls.directory / "fixture.py"),
            "bundle_sha256": cls.bundle.name,
            "manifest": {"project": "hermes-standalone", "build_variant": LITE},
        }
        Path(cls.context["artifact"]).write_text("# source-consumer fixture\n")

    def test_original_tts_refuses_piper_before_data_access_or_download_and_keeps_cloud_dispatch(self):
        with tempfile.TemporaryDirectory(prefix="consumer-", dir=self.directory) as temporary:
            scratch = Path(temporary)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            process = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _LITE_TTS_WORKER,
                 json.dumps(self.context), str(scratch)],
                cwd=scratch / "work", env=environment, capture_output=True,
                text=True, encoding="utf-8", timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            self.assertTrue(report["piper_unavailable"])
            self.assertTrue(report["cloud_tts_dispatch_available"])
            self.assertTrue(report["cloud_stt_import_available"])
            self.assertEqual(report["blocked_events"], [])


if __name__ == "__main__":
    unittest.main()
