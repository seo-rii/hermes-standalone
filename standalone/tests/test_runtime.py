"""Exercise bundle-to-upstream startup without any external dependencies."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hermes runtime binding ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.digest = "a" * 64
        self.root = self.directory / self.digest
        self.app = self.root / "app"
        package = self.app / "standalone"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(PROJECT / "standalone/runtime.py", package / "runtime.py")
        shutil.copyfile(PROJECT / "hermes_bootstrap.py", self.app / "hermes_bootstrap.py")
        self.artifact = self.directory / "fixture.py"
        self.artifact.write_text("# fixture\n", encoding="utf-8")

    def invoke(self, body, *, isolated=True):
        context = {
            "root": str(self.root),
            "artifact": str(self.artifact),
            "bundle_sha256": self.digest,
            "manifest": {"project": "hermes-standalone"},
        }
        prefix = """
import importlib.abc, json, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
context = json.loads(sys.argv[2])
from standalone.runtime import initialize_bundle, is_active_bundle
class ForbiddenStartup(ImportError):
    pass
class NoPM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pm' or fullname.startswith('pm.'):
            raise ForbiddenStartup('PM_WAS_REACHED')
        if fullname in ('hermes_cli._early_recovery', 'hermes_cli.venv_sync'):
            raise ForbiddenStartup('UPDATER_WAS_REACHED')
sys.meta_path.insert(0, NoPM())
def offline(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.exec', 'os.system'):
        raise AssertionError('Unexpected external operation: ' + event)
sys.addaudithook(offline)
"""
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith(("HERMES_", "PYTHON")):
                environment.pop(name)
        environment["HERMES_HOME"] = str(self.directory / "profile")
        environment["HOME"] = str(self.directory / "home")
        environment["USERPROFILE"] = str(self.directory / "home")
        environment["HERMES_STANDALONE"] = "1"  # An env flag grants no startup exception.
        return subprocess.run(
            [sys.executable, *(["-I", "-S"] if isolated else []), "-B", "-X", "utf8",
             "-c", prefix + body, str(self.app), json.dumps(context)],
            env=environment, cwd=self.directory, capture_output=True,
            text=True, encoding="utf-8", timeout=30,
        )

    def test_bound_upstream_bootstrap_never_imports_pm_or_updater(self):
        result = self.invoke("""
initialize_bundle(context)
initialize_bundle(context)
assert is_active_bundle(sys.argv[1])
assert not is_active_bundle(pathlib.Path(sys.argv[1]).parent / 'other')
sys.argv = ['hermes', 'chat']
import hermes_bootstrap
assert hermes_bootstrap._standalone_bundle
assert not hermes_bootstrap._pm_repair
assert 'pm' not in sys.modules
assert 'hermes_cli.venv_sync' not in sys.modules
print('bound bootstrap passed')
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bound bootstrap passed", result.stdout)

    def test_ordinary_checkout_still_reaches_original_startup(self):
        result = self.invoke("""
assert not is_active_bundle(sys.argv[1])
try:
    import hermes_bootstrap
except ForbiddenStartup as exc:
    print(str(exc))
else:
    raise AssertionError('Ordinary startup incorrectly bypassed PM')
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PM_WAS_REACHED", result.stdout)

    def test_binding_rejects_other_source_roots_and_cache_identities(self):
        result = self.invoke("""
for key, value in (
    ('root', str(pathlib.Path(context['root']).parent / 'other')),
    ('bundle_sha256', 'b' * 64),
):
    invalid = dict(context)
    invalid[key] = value
    try:
        initialize_bundle(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid binding was accepted')
    assert not is_active_bundle(sys.argv[1])
initialize_bundle(context)
changed = dict(context)
changed['artifact'] = str(pathlib.Path(context['artifact']).with_name('other.py'))
try:
    initialize_bundle(changed)
except ValueError:
    print('foreign identities rejected')
else:
    raise AssertionError('Initialized process switched artifacts')
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("foreign identities rejected", result.stdout)

    def test_binding_rejects_nonisolated_python(self):
        result = self.invoke("""
try:
    initialize_bundle(context)
except ValueError as exc:
    print(str(exc))
else:
    raise AssertionError('Non-isolated process was accepted')
""", isolated=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("requires isolated Python", result.stdout)


if __name__ == "__main__":
    unittest.main()
