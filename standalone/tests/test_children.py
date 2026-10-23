"""Original plugin and code-kernel children reenter the same verified runtime."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


_DRIVER = r'''
from hermes_cli.plugins import PluginManager
from tools.registry import registry
from tools.code_kernel import execute_in_session_kernel, shutdown_all_kernels
from hermes_cli.plugin_host_profiles import _extract
from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath
home = pathlib.Path(os.environ["HERMES_HOME"])
os.environ["STANDALONE_FIXTURE_API_KEY"] = "synthetic-do-not-inherit"
inherited = {"PYTHONPATH": os.pathsep.join((str(source), str(source.parent / "vendor"), str(scratch / "work")))}
_strip_hermes_owned_pythonpath(inherited)
assert inherited["PYTHONPATH"] == str(scratch / "work"), inherited
manager = PluginManager()
try:
    manager.discover_and_load()
    loaded = manager._plugins["childprobe"]
    assert loaded.error is None, str(loaded.error)
    assert not any("childprobe" in name for name in sys.modules)
    first = json.loads(registry.dispatch("standalone_child_pid", {"x": "direct"}, scope=manager.scope_key))
    nested = json.loads(registry.dispatch("standalone_child_nested", {}, scope=manager.scope_key))
    assert first["pid"] != os.getpid() and first["x"] == "direct", first
    assert nested["pid"] == first["pid"] and nested["x"] == "nested", nested
    assert pathlib.Path(first["home"]) == home, first
    profile = _extract(scratch / "providerprobe", "_standalone_profile_child")
    assert "error" not in profile, profile
    entry = profile["profiles"][0]
    assert entry["name"] == "standalone-profile", profile
    identity = json.loads(entry["fields"]["description"])
    assert identity["pid"] != os.getpid() and identity["secret_inherited"] is False, profile
    kwargs = dict(task_id="standalone-kernel-owner", mode="strict", child_python=sys.executable,
                  child_cwd="", sandbox_tools=frozenset(), timeout=15, max_tool_calls=2,
                  reset=False, is_interrupted=lambda: False)
    first_cell = json.loads(execute_in_session_kernel(
        "import json,os; counter=41; print(json.dumps({'pid':os.getpid(),'secret_inherited': 'STANDALONE_FIXTURE_API_KEY' in os.environ}))",
        **kwargs))
    assert first_cell["exit_code"] == 0, first_cell
    kernel_identity = json.loads(first_cell["output"])
    assert kernel_identity["pid"] not in {os.getpid(), first["pid"]}, kernel_identity
    assert kernel_identity["secret_inherited"] is False, kernel_identity
    second_cell = json.loads(execute_in_session_kernel("counter += 1; print(counter)", **kwargs))
    assert second_cell["exit_code"] == 0 and second_cell["output"].strip() == "42", second_cell
    assert second_cell["kernel"]["reused"] is True, second_cell
    reset_cell = json.loads(execute_in_session_kernel("print('counter' in globals())", **dict(kwargs, reset=True)))
    assert reset_cell["exit_code"] == 0 and reset_cell["output"].strip() == "False", reset_cell
    assert reset_cell["kernel"]["state_reset"] is True, reset_cell
    report = {"plugin_pid": first["pid"], "nested_same_host": nested["pid"] == first["pid"],
              "profile_pid": identity["pid"], "kernel_pid": kernel_identity["pid"],
              "state_survived": True, "reset_cleared_state": True, "credential_scrubbing": True}
finally:
    shutdown_all_kernels()
    manager.unload()
    host = manager._plugin_host()
    if host is not None:
        host.shutdown()
print(json.dumps(report), flush=True)
os._exit(0)
'''

_PLUGIN = '''
import json, os
def register(ctx):
    ctx.register_tool(name="standalone_child_pid", toolset="childprobe",
        schema={"name":"standalone_child_pid","description":"Fixture","parameters":{"type":"object","properties":{}}},
        handler=lambda args, **kw: json.dumps({"pid":os.getpid(), "x":args.get("x"), "home":os.environ["HERMES_HOME"]}))
    ctx.register_tool(name="standalone_child_nested", toolset="childprobe",
        schema={"name":"standalone_child_nested","description":"Fixture","parameters":{"type":"object","properties":{}}},
        handler=lambda args, **kw: ctx.dispatch_tool("standalone_child_pid", {"x":"nested"}))
'''

_PROVIDER = '''
import json, os
from providers import register_provider
from providers.base import ProviderProfile
register_provider(ProviderProfile(name="standalone-profile", supports_model_listing=False,
    description=json.dumps({"pid":os.getpid(), "secret_inherited":"STANDALONE_FIXTURE_API_KEY" in os.environ})))
'''


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14),
                     "Original child-process integration currently verified on Linux/Python 3.14")
class OriginalChildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from standalone.tests.test_entrypoints import BundledSourceTestCase
        BundledSourceTestCase.setUpClass.__func__(cls)

    def test_plugin_host_profile_extraction_and_persistent_code_kernel(self):
        from standalone.tests.test_chat_entry import _LAUNCHER, OriginalChatTests
        from standalone.tests.test_entrypoints import support
        with tempfile.TemporaryDirectory(prefix="c-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            context = dict(self.context, artifact=str(scratch / "hermes.py"))
            context["manifest"] = dict(context["manifest"], internal_modules=[
                *context["manifest"]["internal_modules"], "hermes_cli.plugin_host_child",
                "standalone.code_kernel_worker",
            ])
            launcher = _LAUNCHER.replace("CONTEXT_JSON", repr(json.dumps(context))).replace(
                "SCRATCH_ROOT", repr(str(scratch))).replace("ENDPOINT_PORT", "0")
            Path(context["artifact"]).write_text(launcher, encoding="utf-8")
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment["PATH"] = ""
            home = Path(environment["HERMES_HOME"])
            plugin = home / "plugins" / "childprobe"
            plugin.mkdir(parents=True)
            (plugin / "plugin.yaml").write_text(json.dumps({"name": "childprobe", "version": "1.0"}))
            (plugin / "__init__.py").write_text(_PLUGIN)
            provider = scratch / "providerprobe"
            provider.mkdir()
            (provider / "__init__.py").write_text(_PROVIDER)
            (home / "config.yaml").write_text(json.dumps({
                "plugins": {"enabled": ["childprobe"], "isolation": "host"},
                "agent": {"environment_probe": False}, "tools": {"tool_search": {"enabled": False}},
            }))
            driver = launcher.split("arguments = sys.argv[1:]")[0] + _DRIVER
            try:
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", driver],
                                        env=environment, cwd=scratch / "work", capture_output=True,
                                        text=True, encoding="utf-8", timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr[-6000:])
                report = json.loads(result.stdout)
                self.assertTrue(report["nested_same_host"] and report["state_survived"]
                                and report["reset_cleared_state"] and report["credential_scrubbing"], report)
                events = scratch / "guard-events.jsonl"
                if events.exists():
                    blocked = [json.loads(line) for line in events.read_text().splitlines()]
                    self.assertTrue(all(event.get("recoverable_capability_probe") for event in blocked), blocked)
            finally:
                OriginalChatTests._stop_fixture_processes(self, scratch)


if __name__ == "__main__":
    unittest.main()
