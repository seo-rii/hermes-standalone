import contextlib
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from standalone.cli import main


class CommandTests(unittest.TestCase):
    def setUp(self):
        capabilities_path = Path(__file__).resolve().parents[1] / "capabilities.json"
        self.context = {
            "artifact": "/unused/hermes.py",
            "root": "/unused/cache",
            "bundle_sha256": "a" * 64,
            "manifest": {
                "project": "hermes-standalone",
                "version": "0.1.0.dev0",
                "upstream": {"commit": "baseline"},
                "runtime": {"bootstrap_python_min": "3.11", "hermes_python": "3.14"},
                "capabilities": json.loads(capabilities_path.read_text(encoding="utf-8")),
                "files": [{"path": "app/example.py", "sha256": "b" * 64, "size": 1}],
                "distributions": [],
            },
        }

    def invoke(self, args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(args, self.context)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_doctor_distinguishes_packaging_from_agent_readiness(self):
        self.context["manifest"]["capabilities"]["agent_runtime_ready"] = False
        status, output, _ = self.invoke(["doctor", "--json", "--require-agent"])
        report = json.loads(output)
        self.assertEqual(status, 2)
        self.assertFalse(report["agent_ready"])
        self.assertFalse(report["full_parity_verified"])
        self.assertTrue(report["blockers"])

    def test_original_commands_require_verified_runtime_before_loading_profile(self):
        status, output, error = self.invoke(["chat", "hello"])
        self.assertEqual(status, 2)
        self.assertEqual(output, "")
        self.assertIn("require the verified standalone runtime", error)

    def test_manifest_summary_retains_provenance_without_mutating_manifest(self):
        status, output, _ = self.invoke(["manifest"])
        report = json.loads(output)
        self.assertEqual(status, 0)
        self.assertEqual(report["file_count"], 1)
        self.assertNotIn("files", report)
        self.assertEqual(report["upstream"], {"commit": "baseline"})
        self.assertEqual(len(self.context["manifest"]["files"]), 1)

    def test_probe_parent_removes_nested_state_on_success_failure_and_timeout(self):
        self.context["artifact"] = str(Path(__file__).resolve())
        self.context["root"] = str(Path(__file__).resolve().parent)
        self.context["manifest"]["internal_modules"] = ["standalone.agent_probe_worker"]
        for outcome in ("passed", "failed", "timeout", "invalid-json", "bad-exit"):
            with self.subTest(outcome=outcome):
                roots = []

                def child(command, **kwargs):
                    scratch = Path(command[command.index("--scratch-root") + 1])
                    roots.append(scratch)
                    self.assertEqual({path.name for path in scratch.iterdir()}, {"home", "work", "tmp"})
                    nested = scratch / "home" / "hermes"
                    nested.mkdir()
                    (nested / "config.yaml").write_text('{"fixture": true}', encoding="utf-8")
                    if outcome == "timeout":
                        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                    content = "not json" if outcome == "invalid-json" else json.dumps({
                        "status": "passed" if outcome in ("passed", "bad-exit") else "failed",
                        "live_provider_verified": False,
                    })
                    return subprocess.CompletedProcess(
                        command, 0 if outcome == "passed" else 2, content, "",
                    )

                with mock.patch("standalone.cli.subprocess.run", side_effect=child):
                    status, output, _ = self.invoke(["probe-agent", "--stage", "fixture-turn", "--json"])
                self.assertEqual(status, 0 if outcome == "passed" else 2)
                self.assertEqual(len(roots), 1)
                self.assertFalse(roots[0].exists())
                report = json.loads(output)
                self.assertFalse(report["live_provider_verified"])
                self.assertEqual(report["status"], "passed" if outcome == "passed" else "failed")


if __name__ == "__main__":
    unittest.main()
