from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


AUDIT_PATH = Path(__file__).resolve().parents[1] / "audit.py"
spec = importlib.util.spec_from_file_location("standalone_audit_under_test", AUDIT_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_duplicate_requirement_variants_and_extras_remain_visible(self):
        declared, locked = audit.dependency_inventory(
            {"project": {
                "name": "fixture",
                "dependencies": [
                    "Some_Package[Feature_B]==1; sys_platform == 'win32'",
                    "some.package @ git+https://example.invalid/repo@123; sys_platform != 'win32'",
                    "other==2",
                ],
                "optional-dependencies": {"media": ["Other>=2", "some-package[extra]==1"]},
            }},
            {"package": [
                {"name": "some-package", "version": "1", "source": {"registry": "https://example.invalid"},
                 "wheels": [{"url": "https://example.invalid/some_package-1-py3-none-any.whl"}]},
                {"name": "other", "version": "2", "source": {"git": "https://example.invalid/repo"}},
            ]},
        )
        self.assertEqual(declared["direct"]["entry_count"], 3)
        self.assertEqual(declared["direct"]["unique_name_count"], 2)
        self.assertEqual(declared["direct"]["names"], ["other", "some-package"])
        self.assertEqual(declared["direct"]["entries"][0]["requested_extras"], ["feature-b"])
        self.assertEqual(declared["optional_groups"]["media"]["entry_count"], 2)
        self.assertEqual(locked["package_record_count"], 2)
        self.assertEqual(locked["universal_candidate_record_count"], 1)
        self.assertEqual(locked["without_universal_candidates"][0]["name"], "other")

    def test_dynamic_imports_preserve_unknown_arguments_and_aliases(self):
        result = audit.inspect_source("agent/demo.py", """
import importlib as il
from importlib import import_module as load
import pathlib
from . import sibling
il.import_module("known.module")
load(variable_name)
__import__(name="another.module")
""", {"agent"})
        dynamic = result["dynamic_imports"]
        self.assertEqual([item["literal"] for item in dynamic], ["known.module", None, "another.module"])
        self.assertEqual(dynamic[1]["argument"], "variable_name")
        self.assertTrue(dynamic[1]["unresolved"])
        self.assertEqual(dynamic[0]["line"], 6)
        self.assertEqual(result["imports"][-1]["classification"], "relative")

    def test_native_loading_installers_and_pm_edges_are_reviewable(self):
        result = audit.inspect_source("main.py", """
import ctypes as ct
import subprocess as sp
import sys
from pm.client import ensure as prepare
from cryptography.hazmat import primitives
ct.CDLL("libc.so.6")
sp.run([sys.executable, "-m", "pip", "install", "some-package"])
sp.Popen(["npm", "install"])
sp.run([sys.executable, "-m", module_name])
sp.run(command_name)
prepare("feature")
""", {"pm"})
        self.assertEqual(result["native_load_calls"][0]["callee"], "ctypes.CDLL")
        self.assertEqual(result["native_load_calls"][0]["target"], "libc.so.6")
        self.assertEqual(result["native_import_candidates"][0]["module"], "cryptography.hazmat")
        self.assertEqual([item["manager"] for item in result["installer_candidates"]], ["pip", "npm"])
        self.assertEqual(result["process_calls"][0]["module_flag_argument"], "pip")
        self.assertTrue(result["process_calls"][2]["has_module_flag"])
        self.assertIsNone(result["process_calls"][2]["module_flag_argument"])
        self.assertEqual(result["process_calls"][3]["command"]["kind"], "unresolved")
        self.assertTrue(any(
            item.get("callee") == "pm.client.ensure" and item["kind"] == "call"
            for item in result["pm_edges"]
        ))

    def test_process_keyword_arguments_and_native_loader_aliases(self):
        result = audit.inspect_source("main.py", """
from ctypes import CDLL as loader
from asyncio import create_subprocess_exec as start
import subprocess
loader(library_name)
subprocess.run(args=["uv", "sync"])
start("python", "-m", "pm.worker")
""", set())
        self.assertIsNone(result["native_load_calls"][0]["target"])
        self.assertEqual(result["native_load_calls"][0]["argument"], "library_name")
        self.assertEqual(result["installer_candidates"][0]["manager"], "uv")
        self.assertEqual(result["process_calls"][1]["module_flag_argument"], "pm.worker")
        self.assertEqual(result["pm_edges"][-1]["kind"], "process_module")

    def test_dynamic_file_loader_reports_the_unknown_source_path(self):
        result = audit.inspect_source("main.py", """
from importlib.util import spec_from_file_location as locate
from importlib.machinery import SourceFileLoader
locate("named.module", computed_path)
SourceFileLoader("other.module", "/plugin.py")
""", set())
        self.assertEqual(result["dynamic_imports"][0]["kind"], "file_loader")
        self.assertTrue(result["dynamic_imports"][0]["unresolved"])
        self.assertEqual(result["dynamic_imports"][0]["argument"], "computed_path")
        self.assertEqual(result["dynamic_imports"][1]["literal"], "/plugin.py")

    def test_syntax_errors_and_excluded_tests_are_not_silently_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
            (root / "uv.lock").write_text("version=1\n", encoding="utf-8")
            (root / "main.py").write_text("import json\n", encoding="utf-8")
            (root / "agent").mkdir()
            (root / "agent" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
            (root / "agent" / "tests").mkdir()
            (root / "agent" / "tests" / "ignored.py").write_text("invalid ???\n", encoding="utf-8")
            (root / "setup.py").write_text("invalid ???\n", encoding="utf-8")
            (root / "skills").mkdir()
            (root / "skills" / "outside.py").write_text("invalid ???\n", encoding="utf-8")
            report = audit.inventory(root)
            runtime = report["runtime_inventory"]
            self.assertEqual(runtime["selected_file_count"], 2)
            self.assertEqual(runtime["parsed_file_count"], 1)
            self.assertEqual(runtime["parse_errors"][0]["path"], "agent/broken.py")
            self.assertEqual(runtime["parse_errors"][0]["line"], 1)
            self.assertEqual(runtime["parse_errors"][0]["error_type"], "SyntaxError")
            self.assertEqual(runtime["counts"]["imports"], 1)
            self.assertTrue(any("cannot prove full import closure" in item for item in report["limitations"]))

    def test_cli_is_isolated_deterministic_and_writes_full_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
            (root / "uv.lock").write_text("version=1\n", encoding="utf-8")
            (root / "main.py").write_text("import json\n", encoding="utf-8")
            command = [sys.executable, "-I", "-S", str(AUDIT_PATH), "--root", str(root)]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            second = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first.stdout, second.stdout)
            output = root / "report.json"
            written = subprocess.run(
                [*command, "--output", str(output)], check=True, capture_output=True, text=True,
            )
            self.assertEqual(first.stdout, written.stdout)
            summary = json.loads(written.stdout)
            full = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotIn("imports", summary["runtime_inventory"])
            self.assertEqual(full["runtime_inventory"]["imports"][0]["module"], "json")
            self.assertNotIn(folder, output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
