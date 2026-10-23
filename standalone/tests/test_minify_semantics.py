"""Executable import diagnostics are part of the source-retention contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

from standalone.minify import minify_source


_IMPORT = r'''
import importlib.util,json,pathlib,sys,warnings
path=pathlib.Path(sys.argv[1]);result={}
with warnings.catch_warnings(record=True) as events:
    spec=importlib.util.spec_from_file_location('minify_semantics_fixture',path)
    module=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        result.update(error_type=type(error).__name__,error=error.args[0])
    else:
        result.update(value=module.VALUE)
    result['warnings']=[{'category':type(event.message).__name__,'message':str(event.message)} for event in events]
print(json.dumps(result,sort_keys=True))
'''


class CompileWarningSemanticsTests(unittest.TestCase):
    def import_outcome(self, path, mode):
        environment = {"PATH": "", "HOME": str(path.parent), "USERPROFILE": str(path.parent),
                       "HERMES_HOME": str(path.parent / "unused-hermes-home")}
        process = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-W", mode,
                                  "-c", _IMPORT, str(path)], cwd=path.parent, env=environment,
                                 capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def assert_warning_source_preserved(self, source, *, parser_warns):
        with warnings.catch_warnings(record=True) as diagnostics:
            warnings.simplefilter("always")
            ast.parse(source, filename="consumer.py", type_comments=True)
        self.assertEqual(bool(diagnostics), parser_warns)
        # A caller hiding build-time warnings must not make an unsafe candidate
        # eligible: users can import the resulting artifact with -Werror later.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filters = warnings.filters[:]
            output, report = minify_source(source, "consumer.py")
            self.assertEqual(warnings.filters, filters)
        for action in ("always", "error"):
            with warnings.catch_warnings():
                warnings.simplefilter(action)
                self.assertEqual(minify_source(source, "consumer.py"), (output, report))
        with tempfile.TemporaryDirectory(prefix="hermes-minify-semantics-") as directory:
            root = Path(directory)
            original, transformed = root / "original/consumer.py", root / "transformed/consumer.py"
            for path, content in ((original, source), (transformed, output)):
                path.parent.mkdir()
                path.write_bytes(content)
            for mode in ("always", "error"):
                with self.subTest(warnings=mode):
                    before, after = self.import_outcome(original, mode), self.import_outcome(transformed, mode)
                    self.assertEqual(after, before, mode + " changed import behavior")
                    if mode == "always":
                        self.assertTrue(before["warnings"])
                        self.assertIn("value", before)
                    else:
                        self.assertEqual(before["error_type"], "SyntaxError")
        self.assertIs(output, source)
        self.assertEqual(report["status"], "unchanged")
        self.assertEqual(report["original_sha256"], report["output_sha256"])

    def test_parser_warning_survives_real_import_and_werror(self):
        source = b"\n" * 80 + b"VALUE = '\\q'\n"
        self.assert_warning_source_preserved(source, parser_warns=True)

    def test_compiler_only_warning_survives_real_import_and_werror(self):
        source = b"\n" * 80 + b"VALUE = 256 is 256\n"
        self.assert_warning_source_preserved(source, parser_warns=False)


if __name__ == "__main__":
    unittest.main()
