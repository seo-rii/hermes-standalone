"""Verify the opt-in AST transform using local executable fixtures only."""

import ast
import asyncio
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap
import tokenize
import unittest
from unittest import mock

from standalone.minify import minify_source


def padded(source: str) -> bytes:
    return ("\n" * 80 + textwrap.dedent(source)).encode("utf-8")


def ast_dump(data: bytes) -> str:
    return ast.dump(
        ast.parse(data, type_comments=True),
        annotate_fields=True,
        include_attributes=False,
    )


def fixture_namespace(source: bytes) -> dict:
    namespace = {"__name__": "minifier_fixture"}
    exec(compile(source, "fixture.py", "exec", dont_inherit=True, optimize=0), namespace)
    return namespace


class MinifyTests(unittest.TestCase):
    def test_deterministic_smaller_verified_candidate_is_idempotent(self):
        source = padded(
            """
            def greeting( name ):

                return  "hello "   +  name
            """
        )
        output, report = minify_source(source, "fixture.py")
        self.assertEqual(report["status"], "minified")
        self.assertLess(len(output), len(source))
        self.assertEqual((output, report), minify_source(source, "fixture.py"))
        self.assertEqual(ast_dump(output), ast_dump(source))
        self.assertEqual(report["original_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(report["output_sha256"], hashlib.sha256(output).hexdigest())
        self.assertEqual(report["ast_sha256"], hashlib.sha256(ast_dump(source).encode()).hexdigest())
        self.assertEqual(len(report["code_sha256"]), 64)
        self.assertIn("excluding-source-locations", report["scope"])
        repeated, repeated_report = minify_source(output, "fixture.py")
        self.assertEqual(repeated, output)
        self.assertEqual(repeated_report["reason"], "already_canonical")
        self.assertEqual(repeated_report["ast_sha256"], report["ast_sha256"])
        self.assertEqual(repeated_report["code_sha256"], report["code_sha256"])

    def test_functions_async_generators_match_docs_annotations_and_asserts(self):
        source = padded(
            '''
            """The module docstring is a retained runtime value."""
            from __future__ import annotations

            class Example:
                """Public class documentation."""
                value: int = 4

                def twice(self, value: int = 2) -> int:
                    """Public method documentation."""
                    assert value >= 0, "negative input"
                    return value * 2

            def values(limit):
                yield from range(limit)

            async def async_values(limit):
                for value in range(limit):
                    yield value

            async def collect(limit):
                return [value async for value in async_values(limit)]

            def classify(value):
                match value:
                    case {"kind": kind, "values": [first, *rest]}:
                        return kind, first, rest
                    case _:
                        return None

            def debug(value):
                return f"{ value = }, {value!r:>4}"
            '''
        )
        output, report = minify_source(source, "fixture.py")
        if sys.version_info >= (3, 13):
            # CPython exposes a class's defining line as __firstlineno__,
            # so relocating this class changes executable co_consts.
            self.assertEqual(report["reason"], "compiled_code_mismatch")
            self.assertEqual(output, source)
        else:
            self.assertEqual(report["status"], "minified")
        original = fixture_namespace(source)
        transformed = fixture_namespace(output)
        for namespace in (original, transformed):
            self.assertEqual(namespace["__doc__"], "The module docstring is a retained runtime value.")
            example = namespace["Example"]
            self.assertEqual(example.__doc__, "Public class documentation.")
            self.assertEqual(example.twice.__doc__, "Public method documentation.")
            self.assertEqual(example.twice.__annotations__, {"value": "int", "return": "int"})
            self.assertEqual(example.__annotations__, {"value": "int"})
            self.assertEqual(example().twice(3), 6)
            with self.assertRaisesRegex(AssertionError, "negative input"):
                example().twice(-1)
            self.assertEqual(list(namespace["values"](3)), [0, 1, 2])
            self.assertEqual(asyncio.run(namespace["collect"](3)), [0, 1, 2])
            self.assertEqual(namespace["classify"]({"kind": "x", "values": [2, 3, 4]}), ("x", 2, [3, 4]))
            self.assertEqual(namespace["debug"](7), " value = 7,    7")
        self.assertEqual(
            inspect.signature(original["Example"].twice),
            inspect.signature(transformed["Example"].twice),
        )

    def test_numeric_constant_types_negative_zero_and_frozenset_semantics(self):
        source = padded(
            """
            INTEGER = 1
            REAL = 1.0
            NEGATIVE_ZERO = -0.0
            COMPLEX = -0.0j
            def contains(value):
                return value in {"gamma", "alpha", "beta"}
            def take_middle(value):
                return value[1:-1:2]
            """
        )
        output, report = minify_source(source, "fixture.py")
        self.assertEqual(report["status"], "minified")
        namespace = fixture_namespace(output)
        self.assertIs(type(namespace["INTEGER"]), int)
        self.assertIs(type(namespace["REAL"]), float)
        self.assertIs(type(namespace["COMPLEX"]), complex)
        self.assertEqual(math.copysign(1, namespace["NEGATIVE_ZERO"]), -1)
        self.assertTrue(namespace["contains"]("alpha"))
        self.assertFalse(namespace["contains"]("other"))
        self.assertEqual(namespace["take_middle"]("abcdef"), "bd")

    def test_compiled_fingerprint_is_stable_across_fresh_interpreters_and_filenames(self):
        source = padded(
            """
            def contains(value):
                return value in {"gamma", "alpha", "beta"}
            """
        )
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            (
                "import json, sys; sys.path.insert(0, sys.argv[1]); "
                "from standalone.minify import minify_source; "
                "output, report = minify_source(bytes.fromhex(sys.argv[2]), sys.argv[3]); "
                "print(json.dumps([output.hex(), report], sort_keys=True))"
            ),
            str(Path(__file__).resolve().parents[2]),
            source.hex(),
        ]
        reports = []
        for filename in ("first/path.py", "other/place.py"):
            result = subprocess.run(
                [*command, filename],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
                timeout=30,
            )
            reports.append(json.loads(result.stdout))
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0][1]["status"], "minified")

    def test_type_comments_survive_and_type_ignore_line_fields_are_not_ignored(self):
        source = padded(
            """
            value = []  # type: list[int]
            def identity(value):
                # type: (int) -> int
                return value
            """
        )
        output, report = minify_source(source, "typed.py")
        self.assertEqual(report["status"], "minified")
        self.assertEqual(ast_dump(output), ast_dump(source))
        self.assertIn(b"# type: list[int]", output)
        self.assertIn(b"# type: (int) -> int", output)

        ignored = padded("value = unknown  # type: ignore[name-defined]\n")
        output, report = minify_source(ignored, "ignored.py")
        self.assertEqual(output, ignored)
        self.assertEqual(report["reason"], "ast_mismatch")
        self.assertNotIn("ast_sha256", report)
        self.assertIn("TypeIgnore(lineno=81", ast_dump(ignored))

    def test_all_ordinary_comment_text_is_retained_in_original_order(self):
        source = (
            b"#!/usr/bin/env python3\n"
            b"# An unrecognized legal notice must also survive.\n"
            b"# Copyright (c) 2026 Example Authors\n"
            b"# Permission is granted under the included license.\n"
            + b"\n" * 80
            + b"value = 1  # All rights reserved.\n"
            b"# SPDX-License-Identifier: MIT\n"
            b"# formatting or author notes stay present too  \n"
        )
        output, report = minify_source(source, "notices.py")
        self.assertEqual(report["status"], "minified")
        comments = lambda data: [
            token.string
            for token in tokenize.tokenize(io.BytesIO(data).readline)
            if token.type == tokenize.COMMENT
        ]
        original_comments = comments(source)
        candidate_comments = comments(output)
        candidate_comments.remove("# coding: utf-8")
        self.assertEqual(candidate_comments, original_comments)
        self.assertTrue(output.startswith(b"#!/usr/bin/env python3\n"))
        self.assertEqual(minify_source(output, "notices.py")[0], output)

    def test_type_lookalike_comments_and_real_directive_whitespace_are_preserved(self):
        source = (
            b"\n" * 80
            + b"value = 1  # type: int\n"
            + b"other = 2  #\f type: int\n"
            + b"trailing = 3  # type: int  \n"
        )
        output, report = minify_source(source, "comment_whitespace.py")
        self.assertEqual(report["status"], "minified")
        self.assertIn(b"#\f type: int\n", output)
        self.assertIn(b"# type: int  \n", output)
        self.assertEqual(ast_dump(output), ast_dump(source))
        self.assertEqual(minify_source(output, "comment_whitespace.py")[0], output)

    def test_latin1_encoding_and_utf8_bom_remain_correct(self):
        source = (
            "# coding: latin-1\n# Copyright © café\n"
            + "\n" * 80
            + "MESSAGE = 'café'\n"
        ).encode("latin-1")
        output, report = minify_source(source, "latin1.py")
        self.assertEqual(report["status"], "minified")
        self.assertEqual(fixture_namespace(output)["MESSAGE"], "café")
        self.assertIn("# Copyright © café", output.decode("latin-1"))
        self.assertEqual(ast_dump(output), ast_dump(source))
        self.assertEqual(minify_source(output, "latin1.py")[0], output)

        source = ("\n" * 80 + "MESSAGE = '한글'\n").encode("utf-8-sig")
        output, report = minify_source(source, "bom.py")
        self.assertEqual(report["status"], "minified")
        self.assertTrue(output.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(fixture_namespace(output)["MESSAGE"], "한글")
        self.assertEqual(minify_source(output, "bom.py")[0], output)

    def test_unrepresentable_unparsed_literal_preserves_original_bytes(self):
        source = b"# coding: ascii\n" + b"\n" * 80 + b"MESSAGE = '\\u4e2d'\n"
        output, report = minify_source(source, "ascii.py")
        self.assertEqual(output, source)
        self.assertEqual(report["reason"], "encoding_unrepresentable")
        self.assertEqual(fixture_namespace(output)["MESSAGE"], "中")

    def test_source_positions_change_and_are_explicitly_outside_proof(self):
        source = padded(
            """
            def location():
                import sys
                return sys._getframe().f_lineno
            """
        )
        output, report = minify_source(source, "locations.py")
        self.assertEqual(report["status"], "minified")
        original = fixture_namespace(source)["location"]
        transformed = fixture_namespace(output)["location"]
        self.assertNotEqual(original.__code__.co_firstlineno, transformed.__code__.co_firstlineno)
        self.assertEqual(original.__code__.co_code, transformed.__code__.co_code)
        self.assertEqual(original.__name__, transformed.__name__)
        # Even equal AST/bytecode can produce a different result when the
        # program observes its source positions. Never call this full semantic
        # or observational equivalence.
        self.assertNotEqual(original(), transformed())
        self.assertEqual(report["scope"], "python-ast-and-bytecode-excluding-source-locations")

    def test_transformation_never_executes_or_imports_the_input(self):
        source = padded(
            """
            import hermes_minifier_this_module_does_not_exist
            raise RuntimeError("the transformer must not execute this source")
            """
        )
        output, report = minify_source(source, "untrusted_fixture.py")
        self.assertEqual(report["status"], "minified")
        self.assertEqual(ast_dump(output), ast_dump(source))
        self.assertNotIn("hermes_minifier_this_module_does_not_exist", sys.modules)

    def test_invalid_or_unsupported_source_and_non_smaller_results_stay_byte_exact(self):
        for source, reason in (
            (b"def broken(:\n", "unsupported_syntax"),
            (b"# coding: not-a-real-codec\nx = 1\n", "encoding_error"),
            (b"return 1\n", "compile_error"),
            (b"x=1\n", "not_smaller"),
        ):
            with self.subTest(source=source):
                output, report = minify_source(source, "fixture.py")
                self.assertIs(output, source)
                self.assertEqual(report["status"], "unchanged")
                self.assertEqual(report["reason"], reason)
                self.assertEqual(report["original_sha256"], report["output_sha256"])

    def test_candidate_cannot_change_literals_or_fail_idempotence(self):
        source = padded("value = 1.0\n")
        with mock.patch("standalone.minify.ast.unparse", return_value="value = 1"):
            output, report = minify_source(source, "fixture.py")
        self.assertEqual(output, source)
        self.assertEqual(report["reason"], "ast_mismatch")
        with mock.patch(
            "standalone.minify.ast.unparse",
            side_effect=["value=1.0", "value = 1.0"],
        ):
            output, report = minify_source(source, "fixture.py")
        self.assertEqual(output, source)
        self.assertEqual(report["reason"], "candidate_not_idempotent")


if __name__ == "__main__":
    unittest.main()
