"""Optional, conservative Python source canonicalization using only the stdlib.

This transformation proves equality of the complete parsed AST and compiled
code metadata except filenames and source locations. It does not prove full
observational equivalence: source bytes, comment locations, traceback line
numbers, tracing/coverage positions and source-inspection results can change.
Programs observing their own frames or source can therefore return different
values even when both structural checks pass.
Callers must keep this mode opt-in and retain original bytes by default.

Project source is parsed and compiled for comparison, never imported or run.
Sources that emit parser or compiler warnings retain their original bytes:
warning filters can turn those diagnostics into runtime import failures.
Ordinary comment token text, including copyright and license notices, is kept
in its original order in a header. Type comments remain attached by the AST.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import struct
import sys
import tokenize
import types
import warnings


# Python's tokenizer accepts spaces and tabs here, not arbitrary Unicode
# whitespace. A broader match can mistake an ordinary comment for a directive.
_TYPE_COMMENT = re.compile(r"#[ \t]*type:[ \t]*(.*)\Z")
_SCOPE = "python-ast-and-bytecode-excluding-source-locations"
_TOOL = "stdlib-ast-unparse-v2"


def _json_bytes(value: object) -> bytes:
    """Canonicalize comparison records without locale or hash-order effects."""
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _constant_record(value: object) -> object:
    """Preserve constant types and float bit patterns, including signed zero."""
    if value is None:
        return ["none"]
    if value is Ellipsis:
        return ["ellipsis"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        # Hex avoids Python's configurable limit on decimal integer rendering.
        return ["int", hex(value)]
    if type(value) is float:
        return ["float", struct.pack(">d", value).hex()]
    if type(value) is complex:
        return [
            "complex",
            struct.pack(">d", value.real).hex(),
            struct.pack(">d", value.imag).hex(),
        ]
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if type(value) is tuple:
        return ["tuple", [_constant_record(item) for item in value]]
    if type(value) is slice:
        # CPython 3.14 can constant-fold slices into co_consts.
        return [
            "slice",
            _constant_record(value.start),
            _constant_record(value.stop),
            _constant_record(value.step),
        ]
    if type(value) is frozenset:
        items = [_constant_record(item) for item in value]
        return ["frozenset", sorted(items, key=_json_bytes)]
    if isinstance(value, types.CodeType):
        return ["code", _code_record(value)]
    raise ValueError(f"unsupported compiled constant type: {type(value).__name__}")


def _code_record(code: types.CodeType) -> dict[str, object]:
    """Describe executable semantics, deliberately excluding source positions."""
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "exceptiontable": code.co_exceptiontable.hex(),
        "constants": [_constant_record(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "name": code.co_name,
        "qualname": code.co_qualname,
    }


def _candidate(
    data: bytes, tree: ast.AST, encoding: str
) -> tuple[bytes, int]:
    """Produce a stable candidate while preserving ordinary comment text."""
    text = data.decode(encoding)
    # A source's coding cookie is kept as ordinary comment text when its
    # spelling differs. This canonical declaration remains first (or second
    # after a shebang) so relocated comments cannot change source decoding.
    cookie_encoding = "utf-8" if encoding == "utf-8-sig" else encoding
    cookie = f"# coding: {cookie_encoding}"
    comments = []
    shebang = None
    type_comments = {
        node.type_comment
        for node in ast.walk(tree)
        if isinstance(getattr(node, "type_comment", None), str)
    }
    type_comments.update(
        f"ignore{node.tag}" for node in ast.walk(tree) if isinstance(node, ast.TypeIgnore)
    )
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        if token.start == (1, 0) and token.string.startswith("#!"):
            shebang = token.string
            continue
        directive = _TYPE_COMMENT.fullmatch(token.string)
        if directive and directive.group(1) in type_comments:
            # ast.unparse emits recognized directives with their owning nodes.
            # Exact AST verification below also compares TypeIgnore.lineno.
            continue
        cookie_line = 2 if shebang else 1
        if token.start == (cookie_line, 0) and token.string == cookie:
            # Avoid adding another copy of our exact declaration on the next
            # pass. Its text is retained by the canonical declaration itself.
            continue
        comments.append(token.string)
    header = ([shebang] if shebang else []) + [cookie, *comments]
    output = "\n".join([*header, ast.unparse(tree), ""])
    return output.encode(encoding), len(comments) + (1 if shebang else 0)


def minify_source(
    data: bytes, filename: str, *, preserve_exact: bool = False,
) -> tuple[bytes, dict[str, object]]:
    """Return a smaller verified candidate, or the exact original bytes.

    The report is deterministic for a fixed Python toolchain. AST unparser and
    compiler versions are part of that toolchain; cross-version byte-for-byte
    reproducibility is not promised. No filesystem or project import is used.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    digest = hashlib.sha256(data).hexdigest()
    report: dict[str, object] = {
        "status": "unchanged",
        "reason": "unverified",
        "original_size": len(data),
        "output_size": len(data),
        "original_sha256": digest,
        "output_sha256": digest,
        "scope": _SCOPE,
        "tool": _TOOL,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
    }
    if preserve_exact:
        report.update(reason="reviewed_resource", scope="byte-identical-reviewed-resource")
        return data, report
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        data.decode(encoding)
    except (SyntaxError, LookupError, UnicodeError):
        report["reason"] = "encoding_error"
        return data, report
    try:
        # AST equality cannot prove preservation of compiler diagnostics. In
        # particular, unparse normalizes invalid escapes and removes warnings
        # that can make the original fail to import under -Werror.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            original_tree = ast.parse(data, filename=filename, type_comments=True)
        original_ast = ast.dump(
            original_tree, annotate_fields=True, include_attributes=False
        )
    except (SyntaxError, ValueError, RecursionError, Warning):
        report["reason"] = "unsupported_syntax"
        return data, report
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            original_code = compile(data, filename, "exec", dont_inherit=True, optimize=0)
    except (SyntaxError, ValueError, RecursionError, Warning):
        report["reason"] = "compile_error"
        return data, report
    try:
        candidate, comment_count = _candidate(data, original_tree, encoding)
    except UnicodeError:
        report["reason"] = "encoding_unrepresentable"
        return data, report
    except (ValueError, RecursionError, tokenize.TokenError):
        report["reason"] = "unparse_error"
        return data, report
    try:
        candidate_tree = ast.parse(candidate, filename=filename, type_comments=True)
        candidate_ast = ast.dump(
            candidate_tree, annotate_fields=True, include_attributes=False
        )
    except (SyntaxError, ValueError, RecursionError):
        report["reason"] = "candidate_parse_error"
        return data, report
    if candidate_ast != original_ast:
        report["reason"] = "ast_mismatch"
        return data, report
    try:
        candidate_code = compile(
            candidate, filename, "exec", dont_inherit=True, optimize=0
        )
        original_record = _json_bytes(_code_record(original_code))
        candidate_record = _json_bytes(_code_record(candidate_code))
    except (SyntaxError, ValueError, RecursionError):
        report["reason"] = "candidate_compile_error"
        return data, report
    if candidate_record != original_record:
        report["reason"] = "compiled_code_mismatch"
        return data, report
    try:
        second_candidate, _ = _candidate(candidate, candidate_tree, encoding)
    except (ValueError, UnicodeError, RecursionError, tokenize.TokenError):
        report["reason"] = "candidate_not_idempotent"
        return data, report
    if second_candidate != candidate:
        report["reason"] = "candidate_not_idempotent"
        return data, report
    report.update(
        ast_sha256=hashlib.sha256(original_ast.encode("utf-8")).hexdigest(),
        code_sha256=hashlib.sha256(original_record).hexdigest(),
        preserved_comments=comment_count,
    )
    if len(candidate) >= len(data):
        report["reason"] = "already_canonical" if candidate == data else "not_smaller"
        return data, report
    report.update(
        status="minified",
        reason="smaller_verified_ast_and_code",
        output_size=len(candidate),
        output_sha256=hashlib.sha256(candidate).hexdigest(),
    )
    return candidate, report
