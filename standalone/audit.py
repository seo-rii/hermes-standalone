#!/usr/bin/env python3
"""Inventory upstream portability edges without importing or installing Hermes.

The default output is a bounded JSON summary. --output writes the full inventory;
--full also prints that inventory. Results are stable for the same input files and
Python parser, with no timestamps, absolute source paths, or environment discovery.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tokenize
import tomllib
from urllib.parse import unquote, urlsplit


RUNTIME_DIRECTORIES = (
    "acp_adapter", "agent", "cron", "gateway", "hermes_cli", "hermes_platform",
    "plugins", "pm", "providers", "tools", "tui_gateway",
)
EXCLUDED_DIRECTORIES = {
    ".cache", ".git", ".venv", "__pycache__", "node_modules", "test", "tests",
}
EXCLUDED_ROOT_FILES = {"setup.py"}
NATIVE_IMPORT_CANDIDATES = {
    "_cffi_backend", "cffi", "cryptography", "fitz", "httptools", "jiter",
    "numpy", "onnxruntime", "PIL", "pillow_heif", "psutil", "pydantic_core",
    "pymupdf", "sounddevice", "soundfile", "tiktoken", "torch", "uvloop",
    "watchfiles", "win32api", "win32file", "win32security", "winpty", "winrt",
}
PROCESS_CALLS = {
    "asyncio.create_subprocess_exec", "asyncio.create_subprocess_shell",
    "os.popen", "os.system", "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe", "os.spawnl", "os.spawnle",
    "os.spawnlp", "os.spawnlpe", "os.spawnv", "os.spawnve", "os.spawnvp",
    "os.spawnvpe", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "subprocess.getoutput", "subprocess.getstatusoutput",
    "subprocess.Popen", "subprocess.run",
}
PACKAGE_MANAGERS = {
    "apt", "apt-get", "brew", "choco", "conda", "dnf", "npm", "npx",
    "pip", "pip3", "pipx", "pnpm", "uv", "uvx", "winget", "yarn",
}
LIMITATIONS = [
    "This static inventory cannot prove full import closure or runtime compatibility.",
    "Dependency markers, extras, and Python/wheel tags are recorded but not resolved; "
    "lock counts include all platforms and optional extras, not a core installation.",
    "A none-any wheel filename is only a portability candidate. Its contents have not "
    "been checked; such wheels can still ship binaries, invoke programs, or load native code.",
    "Import aliases are matched within each file, without scope or data-flow analysis. "
    "Shadowing, generated code, exec/eval, and indirect calls can hide or misclassify edges.",
    "Imports include conditional, optional, and type-checking branches. An import edge "
    "does not establish that a dependency is required for every invocation.",
    "Dynamic import and process arguments are resolved only when directly literal. "
    "Unresolved arguments remain explicit; commands are never executed.",
    "Process -m tokens are argument candidates, not proof of a Python module launch; "
    "the token can belong to a helper, shell, or non-Python program.",
    "Native import names and loader calls are review candidates. OS-library ctypes "
    "calls are distinct from bundling third-party native extensions.",
    "Only selected upstream runtime Python files are parsed; tests, skill scripts, "
    "shell scripts, JavaScript applications, and external plugins are outside this scope.",
    "Files rejected by this Python parser are listed as parse errors, not counted as "
    "successfully inspected. Re-run with the upstream Python version for those files.",
]


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_group(requirements: list[str]) -> dict:
    entries = []
    for requirement in requirements:
        match = re.match(
            r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\s*\[([A-Za-z0-9.,_\s-]+)\])?",
            requirement,
        )
        entry = {"requirement": requirement, "name": None, "requested_extras": []}
        if match:
            entry["name"] = normalized_name(match.group(1))
            entry["requested_extras"] = sorted({
                normalized_name(extra.strip())
                for extra in (match.group(2) or "").split(",") if extra.strip()
            })
        else:
            entry["parse_error"] = "Cannot identify the distribution name."
        entries.append(entry)
    names = sorted({entry["name"] for entry in entries if entry["name"]})
    return {
        "entry_count": len(entries),
        "unique_name_count": len(names),
        "names": names,
        "entries": entries,
    }


def dependency_inventory(project: dict, lock: dict) -> tuple[dict, dict]:
    metadata = project.get("project", {})
    direct = dependency_group(metadata.get("dependencies", []))
    extras = {
        name: dependency_group(requirements)
        for name, requirements in sorted(metadata.get("optional-dependencies", {}).items())
    }
    declared = {
        "project": metadata.get("name"),
        "requires_python": metadata.get("requires-python"),
        "direct": direct,
        "optional_group_count": len(extras),
        "optional_groups": extras,
    }
    packages = []
    for package in lock.get("package", []):
        wheels = sorted({
            unquote(urlsplit(wheel.get("url", wheel.get("path", ""))).path).rsplit("/", 1)[-1]
            for wheel in package.get("wheels", [])
        })
        candidates = [name for name in wheels if name.endswith("-none-any.whl")]
        packages.append({
            "name": normalized_name(package["name"]),
            "version": package.get("version"),
            "source": package.get("source", {}),
            "wheels": wheels,
            "universal_wheel_candidates": candidates,
            "has_sdist": "sdist" in package,
        })
    packages.sort(key=lambda item: (
        item["name"], item["version"] or "", json.dumps(item["source"], sort_keys=True),
    ))
    without_candidate = [
        {"name": item["name"], "version": item["version"], "source": item["source"]}
        for item in packages if not item["universal_wheel_candidates"]
    ]
    locked = {
        "requires_python": lock.get("requires-python"),
        "supported_markers": lock.get("supported-markers", []),
        "resolution_markers": lock.get("resolution-markers", []),
        "package_record_count": len(packages),
        "unique_name_count": len({item["name"] for item in packages}),
        "universal_candidate_record_count": len(packages) - len(without_candidate),
        "without_universal_candidate_record_count": len(without_candidate),
        "without_universal_candidates": without_candidate,
        "packages": packages,
    }
    return declared, locked


def expression(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    text = ast.unparse(node)
    return text if len(text) <= 240 else text[:237] + "..."


def literal_string(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def dotted_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value, aliases)
        return parent + "." + node.attr if parent else node.attr
    return ""


def call_argument(node: ast.Call, keyword: str = "name") -> ast.AST | None:
    if node.args:
        return node.args[0]
    return next((item.value for item in node.keywords if item.arg == keyword), None)


def process_command(node: ast.Call, callee: str) -> dict:
    argument = call_argument(node, "args")
    if callee == "asyncio.create_subprocess_exec":
        argument = ast.List(elts=node.args)
    elif callee.startswith("os.spawn"):
        # os.spawn* starts with the process mode, unlike exec*.
        argument = node.args[1] if len(node.args) > 1 else None
    if callee.startswith(("os.execv", "os.spawnv")):
        index = 2 if callee.startswith("os.spawn") else 1
        argument = node.args[index] if len(node.args) > index else argument
    elif callee.startswith(("os.execl", "os.spawnl")):
        index = 2 if callee.startswith("os.spawn") else 1
        argument = ast.List(elts=node.args[index:])
    tokens = None
    error = None
    if isinstance(argument, (ast.List, ast.Tuple)):
        tokens = [literal_string(item) for item in argument.elts]
        kind = "argv"
    elif literal_string(argument) is not None:
        kind = "string"
        try:
            tokens = shlex.split(literal_string(argument))
        except ValueError as exc:
            error = str(exc)
    else:
        kind = "unresolved"
    result = {"kind": kind, "expression": expression(argument), "tokens": tokens}
    if error:
        result["tokenization_error"] = error
    return result


def import_inventory(tree: ast.AST, path: str, local_roots: set[str]) -> tuple[dict, dict]:
    aliases = {}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0]
                )
                imports.append({
                    "path": path, "line": node.lineno, "module": item.name,
                    "level": 0, "names": [],
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for item in node.names:
                if not node.level:
                    aliases[item.asname or item.name] = module + "." + item.name
            imports.append({
                "path": path, "line": node.lineno, "module": module,
                "level": node.level, "names": [item.name for item in node.names],
            })
    native_imports = []
    pm_edges = []
    for item in imports:
        root = item["module"].split(".")[0]
        if item["level"]:
            item["classification"] = "relative"
        elif root in local_roots:
            item["classification"] = "bundled_module_candidate"
        elif root in sys.stdlib_module_names:
            item["classification"] = "stdlib"
        else:
            item["classification"] = "external_or_unresolved"
        if not item["level"] and root in NATIVE_IMPORT_CANDIDATES:
            native_imports.append(dict(item))
        if not item["level"] and root == "pm":
            pm_edges.append({**item, "kind": "import"})
    return aliases, {
        "imports": imports, "dynamic_imports": [], "process_calls": [],
        "native_import_candidates": native_imports, "native_load_calls": [],
        "installer_candidates": [], "pm_edges": pm_edges,
    }


def record_process(node: ast.Call, location: dict, result: dict) -> None:
    command = process_command(node, location["callee"])
    tokens = command["tokens"] or []
    module_index = next(
        (index + 1 for index, token in enumerate(tokens) if token == "-m"), None,
    )
    module = tokens[module_index] if module_index is not None and module_index < len(tokens) else None
    process = {
        **location, "command": command, "has_module_flag": module_index is not None,
        "module_flag_argument": module,
    }
    result["process_calls"].append(process)
    executable = (tokens[0] or "").replace("\\", "/").rsplit("/", 1)[-1] if tokens else ""
    executable = executable.lower().removesuffix(".exe").removesuffix(".cmd")
    if executable in PACKAGE_MANAGERS or module in {"pip", "ensurepip", "uv"}:
        result["installer_candidates"].append({
            **process,
            "manager": module if module in {"pip", "ensurepip", "uv"} else executable,
        })
    if module and (module == "pm" or module.startswith("pm.")):
        result["pm_edges"].append({**location, "kind": "process_module", "module": module})


def record_call(node: ast.Call, location: dict, result: dict) -> None:
    callee = location["callee"]
    if callee in {
        "__import__", "builtins.__import__", "importlib.import_module",
        "importlib.util.spec_from_file_location", "importlib.machinery.SourceFileLoader",
    }:
        argument = call_argument(node)
        kind = "module"
        if callee in {
            "importlib.util.spec_from_file_location", "importlib.machinery.SourceFileLoader",
        }:
            kind = "file_loader"
            argument = node.args[1] if len(node.args) > 1 else next(
                (item.value for item in node.keywords if item.arg in {"location", "path"}), None,
            )
        literal = literal_string(argument)
        result["dynamic_imports"].append({
            **location, "literal": literal, "unresolved": literal is None,
            "argument": expression(argument), "kind": kind,
        })
    if callee.startswith("pm."):
        result["pm_edges"].append({**location, "kind": "call"})
    if (
        callee in {
            "ctypes.CDLL", "ctypes.PyDLL", "ctypes.WinDLL", "ctypes.OleDLL",
            "importlib.machinery.ExtensionFileLoader", "_imp.create_dynamic",
            "_imp.exec_dynamic",
        }
        or callee.startswith(("ctypes.cdll.", "ctypes.windll.", "ctypes.oledll.", "ctypes.pydll."))
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "dlopen")
    ):
        argument = call_argument(node)
        result["native_load_calls"].append({
            **location, "target": literal_string(argument),
            "argument": expression(argument),
        })
    if callee in PROCESS_CALLS:
        record_process(node, location, result)


def inspect_source(path: str, source: str, local_roots: set[str]) -> dict:
    """Parse only: never import inspected modules or resolve installed packages."""
    tree = ast.parse(source, filename=path)
    aliases, result = import_inventory(tree, path, local_roots)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            location = {
                "path": path, "line": node.lineno,
                "callee": dotted_name(node.func, aliases),
            }
            record_call(node, location, result)
    return result


def selected_sources(root: Path) -> tuple[list[Path], list[str]]:
    paths = []
    symlinks = []
    for path in root.glob("*.py"):
        if path.name in EXCLUDED_ROOT_FILES:
            continue
        if path.is_symlink():
            symlinks.append(path.relative_to(root).as_posix())
        elif path.is_file():
            paths.append(path)
    for directory in RUNTIME_DIRECTORIES:
        parent = root / directory
        if parent.is_symlink():
            symlinks.append(directory)
            continue
        if not parent.is_dir():
            continue
        for current, directories, filenames in os.walk(parent, followlinks=False):
            current = Path(current)
            for name in sorted(directories):
                if (current / name).is_symlink():
                    symlinks.append((current / name).relative_to(root).as_posix())
            directories[:] = sorted(
                name for name in directories
                if name not in EXCLUDED_DIRECTORIES and not (current / name).is_symlink()
            )
            for name in sorted(filenames):
                path = current / name
                if path.suffix != ".py":
                    continue
                if path.is_symlink():
                    symlinks.append(path.relative_to(root).as_posix())
                elif path.is_file():
                    paths.append(path)
    return sorted(paths), sorted(symlinks)


def inventory(root: Path) -> dict:
    root = Path(root)
    inputs = {}
    parsed = {}
    for name in ("pyproject.toml", "uv.lock"):
        data = (root / name).read_bytes()
        inputs[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        parsed[name] = tomllib.loads(data.decode("utf-8"))
    declared, locked = dependency_inventory(parsed["pyproject.toml"], parsed["uv.lock"])
    paths, symlinks = selected_sources(root)
    local_roots = {path.stem for path in paths if path.parent == root}
    local_roots.update(name for name in RUNTIME_DIRECTORIES if (root / name).is_dir())
    categories = (
        "imports", "dynamic_imports", "process_calls", "native_import_candidates",
        "native_load_calls", "installer_candidates", "pm_edges",
    )
    runtime = {name: [] for name in categories}
    errors = []
    source_inputs = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
            source_inputs.append({
                "path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
            })
            encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
            found = inspect_source(relative, data.decode(encoding), local_roots)
        except (SyntaxError, UnicodeError, OSError) as exc:
            message = exc.msg if isinstance(exc, SyntaxError) else (
                exc.strerror if isinstance(exc, OSError) else str(exc)
            )
            errors.append({
                "path": relative, "line": getattr(exc, "lineno", None),
                "column": getattr(exc, "offset", None), "error_type": type(exc).__name__,
                "message": message,
            })
            continue
        for category in categories:
            runtime[category].extend(found[category])
    for category in categories:
        runtime[category].sort(key=lambda item: (
            item["path"], item["line"], json.dumps(item, sort_keys=True),
        ))
    runtime.update({
        "scope": {
            "runtime_directories": list(RUNTIME_DIRECTORIES),
            "root_pattern": "*.py",
            "excluded_root_files": sorted(EXCLUDED_ROOT_FILES),
            "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
            "excluded_symlinks": symlinks,
        },
        "source_files": source_inputs,
        "selected_file_count": len(paths),
        "parsed_file_count": len(paths) - len(errors),
        "parse_errors": errors,
        "counts": {name: len(runtime[name]) for name in categories},
    })
    runtime["counts"]["unresolved_dynamic_imports"] = sum(
        item["unresolved"] for item in runtime["dynamic_imports"]
    )
    runtime["counts"]["unresolved_process_commands"] = sum(
        item["command"]["kind"] == "unresolved" for item in runtime["process_calls"]
    )
    runtime["counts"]["process_module_flags"] = sum(
        item["has_module_flag"] for item in runtime["process_calls"]
    )
    roots = Counter(
        item["module"].split(".")[0] for item in runtime["imports"]
        if item["classification"] == "external_or_unresolved"
    )
    runtime["external_import_roots"] = dict(sorted(roots.items()))
    return {
        "schema_version": 1,
        "parser_python": ".".join(map(str, sys.version_info[:3])),
        "inputs": inputs,
        "declared_dependencies": declared,
        "lock_inventory": locked,
        "runtime_inventory": runtime,
        "limitations": LIMITATIONS,
    }


def summary(report: dict) -> dict:
    runtime = report["runtime_inventory"]
    declared = report["declared_dependencies"]
    lock = report["lock_inventory"]
    return {
        "schema_version": report["schema_version"],
        "parser_python": report["parser_python"],
        "inputs": report["inputs"],
        "declared_dependencies": {
            "project": declared["project"],
            "requires_python": declared["requires_python"],
            "direct_entry_count": declared["direct"]["entry_count"],
            "direct_unique_name_count": declared["direct"]["unique_name_count"],
            "direct_names": declared["direct"]["names"],
            "optional_group_count": declared["optional_group_count"],
            "optional_groups": {
                name: {"entry_count": group["entry_count"], "unique_name_count": group["unique_name_count"]}
                for name, group in declared["optional_groups"].items()
            },
        },
        "lock_inventory": {
            key: value for key, value in lock.items()
            if key not in {"packages", "without_universal_candidates"}
        },
        "runtime_inventory": {
            key: runtime[key] for key in (
                "scope", "selected_file_count", "parsed_file_count", "counts", "parse_errors",
            )
        },
        "examples": {
            **{
                name: runtime[name][:5] for name in (
                    "dynamic_imports", "native_import_candidates", "native_load_calls",
                    "installer_candidates", "pm_edges",
                )
            },
            "process_module_flags": [
                item for item in runtime["process_calls"] if item["has_module_flag"]
            ][:5],
            "without_universal_candidates": lock["without_universal_candidates"][:5],
        },
        "limitations": report["limitations"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Write the complete inventory to this JSON file.")
    parser.add_argument("--full", action="store_true", help="Print the complete inventory instead of its summary.")
    args = parser.parse_args(argv)
    try:
        report = inventory(args.root)
        if args.output:
            args.output.write_text(
                json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(
        report if args.full else summary(report), sort_keys=True, indent=2, ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
