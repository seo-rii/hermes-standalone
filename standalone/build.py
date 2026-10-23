"""Build dist/hermes.py from reviewed sources and verified dependency archives.

No network request or package installation is performed by this command.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import zipfile


# Direct file loading makes `python3 -I -S standalone/build.py` independent of
# the working directory, site-packages and PYTHONPATH.
_vendor_spec = importlib.util.spec_from_file_location(
    "_hermes_standalone_build_vendor", Path(__file__).with_name("vendor.py")
)
if _vendor_spec is None or _vendor_spec.loader is None:
    raise RuntimeError("Cannot load standalone/vendor.py")
vendor = importlib.util.module_from_spec(_vendor_spec)
_vendor_spec.loader.exec_module(vendor)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIRECTORIES = (
    "agent", "hermes_cli", "tools", "gateway", "cron", "plugins", "pm",
    "hermes_platform", "tui_gateway", "acp_adapter", "providers",
)
RESOURCE_DIRECTORIES = ("skills", "optional-skills", "locales")
REQUIRED_ROOT_RESOURCES = ("LICENSE", "SOUL.md", "pyproject.toml", "uv.lock")
OPTIONAL_ROOT_RESOURCES = ("NOTICE", "NOTICE.md", "README.md")
SUPPLEMENTAL_LOCKFILE = "standalone/extra-dependencies.lock.json"
EXCLUDED_DIRECTORIES = {
    "tests", "test", "__pycache__", "node_modules", ".git", ".cache", ".venv",
}
EXCLUDED_STANDALONE_FILES = {
    "build.py", "vendor.py", "bootstrap.py", "check.py", "audit.py", "minify.py", "size_report.py",
    "offline_agent_probe.py", "upstream_check.py", "web_assets.py", "acp_port.py", "deflate_pack.py",
    "dictionary_pack.py",
    "mecab_pack.py", "model_pack.py", "pickle_pack.py",
}
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAYLOAD_LITERAL_WIDTH = 8192
LZMA_OPTIONS = {"preset": 8, "pb": 0, "dict_size": 48 * 1024 * 1024}
BUILD_VARIANTS = ("full", "lite")
LITE_DATA_PREFIXES = (
    "app/standalone/assets/piper/",
    "app/standalone/local_piper_japanese/data/",
    "app/standalone/piper_chinese_data/",
)
GOOGLE_DISCOVERY_PREFIX = "vendor/googleapiclient/discovery_cache/documents/"
GOOGLE_CHAT_DISCOVERY = GOOGLE_DISCOVERY_PREFIX + "chat.v1.json"
LITE_EXCLUDED_FEATURES = ("local-piper-model", "google-static-discovery")
# Original Piper package data, reviewed as model/character-map data rather
# than executable payloads. Include only these exact bytes, never *.onnx.
PIPER_RESOURCE_SHA256 = {
    "assets/piper/hebrew/nakdimon.onnx": "9ff491dcc7d66392019d427a98b97d5de10c0d721628ae740858174ae22b190e",
    "assets/piper/tashkeel/model.onnx": "52c40d42ca094ff9ff07a091d0a1eafd7baa5cf2388e60a9cc89bacd8e5e955f",
    "assets/piper/tashkeel/input_id_map.json": "727a49aa3c7e939dc06f32b27c2e06c4c4bbef596dce4bf51276261ca736c3c3",
    "assets/piper/tashkeel/target_id_map.json": "6da34024bfd4c0ff54f7598bb74d5a00405144d746afe72c141fceeaba21ed30",
    "assets/piper/tashkeel/hint_id_map.json": "80931db53b1f1031609a66f23b64a1c37d79ae4a8a4c785023728636a7fa93e5",
}
PIPER_ESPEAK_MANIFEST = "piper-espeak-resources-provenance.json"
PIPER_ESPEAK_MANIFEST_SHA256 = "16c885906be9b0aa5295961d2f2d67d5a531bc4ce5cf0c1a66a4ba5cd2267fc4"
SUDACHI_DICTIONARY = "app/standalone/local_piper_japanese/data/sudachi/system.dic"
PACKED_RESOURCES = {
    SUDACHI_DICTIONARY: ("dictionary_pack.py", "transform_dictionary", "sudachi-v1-blocks"),
    "app/standalone/local_piper_japanese/data/openjtalk/dictionary/sys.dic":
        ("mecab_pack.py", "transform_mecab", "mecab-columns-v1"),
    "app/standalone/piper_chinese_data/model/g2pw.onnx":
        ("model_pack.py", "transform_g2pw_embedding", "g2pw-embedding-v1"),
    "vendor/hermes_tltk/sylseg.3g":
        ("pickle_pack.py", "transform_pickle_memo", "pickle-memo-delta-v1"),
}
PIPER_REVIEWED_RESOURCE_MANIFESTS = (
    ("local_piper_japanese/__init__.py", "piper-japanese-resources-provenance.json",
     "99b3d3cd6364f117f8d31d476f0f4dfa7648abfdce996cacfa723888f1502e68"),
    ("local_piper_chinese.py", "piper-chinese-resources-provenance.json",
     "5f18e1c4d53a410d92183587706d0278927222f9ad6787bc35e5623988b09838"),
)
INTERNAL_MODULES = (
    "standalone.selftest_worker", "standalone.agent_probe_worker", "standalone.entry_worker",
    "hermes_cli.main", "gateway.run", "cli", "gateway.restart_watcher",
    "hermes_cli.stderr_timestamp", "gateway.cgroup_cleanup", "gateway.systemd_stop_mark",
    "agent.managed_worker", "hermes_cli.kanban_worker_client", "tools.mcp_death_supervisor",
    "hermes_cli.plugin_host_child", "standalone.code_kernel_worker",
    "standalone.profile_worker",
    "standalone.local_piper_download_voices",
    "browser_harness.run", "browser_harness.daemon", "browser_harness.mcp_cli",
    "plugins.web.ddgs._search_worker",
    "plugins.google_meet.meet_bot",
    "acp_adapter.entry", "acp_adapter",
)


class BuildError(ValueError):
    """An input does not satisfy the standalone bundle contract."""


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _validate_variant(variant: str) -> None:
    if variant not in BUILD_VARIANTS:
        raise BuildError(f"Unsupported build variant: {variant}")


def _include_variant_path(path: str, variant: str) -> bool:
    if variant == "full":
        return True
    if path.startswith(LITE_DATA_PREFIXES):
        return False
    if path.startswith("vendor/hermes_tltk/") and not path.endswith(".py"):
        return False
    if path.startswith(GOOGLE_DISCOVERY_PREFIX) and path != GOOGLE_CHAT_DISCOVERY:
        return False
    return True


def _variant_capabilities(capabilities: dict, variant: str) -> dict:
    if variant == "full":
        return capabilities
    # This is artifact configuration. The source checkout is never rewritten.
    capabilities = json.loads(json.dumps(capabilities))
    features = capabilities.setdefault("features", [])
    piper = next((item for item in features if item["id"] == "local-piper-model"), None)
    if piper is None:
        piper = {"id": "local-piper-model"}
        features.append(piper)
    piper.update(status="excluded", description=(
        "The lite distribution omits local Piper voice resources, auxiliary models "
        "and pronunciation dictionaries. Use the full distribution for local Piper; "
        "cloud TTS and other retained speech paths remain available."
    ))
    features.append({
        "id": "google-static-discovery", "status": "partial",
        "description": "The lite distribution retains the Google Chat v1 discovery document "
                       "and Google SDK/authentication code. Other preloaded Google API "
                       "discovery documents are omitted; explicit online discovery remains "
                       "a Google SDK option. No discovery request runs during startup.",
    })
    return capabilities


def _pack_model_planes(content: bytes) -> bytes:
    """Reorder four byte planes and prove the exact inverse before packaging."""
    packed = b"".join(content[index::4] for index in range(4))
    restored, offset = bytearray(len(content)), 0
    for index in range(4):
        count = (len(content) + 3 - index) // 4
        restored[index::4] = packed[offset:offset + count]
        offset += count
    if restored != content:
        raise BuildError("Model byte-plane transform failed its exact inverse")
    return packed


def _pack_data_files(files: dict[str, bytes]) -> tuple[dict, dict]:
    packed = {path: _pack_model_planes(content) for path, content in files.items()
              if path.removeprefix("app/standalone/") in PIPER_RESOURCE_SHA256 and path.endswith(".onnx")}
    encodings = {path: 'byteplanes4' for path in packed}
    for path, (filename, function, encoding) in PACKED_RESOURCES.items():
        if path not in files:
            continue
        spec = importlib.util.spec_from_file_location(
            '_hermes_' + Path(filename).stem, Path(__file__).with_name(filename))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        transform = getattr(module, function)
        original = files[path]
        transformed = transform(original)
        if len(transformed) != len(original) or transform(transformed, inverse=True) != original:
            raise BuildError('Resource packing failed its exact inverse: ' + path)
        packed[path] = transformed
        encodings[path] = encoding
    return packed, encodings


def _add_file(files: dict[str, bytes], path: str, content: bytes, paths: vendor.ArchivePaths) -> None:
    vendor.reject_native(path, content)
    paths.add(path)
    files[path] = content


def _source_file(path: Path) -> bytes:
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise BuildError(f"Source must be a regular file: {path}")
    return path.read_bytes()


def _reviewed_resources(root: Path) -> dict[str, str]:
    """Read only individually pinned manifests, keeping their data byte exact."""
    standalone = root / "standalone"
    reviewed = dict(PIPER_RESOURCE_SHA256)
    manifests = (("local_piper_voices.py", PIPER_ESPEAK_MANIFEST,
                  PIPER_ESPEAK_MANIFEST_SHA256), *PIPER_REVIEWED_RESOURCE_MANIFESTS)
    paths = vendor.ArchivePaths()
    for module, name, digest in manifests:
        path = standalone / name
        if not path.exists():
            if (standalone / module).exists():
                label = "eSpeak " if name == PIPER_ESPEAK_MANIFEST else ""
                raise BuildError(f"Piper voice configuration requires its reviewed {label}resource manifest")
            continue
        content = _source_file(path)
        if hashlib.sha256(content).hexdigest() != digest:
            raise BuildError("Reviewed Piper resource manifest hash differs: " + name)
        for resource, record in json.loads(content)["resources"].items():
            paths.add("app/standalone/" + resource)
            expected = record["sha256"]
            if resource in reviewed and reviewed[resource] != expected:
                raise BuildError("Conflicting reviewed Piper resource: " + resource)
            reviewed[resource] = expected
        reviewed[name] = digest
    return reviewed


def _walk_sources(directory: Path):
    if directory.is_symlink() or not directory.is_dir():
        raise BuildError(f"Source directory is missing or is a symlink: {directory}")
    for current, dirs, filenames in os.walk(directory, followlinks=False):
        dirs[:] = sorted(
            name for name in dirs
            if name not in EXCLUDED_DIRECTORIES and not name.startswith(".")
        )
        for name in dirs:
            if (Path(current) / name).is_symlink():
                raise BuildError(f"Source directory must not be a symlink: {Path(current) / name}")
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith((".pyc", ".pyo")):
                continue
            yield Path(current) / name


def collect_sources(root: Path, *, variant: str = "full") -> dict[str, bytes]:
    """Apply the explicit distribution include list, preserving module layout."""
    _validate_variant(variant)
    root = Path(root)
    files: dict[str, bytes] = {}
    paths = vendor.ArchivePaths()
    sources = [
        path for path in sorted(root.glob("*.py"))
        if path.name != "setup.py" and not path.name.startswith(("test_", "."))
    ]
    sources.extend(root / name for name in REQUIRED_ROOT_RESOURCES)
    sources.extend(root / name for name in OPTIONAL_ROOT_RESOURCES if (root / name).exists())
    for name in RUNTIME_DIRECTORIES + RESOURCE_DIRECTORIES:
        sources.extend(_walk_sources(root / name))
    standalone = root / "standalone"
    reviewed_resources = _reviewed_resources(root)
    sources.extend(standalone / name for name in reviewed_resources
                   if name not in PIPER_RESOURCE_SHA256 or (standalone / name).exists())
    for path in _walk_sources(standalone):
        relative = path.relative_to(standalone)
        if (
            (path.suffix == ".py" or relative.parts[0] == "licenses" or path.name.endswith("-provenance.json")
             or relative.as_posix() in PIPER_RESOURCE_SHA256
             or relative.as_posix() in ("wake_models/melspectrogram.tflite", "wake_models/embedding_model.tflite",
                                        "assets/silero_vad_v6.onnx"))
            and path.name not in EXCLUDED_STANDALONE_FILES
            and not path.name.startswith("test_")
            and "docs" not in relative.parts
        ):
            sources.append(path)
    sources.extend(standalone / name for name in ("CONTRACT.json", "dependencies.json", "capabilities.json"))
    if (root / SUPPLEMENTAL_LOCKFILE).exists():
        sources.append(root / SUPPLEMENTAL_LOCKFILE)
    for path in sorted(set(sources)):
        relative = path.relative_to(root).as_posix()
        if not _include_variant_path("app/" + relative, variant):
            continue
        content = _source_file(path)
        resource = path.relative_to(standalone).as_posix() if path.is_relative_to(standalone) else None
        if resource in reviewed_resources and hashlib.sha256(content).hexdigest() != reviewed_resources[resource]:
            raise BuildError("Reviewed Piper resource hash differs: " + resource)
        _add_file(files, "app/" + relative, content, paths)
    return files


def _validate_pin_source(root: Path, dependencies: dict, contract: dict) -> None:
    """Detect a dependency manifest detached from its upstream provenance."""
    import tomllib

    source = dependencies.get("source", {})
    if source.get("upstream_commit") != contract["upstream"]["commit"]:
        raise BuildError("Dependency pins and project contract name different upstream commits")
    if source.get("lockfile") != "uv.lock":
        raise BuildError("Dependency pin source must be the upstream uv.lock")
    if source.get("target_python") != contract["runtime"]["hermes_python"]:
        raise BuildError("Dependency pins and project contract name different Python targets")
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked = {(package["name"], package.get("version")): package for package in lock.get("package", [])}
    supplemental = {}
    if source.get("supplemental_lockfile") is not None:
        if source["supplemental_lockfile"] != SUPPLEMENTAL_LOCKFILE:
            raise BuildError("Supplemental dependency pins require the explicit project lockfile")
        extra = json.loads((root / SUPPLEMENTAL_LOCKFILE).read_text(encoding="utf-8"))
        if extra.get("format_version") != 1 or extra.get("authority") != "pypi-release-metadata":
            raise BuildError("Unsupported supplemental dependency lockfile")
        for pin in extra.get("packages", []):
            identity = (pin["name"], pin["version"])
            if identity in supplemental:
                raise BuildError("Duplicate supplemental dependency identity")
            expected_release = f"https://pypi.org/project/{pin['name']}/{pin['version']}/"
            expected_metadata = f"https://pypi.org/pypi/{pin['name']}/{pin['version']}/json"
            if (
                pin.get("release_url") != expected_release or pin.get("metadata_url") != expected_metadata
                or not isinstance(pin.get("metadata_sha256"), str)
                or len(pin["metadata_sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in pin["metadata_sha256"])
                or not pin.get("review_reason")
            ):
                raise BuildError(f"Supplemental release provenance is incomplete: {pin['name']}")
            supplemental[identity] = pin
    for package in dependencies["packages"]:
        original = locked.get((package["name"], package["version"]))
        if original is None:
            pin = supplemental.get((package["name"], package["version"]))
            if package.get("pin_source") != "supplemental-lock" or pin is None:
                raise BuildError(f"Dependency pin is absent from upstream uv.lock: {package['name']}")
            for key in ("filename", "url", "sha256", "size", "dependencies"):
                if package.get(key) != pin.get(key):
                    raise BuildError(f"Supplemental dependency pin differs from its lock: {package['name']}")
            if package.get("kind", "wheel") != pin.get("kind", "wheel"):
                raise BuildError(f"Supplemental archive kind differs from its lock: {package['name']}")
            continue
        if package.get("pin_source", "upstream-lock") != "upstream-lock":
            raise BuildError(f"An upstream dependency cannot bypass uv.lock: {package['name']}")
        candidates = (
            [original["sdist"]] if package.get("kind", "wheel") == "sdist" and "sdist" in original
            else original.get("wheels", []) if package.get("kind", "wheel") == "wheel" else []
        )
        matching = [
            archive for archive in candidates
            if archive.get("url") == package["url"]
            and archive.get("hash") == "sha256:" + package["sha256"]
            and archive.get("size") == package["size"]
        ]
        if not matching:
            raise BuildError(f"Dependency archive pin differs from upstream uv.lock: {package['name']}")
        original_dependencies = {item["name"] for item in original.get("dependencies", [])}
        if set(package.get("dependencies", [])) != original_dependencies:
            raise BuildError(f"Reviewed dependency edges differ from upstream uv.lock: {package['name']}")


def create_archive(
    root: Path, cache_dir: Path, *, compression: str = "lzma", minify: str = "none",
    proof_collector: dict | None = None, variant: str = "full",
) -> tuple[bytes, dict]:
    """Create deterministic ZIP bytes and the manifest without writing output."""
    _validate_variant(variant)
    if compression not in {"lzma", "deflate"}:
        raise BuildError(f"Unsupported compression: {compression}")
    if minify not in {"none", "ast"}:
        raise BuildError(f"Unsupported source transformation: {minify}")
    if proof_collector is not None:
        proof_collector.clear()
    root = Path(root)
    contract = json.loads((root / "standalone" / "CONTRACT.json").read_text(encoding="utf-8"))
    if minify == "ast":
        target = tuple(int(part) for part in contract["runtime"]["hermes_python"].split("."))
        if sys.version_info[:len(target)] < target:
            raise BuildError("AST verification requires the target Hermes Python version or later; "
                             "use --minify none for packaging with an older interpreter")
    dependencies = vendor.load_dependencies(root / "standalone" / "dependencies.json")
    _validate_pin_source(root, dependencies, contract)
    files = collect_sources(root, variant=variant)
    paths = vendor.ArchivePaths()
    for path in files:
        paths.add(path)
    distributions = []
    for package in sorted(dependencies["packages"], key=lambda item: item["name"]):
        for replacement in package.get("replaced_dependencies", {}).values():
            module_path = "app/" + replacement["module"].replace(".", "/")
            if module_path + ".py" not in files and module_path + "/__init__.py" not in files:
                raise BuildError(f"Reviewed dependency replacement is missing: {replacement['module']}")
        package_files, provenance = vendor.read_wheel(package, Path(cache_dir))
        omitted = []
        for path, content in sorted(package_files.items()):
            if not _include_variant_path("vendor/" + path, variant):
                omitted.append({"path": "vendor/" + path, "size": len(content),
                                "sha256": hashlib.sha256(content).hexdigest()})
                continue
            _add_file(files, "vendor/" + path, content, paths)
        if omitted:
            provenance["variant_omitted_resources"] = omitted
        distributions.append(provenance)
    capabilities = _variant_capabilities(
        json.loads((root / "standalone" / "capabilities.json").read_text(encoding="utf-8")), variant,
    )
    if variant == "lite":
        files["app/standalone/capabilities.json"] = canonical_json(capabilities)
    source_transform = {
        "mode": minify, "scope": "app-python-sources", "byte_exact": minify == "none",
    }
    if minify == "ast":
        spec = importlib.util.spec_from_file_location(
            "_hermes_standalone_minify", Path(__file__).with_name("minify.py")
        )
        if spec is None or spec.loader is None:
            raise BuildError("Cannot load standalone/minify.py")
        transformer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer)
        reports = []
        reviewed = _reviewed_resources(root)
        for path, content in sorted(files.items()):
            if path.startswith("app/") and path.endswith(".py"):
                files[path], report = transformer.minify_source(
                    content, path, preserve_exact=path.removeprefix("app/standalone/") in reviewed,
                )
                reports.append({"path": path, **report})
        summary = {
            "files": len(reports),
            "changed": sum(item["status"] == "minified" for item in reports),
            "unchanged": sum(item["status"] != "minified" for item in reports),
            "original_bytes": sum(item["original_size"] for item in reports),
            "output_bytes": sum(item["output_size"] for item in reports),
        }
        reason_counts = {
            reason: sum(item["reason"] == reason for item in reports)
            for reason in sorted({item["reason"] for item in reports})
        }
        toolchain = {
            "tool": transformer._TOOL,
            "proof_scope": transformer._SCOPE,
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "python_implementation": sys.implementation.name,
        }
        proof = {
            "format_version": 1,
            "project": "hermes-standalone",
            "version": contract["version"],
            "upstream": contract["upstream"],
            "mode": minify,
            "scope": source_transform["scope"],
            "toolchain": toolchain,
            "summary": summary,
            "reason_counts": reason_counts,
            "files": reports,
        }
        if proof_collector is not None:
            proof_collector.update(proof)
        source_transform.update({
            "report_sha256": hashlib.sha256(canonical_json(proof)).hexdigest(),
            "toolchain": toolchain,
            "summary": summary,
            "reason_counts": reason_counts,
            "caveat": "AST equivalence does not preserve source text, line locations, or source introspection.",
        })
    packed_models, data_encodings = _pack_data_files(files)
    manifest = {
        "format_version": contract["format_version"],
        "project": "hermes-standalone",
        "version": contract["version"],
        "profile": contract["profile"],
        "build_variant": {
            "name": variant,
            "excluded_features": list(LITE_EXCLUDED_FEATURES) if variant == "lite" else [],
            "generated_files": ["app/standalone/capabilities.json"] if variant == "lite" else [],
        },
        "upstream": contract["upstream"],
        "runtime": contract["runtime"],
        "entrypoint": contract["entrypoint"],
        "internal_modules": list(INTERNAL_MODULES),
        "files": [
            {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content),
             **({"encoding": data_encodings[path]} if path in packed_models else {})}
            for path, content in sorted(files.items())
        ],
        "distributions": distributions,
        "capabilities": capabilities,
        "source_transform": source_transform,
        "source_selection": {
            "root_python": "*.py, except setup.py and test_*.py",
            "runtime_directories": list(RUNTIME_DIRECTORIES),
            "resource_directories": list(RESOURCE_DIRECTORIES),
            "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
            "excluded_standalone_files": sorted(EXCLUDED_STANDALONE_FILES),
        },
    }
    if packed_models:
        mode = 'byteplanes4' if set(data_encodings.values()) == {'byteplanes4'} else 'per-file-lossless'
        manifest['data_transform'] = {'mode': mode, 'byte_exact': True, 'files': len(packed_models),
                                      'encodings': sorted(set(data_encodings.values())),
                                      'proof': 'build inverse equality and extracted original SHA-256'}
    files.update(packed_models)
    files["bundle-manifest.json"] = canonical_json(manifest)
    output = io.BytesIO()
    zip_compression = zipfile.ZIP_STORED if compression == "lzma" else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output, "w", compression=zip_compression, compresslevel=9) as archive:
        for path, content in sorted(files.items()):
            member = zipfile.ZipInfo(path, date_time=ZIP_TIMESTAMP)
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            member.compress_type = zip_compression
            archive.writestr(member, content, compress_type=zip_compression, compresslevel=9)
    return output.getvalue(), manifest


def encode_payload(archive: bytes, version: str, compression: str = "lzma") -> tuple[dict, bytes]:
    info = {
        "format_version": 1,
        "sha256": hashlib.sha256(archive).hexdigest(),
        "size": len(archive),
        "version": version,
    }
    if compression == "deflate":
        return info, archive
    if compression != "lzma":
        raise BuildError(f"Unsupported compression: {compression}")
    try:
        import lzma
    except ImportError:
        raise BuildError(
            "Python's standard-library lzma module is unavailable; use --compression deflate"
        ) from None
    payload = lzma.compress(archive, format=lzma.FORMAT_XZ,
                            filters=[{"id": lzma.FILTER_LZMA2, **LZMA_OPTIONS}])
    info.update({
        "format_version": 2,
        "compression": "lzma",
        "payload_size": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    })
    return info, payload


def _latin1_literal(payload_bytes: bytes) -> str:
    # Keep NUL, universal-newline CR, backslash and delimiter escaping local
    # to each literal. An octal escape can never consume the next chunk.
    payload = payload_bytes.decode("latin-1").replace("\\", "\\\\")
    payload = re.sub("\0(?=[0-7])", lambda _match: "\\x00", payload)
    payload = payload.replace("\0", "\\0").replace("\r", "\\r").replace("'''", "''\\'")
    return "'''\n" + payload + "\n'''"


def _latin1_payload(template: str, info: dict, payload_bytes: bytes, encoding: str):
    if encoding == "latin1-chunks":
        width = 256 * 1024
        literal = "(\n" + "".join(
            _latin1_literal(payload_bytes[start:start + width]) + ",\n"
            for start in range(0, len(payload_bytes), width)
        ) + ")"
        version = 4
    else:
        literal = _latin1_literal(payload_bytes)
        version = 3
    info = {**info, "format_version": version, "payload_encoding": encoding,
            "compression": info.get("compression", "zip"),
            "payload_size": len(payload_bytes),
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest()}
    lines = template.splitlines(keepends=True)
    lines.insert(1 if template.startswith("#!") else 0, "# coding: latin-1\n")
    return "".join(lines), info, literal


def _render_encoded(
    template: str, info: dict, payload_bytes: bytes, *, payload_encoding: str = "base85"
) -> str:
    markers = ("__HERMES_BUNDLE_INFO_LITERAL__", "__HERMES_BUNDLE_PAYLOAD_LITERAL__")
    for marker in markers:
        if template.count(marker) != 1:
            raise BuildError(f"Bootstrap must contain exactly one {marker} placeholder")
    if payload_encoding == "base85":
        payload = base64.b85encode(payload_bytes).decode("ascii")
        literal = "(\n" + "\n".join(
            repr(payload[index:index + PAYLOAD_LITERAL_WIDTH])
            for index in range(0, len(payload), PAYLOAD_LITERAL_WIDTH)
        ) + "\n)"
    elif payload_encoding in ("latin1", "latin1-chunks"):
        template, info, literal = _latin1_payload(template, info, payload_bytes, payload_encoding)
    else:
        raise BuildError(f"Unsupported payload encoding: {payload_encoding}")
    replacements = {markers[0]: ascii(info), markers[1]: literal}
    # Cache ownership must be checked before importing anything from the cache.
    # Embed the small stdlib-only Windows ACL boundary in the launcher itself.
    acl_marker = "__HERMES_WINDOWS_ACL_SOURCE__"
    if acl_marker in template:
        if template.count(acl_marker) != 1:
            raise BuildError(f"Bootstrap must contain exactly one {acl_marker} placeholder")
        replacements[acl_marker] = ascii(Path(__file__).with_name("windows_acl.py").read_text(encoding="utf-8"))
    dictionary_marker = '__HERMES_DICTIONARY_PACK_SOURCE__'
    if dictionary_marker in template:
        if template.count(dictionary_marker) != 1:
            raise BuildError('Bootstrap must contain exactly one dictionary decoder placeholder')
        replacements[dictionary_marker] = Path(__file__).with_name('dictionary_pack.py').read_text(encoding='utf8')
    resource_marker = '__HERMES_RESOURCE_PACK_SOURCE__'
    if resource_marker in template or re.search(r'(?m)^_FILE_CODECS\s*=', template):
        if template.count(resource_marker) != 1:
            raise BuildError('Bootstrap must contain exactly one resource decoder placeholder')
        replacements[resource_marker] = '\n'.join(
            Path(__file__).with_name(name).read_text(encoding='utf8')
            for name in ('mecab_pack.py', 'model_pack.py', 'pickle_pack.py')
        )
    # Replace template tokens once; opaque data may itself contain a token.
    rendered = re.sub("|".join(map(re.escape, replacements)), lambda m: replacements[m[0]], template)
    encoding = "latin-1" if payload_encoding.startswith("latin1") else "utf-8"
    try:
        source = rendered.encode(encoding)
    except UnicodeEncodeError as error:
        raise BuildError("Bootstrap source cannot use Latin-1; select --payload-encoding base85") from error
    compile(source, "<hermes-standalone>", "exec")
    return rendered


def render_artifact(
    template: str, archive: bytes, version: str, *, compression: str = "lzma",
    payload_encoding: str = "base85",
) -> str:
    info, payload = encode_payload(archive, version, compression)
    return _render_encoded(template, info, payload, payload_encoding=payload_encoding)


def _render_zipapp(template: str, info: dict, payload: bytes) -> bytes:
    info = {**info, "format_version": 5, "payload_encoding": "zip-member",
            "compression": info.get("compression", "zip"), "payload_size": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest()}
    source = _render_encoded(template, info, b"", payload_encoding="base85").encode("utf-8")
    if len(source) > 1024 * 1024:
        raise BuildError("ZIP application bootstrap exceeds 1 MiB")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as outer:
        for name, content in (("__main__.py", source), ("payload.bin", payload)):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            outer.writestr(member, content)
    return stream.getvalue()


def render_zipapp(template: str, archive: bytes, version: str, *, compression: str = "lzma") -> bytes:
    info, payload = encode_payload(archive, version, compression)
    return _render_zipapp(template, info, payload)


def _render_container(template, info, payload, container, payload_encoding):
    if container == "zipapp":
        return _render_zipapp(template, info, payload)
    return _render_encoded(template, info, payload, payload_encoding=payload_encoding).encode(
        "latin-1" if payload_encoding.startswith("latin1") else "utf-8"
    )


def build_bundle(
    root: Path, output: Path, cache_dir: Path, *, compression: str = "lzma", minify: str = "none",
    payload_encoding: str = "latin1-chunks", container: str = "source", variant: str = "full",
) -> dict:
    _validate_variant(variant)
    root, output = Path(root), Path(output)
    if output.suffix not in {"source": (".py",), "zipapp": (".py", ".pyz")}.get(container, ()):
        raise BuildError("Source output requires .py; ZIP application output requires .py or .pyz")
    # A generated file inside the source include set would embed itself during
    # the next build or overwrite a real upstream module.
    try:
        relative_output = output.resolve().relative_to(root.resolve())
    except ValueError:
        relative_output = None
    if relative_output is not None and (
        len(relative_output.parts) == 1
        or relative_output.parts[0] in RUNTIME_DIRECTORIES + RESOURCE_DIRECTORIES + ("standalone",)
    ):
        raise BuildError("Artifact output must be outside the source include set; use dist/hermes.py")
    proof = {}
    archive, manifest = create_archive(
        root, cache_dir, compression=compression, minify=minify, proof_collector=proof, variant=variant,
    )
    template = (root / "standalone" / "bootstrap.py").read_text(encoding="utf-8")
    info, payload = encode_payload(archive, manifest["version"], compression)
    rendered = _render_container(template, info, payload, container, payload_encoding)
    output.parent.mkdir(parents=True, exist_ok=True)
    outputs = [(output, rendered, 0o755)]
    if proof:
        report_path = output.with_name(output.name + ".minify.json")
        report_bytes = canonical_json(proof)
        outputs.insert(0, (report_path, report_bytes, 0o600))
    for destination, _, _ in outputs:
        if destination.is_symlink():
            raise BuildError(f"Refusing to overwrite an output symlink: {destination}")
    staged = []
    try:
        # Stage both complete outputs before replacing either. Publish the
        # optional review report first; the runnable artifact never depends on it.
        for destination, content, mode in outputs:
            descriptor, temporary = tempfile.mkstemp(prefix=".hermes-build-", dir=output.parent)
            temporary_path = Path(temporary)
            staged.append((temporary_path, destination))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            temporary_path.chmod(mode)
        for temporary_path, destination in staged:
            temporary_path.replace(destination)
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
    result = {
        "artifact": str(output.resolve()),
        "sha256": hashlib.sha256(rendered).hexdigest(),
        "size": len(rendered),
        "bundle_sha256": hashlib.sha256(archive).hexdigest(),
        "bundle_size": len(archive),
        "compression": compression,
        "payload_encoding": {"source": payload_encoding, "zipapp": "zip-member"}[container],
        "container": container,
        "payload_size": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "minify": minify,
        "files": len(manifest["files"]),
        "distributions": len(manifest["distributions"]),
        "profile": manifest["profile"],
        "variant": variant,
    }
    if proof:
        result.update({
            "minify_report": str(report_path.resolve()),
            "minify_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "minify_report_size": len(report_bytes),
        })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Output file for a single variant")
    parser.add_argument("--output-dir", type=Path, help="Directory for automatically named outputs (default: dist)")
    parser.add_argument(
        "--variant", choices=(*BUILD_VARIANTS, "both"), default="full",
        help="full includes local Piper data; lite omits it and the unused Google discovery catalog",
    )
    parser.add_argument("--cache-dir", type=Path, default=vendor.DEFAULT_CACHE)
    parser.add_argument(
        "--container", choices=("source", "zipapp"), default="source",
        help="Source-text Python by default; zipapp stores the same payload in a single executable ZIP",
    )
    parser.add_argument(
        "--compression", choices=("lzma", "deflate"), default="lzma",
        help="Solid LZMA compression by default; deflate supports Python builds without lzma",
    )
    parser.add_argument(
        "--payload-encoding", choices=("base85", "latin1", "latin1-chunks"), default="latin1-chunks",
        help="Bounded Latin-1 literals reduce compiler memory; base85 keeps the generated file ASCII",
    )
    parser.add_argument(
        "--minify", choices=("none", "ast"), default="none",
        help="Optional AST-verified source rewriting; changes source text and introspection",
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.output_dir is not None:
        parser.error("--output and --output-dir cannot be combined")
    if args.variant == "both" and args.output is not None:
        parser.error("--variant both uses --output-dir rather than a single --output")
    directory = args.output_dir if args.output_dir is not None else DEFAULT_ROOT / "dist"
    variants = BUILD_VARIANTS if args.variant == "both" else (args.variant,)
    suffix = ".pyz" if args.container == "zipapp" else ".py"
    try:
        results = []
        for variant in variants:
            name = "hermes" if args.variant == "full" else "hermes-" + variant
            output = args.output if args.output is not None else directory / (name + suffix)
            results.append(build_bundle(
                DEFAULT_ROOT, output, args.cache_dir,
                compression=args.compression, minify=args.minify, payload_encoding=args.payload_encoding,
                container=args.container, variant=variant,
            ))
        print(json.dumps({"variants": results} if args.variant == "both" else results[0],
                         indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"build: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
