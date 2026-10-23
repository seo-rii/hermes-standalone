"""Process re-entry helpers for an initialized single-file distribution."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import sys
from types import MappingProxyType


_active_bundle: Mapping[str, object] | None = None


def initialize_bundle(context: Mapping[str, object]) -> None:
    """Bind the already-verified launch to the source tree it actually loaded.

    Integrity verification belongs to the generated bootstrap. This process-local
    binding lets upstream startup recognize that dependency activation is already
    complete; environment variables cannot opt an ordinary checkout into this path.
    """
    global _active_bundle
    root = context.get("root")
    artifact = context.get("artifact")
    digest = context.get("bundle_sha256")
    manifest = context.get("manifest")
    if not (sys.flags.isolated and sys.flags.no_site):
        raise ValueError("bundle initialization requires isolated Python with site disabled")
    if (
        not isinstance(root, str)
        or not Path(root).is_absolute()
        or not isinstance(artifact, str)
        or not Path(artifact).is_absolute()
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or not isinstance(manifest, Mapping)
        or manifest.get("project") != "hermes-standalone"
    ):
        raise ValueError("bundle initialization requires verified distribution context")
    source_root = (Path(root) / "app").resolve()
    if source_root != Path(__file__).resolve().parents[1]:
        raise ValueError("bundle context does not own the loaded runtime source")
    if Path(root).name != digest:
        raise ValueError("bundle cache identity differs from the verified archive")
    cache = context.get("cache_dir", str(Path(root).parent))
    if not isinstance(cache, str) or not Path(cache).is_absolute():
        raise ValueError("bundle cache directory must be absolute")
    binding = {
        "root": str(Path(root).resolve()),
        "cache_dir": str(Path(cache).resolve()),
        "source_root": str(source_root),
        "artifact": str(Path(artifact).resolve()),
        "bundle_sha256": digest,
        "manifest": MappingProxyType({
            "project": "hermes-standalone",
            "internal_modules": tuple(manifest.get("internal_modules", ())),
        }),
    }
    if _active_bundle is not None and dict(_active_bundle) != binding:
        raise ValueError("a process cannot switch between initialized bundles")
    _active_bundle = MappingProxyType(binding)


def is_active_bundle(source_root: str | Path) -> bool:
    """Whether this source tree belongs to the initialized standalone process."""
    return (
        _active_bundle is not None
        and str(Path(source_root).resolve()) == _active_bundle["source_root"]
    )


def current_bundle() -> Mapping[str, object] | None:
    """Immutable launch context for original Hermes process-entry adapters."""
    return _active_bundle


def bundled_command(
    source_root: str | Path, module: str, args: Sequence[str] = (), *,
    home: str | Path | None = None, python: str | Path | None = None,
) -> list[str] | None:
    """Re-enter this artifact, preserving the caller's profile environment contract."""
    if not is_active_bundle(source_root):
        return None
    if home is None:
        command = child_command(_active_bundle, module, args)
    else:
        if module not in _active_bundle["manifest"]["internal_modules"]:
            raise ValueError("child module is not in the bundled allowlist")
        command = child_command(_active_bundle, "standalone.entry_worker", [
            "--module", module, "--home", str(home), "--", *args,
        ])
    if python is not None:
        command[0] = str(python)
    return command


def bundle_module_argv(argv: Sequence[str]) -> tuple[str, list[str]] | None:
    """Recognize this artifact's exact launch format for canonical process identity."""
    if _active_bundle is None or len(argv) < 7 or list(argv[1:6]) != ["-I", "-S", "-B", "-X", "utf8"]:
        return None
    if re.fullmatch(r"python(?:w|\d+(?:\.\d+)*)?(?:\.exe)?", Path(argv[0]).name.lower()) is None:
        return None
    if Path(argv[6]).resolve() != Path(_active_bundle["artifact"]):
        return None
    arguments = list(argv[7:])
    if arguments[:1] == ["--cache-dir"]:
        if len(arguments) < 2:
            return None
        arguments = arguments[2:]
    elif arguments and arguments[0].startswith("--cache-dir="):
        arguments = arguments[1:]
    if arguments[:1] != ["--internal-module"]:
        return "hermes_cli.main", arguments[1:] if arguments[:1] == ["--"] else arguments
    if len(arguments) < 2:
        return None
    module, arguments = arguments[1], arguments[2:]
    if module != "standalone.entry_worker":
        return module, arguments
    if len(arguments) < 5 or arguments[0] != "--module" or arguments[2] != "--home" or arguments[4] != "--":
        return None
    return arguments[1], arguments[5:]


def child_command(
    context: Mapping[str, object],
    module: str,
    args: Sequence[str] = (),
) -> list[str]:
    """Return a shell-free command for an allowlisted bundled Python worker."""
    manifest = context.get("manifest")
    if (
        not isinstance(module, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*", module)
        is None
        or not isinstance(manifest, Mapping)
        or module not in manifest.get("internal_modules", ())
    ):
        raise ValueError("child module is not in the bundled allowlist")
    artifact = context.get("artifact")
    root = context.get("root")
    cache = context.get("cache_dir")
    if cache is None and isinstance(root, str):
        cache = str(Path(root).parent)
    if (
        not isinstance(artifact, str)
        or not Path(artifact).is_absolute()
        or not isinstance(cache, str)
        or not Path(cache).is_absolute()
        or not sys.executable
    ):
        raise ValueError("child command requires initialized bundle paths and Python")
    if isinstance(args, (str, bytes)) or not all(
        isinstance(argument, str) for argument in args
    ):
        raise TypeError("child arguments must be a sequence of strings")
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        artifact,
        "--cache-dir",
        cache,
        "--internal-module",
        module,
        *args,
    ]
