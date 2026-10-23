"""Resolve restart entrypoints for the installation running this gateway."""

import importlib.util
from pathlib import Path
import shutil
import sys


def hermes_command() -> list[str] | None:
    runtime = sys.modules.get("standalone.runtime")
    source = Path(__file__).resolve().parents[1]
    if runtime is not None and runtime.is_active_bundle(source):
        return runtime.bundled_command(source, "hermes_cli.main")
    try:
        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass
    command = shutil.which("hermes")
    return [command] if command else None
