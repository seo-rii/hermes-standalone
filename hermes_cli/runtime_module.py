"""Launch an original Python module from the currently running installation."""

from pathlib import Path
import sys


def python_module_command(module: str, args=()) -> list[str]:
    runtime = sys.modules.get("standalone.runtime")
    if runtime is not None:
        command = runtime.bundled_command(Path(__file__).resolve().parents[1], module, args)
        if command is not None:
            return command
    return [sys.executable, "-m", module, *args]
