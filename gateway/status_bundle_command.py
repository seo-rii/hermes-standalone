"""Normalize verified bundle process entries before canonical gateway matching."""

import shlex
import sys


def gateway_process_tokens(command: str | None) -> list[str] | None:
    if not command:
        return None
    try:
        raw = shlex.split(command, posix=False)
    except ValueError:
        raw = command.split()
    tokens = [token.strip("\"'").replace("\\", "/") for token in raw]
    runtime = sys.modules.get("standalone.runtime")
    entry = runtime.bundle_module_argv(tokens) if runtime is not None else None
    if entry is None:
        return tokens
    module, arguments = entry
    if module not in {"hermes_cli.main", "gateway.run"}:
        # A restart watcher merely carries another gateway's argv as data.
        return None
    return [tokens[0], "-m", module, *arguments]
