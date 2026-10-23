"""Select the Windows PTY backend belonging to this distribution."""

from pathlib import Path
import sys


def windows_pty_process():
    runtime = sys.modules.get("standalone.runtime")
    if runtime is not None and runtime.is_active_bundle(Path(__file__).resolve().parents[1]):
        from standalone.windows_pty import PtyProcess
    else:
        from winpty import PtyProcess
    return PtyProcess
