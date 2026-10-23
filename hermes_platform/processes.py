"""Select the process implementation owned by this Hermes distribution."""

from pathlib import Path
import sys


def process_api():
    """Native psutil for ordinary installs; OS-backed stdlib operations for bundles.

    This is an explicit Hermes dependency boundary. It never installs a module,
    replaces ``sys.modules['psutil']``, or changes third-party imports.
    """
    runtime = sys.modules.get("standalone.runtime")
    if runtime is not None and runtime.is_active_bundle(Path(__file__).resolve().parents[1]):
        from standalone import processes
        return processes
    import psutil
    return psutil
