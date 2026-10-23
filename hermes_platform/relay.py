"""Select this distribution's Relay runtime without impersonating the SDK."""

import importlib
from pathlib import Path
import sys


def relay_api():
    runtime = sys.modules.get("standalone.runtime")
    bundled = runtime is not None and runtime.is_active_bundle(Path(__file__).resolve().parents[1])
    return importlib.import_module("standalone.relay" if bundled else "nemo_relay")
