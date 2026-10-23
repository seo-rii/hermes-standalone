"""Select provider clients only for the verified standalone source tree.

Ordinary source installs retain their upstream SDKs. The standalone transports
implement the actual wire boundary without making fake SDK modules visible to
plugins or weakening their model validators.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


def bundled() -> bool:
    runtime = sys.modules.get("standalone.runtime")
    return runtime is not None and runtime.is_active_bundle(Path(__file__).resolve().parents[1])


def openai_client_type(*, asynchronous: bool = False):
    module = openai_clients()
    return module.AsyncOpenAI if asynchronous else module.OpenAI


def openai_clients():
    return importlib.import_module("standalone.transport_chat" if bundled() else "openai")


def openai_error_types(*names: str) -> tuple[type, ...]:
    module = importlib.import_module("standalone.transport_types" if bundled() else "openai")
    return tuple(getattr(module, name) for name in names)
