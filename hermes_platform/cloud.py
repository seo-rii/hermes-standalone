"""Select qualified cloud boundaries only for this verified standalone bundle."""

import importlib
import importlib.util
from pathlib import Path


def cloud_api(name):
    choices = {"vercel": ("standalone.cloud_vercel", "vercel.sandbox"),
               "daytona": ("standalone.cloud_daytona", "daytona")}
    if name not in choices:
        raise ValueError("Unknown cloud environment API")
    from standalone import runtime
    selected = choices[name][0 if runtime.is_active_bundle(Path(__file__).resolve().parents[1]) else 1]
    return importlib.import_module(selected)


def bundled_cloud_api(name):
    from standalone import runtime
    return cloud_api(name) if runtime.is_active_bundle(Path(__file__).resolve().parents[1]) else None


def cloud_api_available(name):
    try:
        if bundled_cloud_api(name) is not None:
            return True
        return importlib.util.find_spec("vercel" if name == "vercel" else name) is not None
    except ImportError:
        return False
