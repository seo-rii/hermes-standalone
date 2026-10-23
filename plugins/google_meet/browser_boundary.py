"""Select the application's CDP boundary or the source install's real SDK."""

from __future__ import annotations

import importlib.util


class _SourceChromium:
    def __init__(self, manager):
        self.manager = manager

    def __enter__(self):
        return self.manager.__enter__().chromium

    def __exit__(self, *args):
        return self.manager.__exit__(*args)


def chromium():
    from agent.sdk_boundary import bundled
    if bundled():
        from standalone.meet_browser import Chromium
        return Chromium()
    from playwright.sync_api import sync_playwright
    return _SourceChromium(sync_playwright())


def available():
    from agent.sdk_boundary import bundled
    if bundled():
        from standalone.meet_browser import CDPError, find_executable
        try:
            find_executable()
            return True
        except (CDPError, OSError, ValueError):
            return False
    return importlib.util.find_spec("playwright") is not None
