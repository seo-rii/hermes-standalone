"""Select the real Relay implementation belonging to this test distribution."""

import pytest


def relay_or_skip():
    from hermes_platform.relay import relay_api
    try:
        return relay_api()
    except ImportError as error:
        pytest.skip(f"Relay runtime unavailable: {error}", allow_module_level=True)
