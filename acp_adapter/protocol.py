"""Select the original ACP SDK or its explicitly private stdlib port."""

from pathlib import Path

from standalone.runtime import is_active_bundle


if is_active_bundle(Path(__file__).resolve().parents[1]):
    from acp_adapter import _protocol as sdk
    from acp_adapter._protocol import schema
    from acp_adapter._protocol.exceptions import RequestError
    from standalone.web_schema import ValidationError
else:
    import acp as sdk
    from acp import schema
    from acp.exceptions import RequestError
    from pydantic import ValidationError


def __getattr__(name):
    return getattr(schema, name)
