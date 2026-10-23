"""Explicit MCP OAuth dependencies at the verified standalone boundary."""

from agent.sdk_boundary import bundled


def oauth_types():
    if bundled():
        from standalone import mcp_oauth_types
        return mcp_oauth_types
    from mcp.shared import auth
    return auth


def oauth_provider():
    if bundled():
        from standalone import mcp_oauth
        return mcp_oauth
    from mcp.client.auth import oauth2
    return oauth2


def oauth_utils():
    if bundled():
        from standalone import mcp_oauth_utils
        return mcp_oauth_utils
    from mcp.client.auth import utils
    return utils


def validation_error():
    if bundled():
        from standalone.mcp_oauth_types import OAuthValidationError
        return OAuthValidationError
    from pydantic import ValidationError
    return ValidationError


def http_url():
    if bundled():
        from standalone.mcp_oauth_types import oauth_url
        return oauth_url
    from pydantic import AnyHttpUrl
    return AnyHttpUrl
