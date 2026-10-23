"""OAuth discovery and validation shared by browser and device MCP grants."""

from __future__ import annotations

import json
import re
from urllib.parse import urlsplit, urlunsplit

import httpx

from standalone.mcp_oauth_types import (
    OAuthFlowError, OAuthMetadata, OAuthToken, OAuthTokenError, OAuthValidationError,
    ProtectedResourceMetadata, oauth_url,
)


MAX_OAUTH_DOCUMENT_BYTES = 1024 * 1024
_PARAMETER = re.compile(r'\s*([!#$%&\'*+.^_`|~0-9A-Za-z-]+)\s*=\s*(?:"((?:\\.|[^"\\])*)"|([^,\s]+))\s*(?:,|$)')


def bearer_challenge(response):
    """Parse one Bearer challenge, including quoted commas and backslash escapes."""
    for value in response.headers.get_list("www-authenticate"):
        match = re.search(r"(?:^|,)\s*Bearer(?:\s+|$)", value, re.IGNORECASE)
        if match is None:
            continue
        position, parameters = match.end(), {}
        while position < len(value):
            item = _PARAMETER.match(value, position)
            if item is None:
                break  # another challenge scheme, rather than a Bearer parameter
            key = item[1].lower()
            if key in parameters:
                raise OAuthFlowError("Duplicate OAuth challenge parameter")
            parameters[key] = re.sub(r"\\(.)", r"\1", item[2]) if item[2] is not None else item[3]
            position = item.end()
        return parameters
    return {}


def extract_resource_metadata_from_www_auth(response):
    value = bearer_challenge(response).get("resource_metadata")
    return oauth_url(value) if value is not None else None


def build_protected_resource_metadata_discovery_urls(challenge, server_url):
    parts = urlsplit(oauth_url(server_url))
    origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    result = [oauth_url(challenge)] if challenge else []
    path = parts.path.rstrip("/")
    if path:
        result.append(origin + "/.well-known/oauth-protected-resource" + path)
    result.append(origin + "/.well-known/oauth-protected-resource")
    return list(dict.fromkeys(result))


def build_oauth_authorization_server_metadata_discovery_urls(auth_server_url, server_url):
    parts = urlsplit(oauth_url(auth_server_url or server_url))
    origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    path = parts.path.rstrip("/") if auth_server_url else ""
    result = [origin + "/.well-known/oauth-authorization-server" + path,
              origin + "/.well-known/openid-configuration" + path]
    if path:
        result.append(origin + path + "/.well-known/openid-configuration")
    return result


def create_oauth_metadata_request(url):
    return httpx.Request("GET", oauth_url(url), headers={"Accept": "application/json"})


async def oauth_payload(response, *, label="OAuth"):
    """Read a bounded JSON object, reporting fields/status but never body text."""
    data = bytearray()
    async for chunk in response.aiter_bytes():
        data.extend(chunk)
        if len(data) > MAX_OAUTH_DOCUMENT_BYTES:
            raise OAuthFlowError(label + " document exceeds the size limit")
    # HTTPX's auth driver calls aread() on each intermediate response before
    # sending the next request. Retain the bounded bytes exactly as aread()
    # does, so that subsequent read does not consume an exhausted stream.
    response._content = bytes(data)
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError):
        raise OAuthValidationError(label + " response is not valid JSON") from None
    if not isinstance(value, dict):
        raise OAuthValidationError(label + " response must be a JSON object")
    return value


async def handle_protected_resource_response(response):
    if response.status_code in (404, 405):
        return None
    if not response.is_success:
        raise OAuthFlowError(f"Protected resource metadata failed (HTTP {response.status_code})")
    return ProtectedResourceMetadata.model_validate(await oauth_payload(response, label="Resource metadata"))


def _issuer_for_discovery(url):
    parts = urlsplit(str(url))
    for base in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
        if parts.path == base or parts.path.startswith(base + "/"):
            path = parts.path[len(base):]
            return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
    suffix = "/.well-known/openid-configuration"
    if parts.path.endswith(suffix):
        return urlunsplit((parts.scheme, parts.netloc, parts.path[:-len(suffix)], "", "")).rstrip("/")
    return None


def validate_metadata_issuer(metadata, response, expected=None):
    expected = (expected or _issuer_for_discovery(response.request.url) or "").rstrip("/")
    issuer = str(metadata.issuer).rstrip("/")
    if not expected or issuer == expected:
        return
    # Preserve Hermes' reviewed path-scoped-origin exception. The requested and
    # final URL must both be the derived well-known document at that origin.
    from tools.mcp_oauth_provider import metadata_issued_by_origin
    if str(response.request.url) == str(response.url) and metadata_issued_by_origin(metadata, expected, response):
        return
    raise OAuthFlowError("Authorization server metadata issuer mismatch")


async def handle_auth_metadata_response(response, expected_issuer=None):
    if response.status_code == 404:
        return True, None
    if response.status_code == 204:
        return False, None
    if not response.is_success:
        return True, None
    metadata = OAuthMetadata.model_validate(await oauth_payload(response, label="Authorization metadata"))
    validate_metadata_issuer(metadata, response, expected_issuer)
    return False, metadata


async def handle_token_response_scopes(response):
    if not response.is_success:
        raise OAuthTokenError(f"Token exchange failed (HTTP {response.status_code})")
    try:
        data = await oauth_payload(response, label="Token")
        return OAuthToken.model_validate(data)
    except (OAuthValidationError, OAuthFlowError):
        raise OAuthTokenError("Invalid token response") from None


def is_valid_client_metadata_url(url):
    try:
        parts = urlsplit(oauth_url(url, https_only=True))
    except OAuthValidationError:
        return False
    return bool(parts.path.strip("/")) and not parts.query and not any(piece in {".", ".."} for piece in parts.path.split("/"))
