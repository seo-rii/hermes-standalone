"""Validated OAuth wire records used only by the bundled MCP client.

The original Hermes token store and consent handlers retain ownership of
credentials and browser interaction. These records validate that boundary;
they are not a Pydantic replacement or an injected MCP SDK module.
"""

from __future__ import annotations

import copy
import ipaddress
import json
from urllib.parse import urlsplit

from standalone.transport_types import WireObject


class OAuthValidationError(ValueError):
    """A field-only validation error that never echoes credential input."""


class OAuthFlowError(RuntimeError):
    pass


class OAuthTokenError(OAuthFlowError):
    pass


class OAuthRegistrationError(OAuthFlowError):
    pass


def oauth_url(value, *, https_only=False):
    if not isinstance(value, str) or not value or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        raise OAuthValidationError("Invalid OAuth URL")
    try:
        parts = urlsplit(value)
        _port = parts.port
    except ValueError:
        raise OAuthValidationError("Invalid OAuth URL") from None
    if (parts.scheme not in {"https", "http"} or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.fragment or "\\" in value):
        raise OAuthValidationError("Invalid OAuth URL")
    if parts.scheme == "http":
        try:
            loopback = ipaddress.ip_address(parts.hostname).is_loopback
        except ValueError:
            loopback = parts.hostname.lower() == "localhost"
        if https_only or not loopback:
            raise OAuthValidationError("OAuth HTTP URLs require a loopback host")
    return value


class OAuthRecord(WireObject):
    DEFAULTS = {}
    REQUIRED = ()
    STRINGS = ()
    URLS = ()
    LISTS = ()
    INTEGERS = ()
    BOOLEANS = ()

    def __init__(self, value=None, **kwargs):
        if value is not None and not isinstance(value, dict):
            raise OAuthValidationError("OAuth document must be an object")
        data = {**copy.deepcopy(self.DEFAULTS), **(value or {}), **kwargs}
        for name in self.REQUIRED:
            if name not in data or data[name] is None or data[name] == "":
                raise OAuthValidationError("Missing OAuth field: " + name)
        for name in self.STRINGS:
            if data.get(name) is not None and not isinstance(data[name], str):
                raise OAuthValidationError("Invalid OAuth string field: " + name)
        for name in self.URLS:
            if data.get(name) is not None:
                try:
                    oauth_url(data[name])
                except OAuthValidationError:
                    raise OAuthValidationError("Invalid OAuth URL field: " + name) from None
        for name in self.LISTS:
            values = data.get(name)
            if values is not None and (not isinstance(values, list) or any(not isinstance(item, str) for item in values)):
                raise OAuthValidationError("Invalid OAuth string list: " + name)
        for name in self.INTEGERS:
            if data.get(name) is not None and type(data[name]) is not int:
                raise OAuthValidationError("Invalid OAuth integer field: " + name)
        for name in self.BOOLEANS:
            if data.get(name) is not None and type(data[name]) is not bool:
                raise OAuthValidationError("Invalid OAuth boolean field: " + name)
        super().__init__(data)

    @classmethod
    def model_validate(cls, data):
        return cls(data)

    @classmethod
    def model_validate_json(cls, data):
        try:
            value = json.loads(data)
        except (ValueError, TypeError, UnicodeError):
            raise OAuthValidationError("Invalid OAuth JSON document") from None
        return cls(value)

    def model_copy(self, *, update=None, deep=False):
        data = copy.deepcopy(dict(self)) if deep else dict(self)
        data.update(update or {})
        return type(self)(data)


class OAuthToken(OAuthRecord):
    DEFAULTS = {"token_type": "Bearer", "expires_in": None, "scope": None, "refresh_token": None}
    REQUIRED = ("access_token",)
    STRINGS = ("access_token", "token_type", "scope", "refresh_token")
    INTEGERS = ("expires_in",)

    def __init__(self, value=None, **kwargs):
        super().__init__(value, **kwargs)
        if self.token_type.lower() != "bearer":
            raise OAuthValidationError("Unsupported OAuth token_type")
        if any(ord(char) < 33 or ord(char) > 126 for char in self.access_token):
            raise OAuthValidationError("Invalid OAuth access_token field")


class OAuthClientMetadata(OAuthRecord):
    DEFAULTS = {"redirect_uris": [], "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"], "token_endpoint_auth_method": "none", "scope": None,
                "client_name": None, "application_type": "native"}
    STRINGS = ("scope", "client_name", "application_type", "token_endpoint_auth_method")
    LISTS = ("redirect_uris", "grant_types", "response_types")

    def __init__(self, value=None, **kwargs):
        super().__init__(value, **kwargs)
        for uri in self.redirect_uris:
            oauth_url(uri)


class OAuthClientInformationFull(OAuthClientMetadata):
    DEFAULTS = {**OAuthClientMetadata.DEFAULTS, "client_secret": None, "issuer": None,
                "client_id_issued_at": None, "client_secret_expires_at": None}
    REQUIRED = ("client_id",)
    STRINGS = (*OAuthClientMetadata.STRINGS, "client_id", "client_secret", "issuer")
    INTEGERS = ("client_id_issued_at", "client_secret_expires_at")


class OAuthMetadata(OAuthRecord):
    DEFAULTS = {"authorization_endpoint": None, "registration_endpoint": None,
                "scopes_supported": None, "response_types_supported": None,
                "grant_types_supported": None, "token_endpoint_auth_methods_supported": None,
                "code_challenge_methods_supported": None,
                "authorization_response_iss_parameter_supported": False,
                "client_id_metadata_document_supported": False,
                "device_authorization_endpoint": None}
    REQUIRED = ("issuer", "token_endpoint")
    URLS = ("issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint", "device_authorization_endpoint")
    LISTS = ("scopes_supported", "response_types_supported", "grant_types_supported",
             "token_endpoint_auth_methods_supported", "code_challenge_methods_supported")
    BOOLEANS = ("authorization_response_iss_parameter_supported", "client_id_metadata_document_supported")


class ProtectedResourceMetadata(OAuthRecord):
    DEFAULTS = {"authorization_servers": [], "scopes_supported": None}
    REQUIRED = ("resource",)
    URLS = ("resource",)
    LISTS = ("authorization_servers", "scopes_supported")

    def __init__(self, value=None, **kwargs):
        super().__init__(value, **kwargs)
        for uri in self.authorization_servers:
            oauth_url(uri)


class AuthorizationCodeResult(OAuthRecord):
    DEFAULTS = {"state": None, "iss": None}
    REQUIRED = ("code",)
    STRINGS = ("code", "state", "iss")
