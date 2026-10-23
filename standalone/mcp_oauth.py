"""HTTPX OAuth provider for Hermes' existing MCP authorization policy.

This implements RFC 8414/9728 discovery, preregistration/CIMD/DCR, authorization
code + S256 PKCE, resource binding and refresh. Hermes' original provider
mixins still own consent, profile storage, refresh fencing and issuer checks.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import time
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

import httpx

from standalone.mcp_oauth_types import (
    AuthorizationCodeResult, OAuthClientInformationFull, OAuthClientMetadata,
    OAuthFlowError, OAuthMetadata, OAuthRegistrationError, OAuthToken,
    OAuthTokenError, OAuthValidationError, oauth_url,
)
from standalone.mcp_oauth_utils import (
    bearer_challenge, build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls, create_oauth_metadata_request,
    handle_auth_metadata_response, handle_protected_resource_response,
    handle_token_response_scopes, is_valid_client_metadata_url, oauth_payload,
)


def check_registration_usable(info):
    method = info.token_endpoint_auth_method or "none"
    if method not in {"none", "client_secret_basic", "client_secret_post"}:
        raise OAuthRegistrationError("Unsupported OAuth token endpoint authentication method")
    if method != "none" and not info.client_secret:
        raise OAuthRegistrationError("OAuth client authentication requires a client secret")
    expiry = info.client_secret_expires_at
    if expiry and expiry <= time.time():
        raise OAuthRegistrationError("OAuth client secret has expired")


def _origin(parts):
    return (parts.scheme.lower(), parts.hostname.lower(), parts.port or (443 if parts.scheme.lower() == "https" else 80))


class OAuthContext:
    def __init__(self, server_url, client_metadata, storage, redirect_handler, callback_handler, client_metadata_url):
        self.server_url = oauth_url(str(server_url))
        self.client_metadata = OAuthClientMetadata.model_validate(client_metadata)
        self.storage, self.redirect_handler, self.callback_handler = storage, redirect_handler, callback_handler
        self.client_metadata_url = client_metadata_url
        self.current_tokens = self.client_info = self.oauth_metadata = None
        self.protected_resource_metadata = self.auth_server_url = self.token_expiry_time = None
        self.lock = asyncio.Semaphore(1)

    def update_token_expiry(self, tokens):
        expires = tokens.expires_in if tokens is not None else None
        self.token_expiry_time = time.time() + expires if expires is not None else None

    def is_token_valid(self):
        return bool(self.current_tokens and self.current_tokens.access_token
                    and (self.token_expiry_time is None or time.time() < self.token_expiry_time))

    def can_refresh_token(self):
        return bool(self.current_tokens and self.current_tokens.refresh_token and self.client_info)

    def clear_tokens(self):
        self.current_tokens = self.token_expiry_time = None

    def get_resource_url(self):
        parts = urlsplit(self.server_url)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))

    @staticmethod
    def get_authorization_base_url(server_url):
        parts = urlsplit(server_url)
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))

    def prepare_token_auth(self, data, headers):
        info = self.client_info
        check_registration_usable(info)
        data, headers = dict(data), dict(headers)
        data.setdefault("client_id", info.client_id)
        if info.token_endpoint_auth_method == "client_secret_post":
            data["client_secret"] = info.client_secret
        elif info.token_endpoint_auth_method == "client_secret_basic":
            pair = quote_plus(info.client_id) + ":" + quote_plus(info.client_secret)
            headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode("ascii")
        return data, headers


class OAuthClientProvider(httpx.Auth):
    requires_request_body = True
    # Resource responses can be an indefinite SSE connection. Read only OAuth
    # metadata/token bodies, never every resource response in the auth driver.
    requires_response_body = False

    def __init__(self, server_url, client_metadata, storage, redirect_handler=None,
                 callback_handler=None, client_metadata_url=None):
        if client_metadata_url is not None and not is_valid_client_metadata_url(client_metadata_url):
            raise OAuthValidationError("Invalid OAuth client metadata URL")
        self.context = OAuthContext(server_url, client_metadata, storage, redirect_handler,
                                    callback_handler, client_metadata_url)
        self._initialized = False

    async def _initialize(self):
        self.context.client_info = await self.context.storage.get_client_info()
        self.context.current_tokens = await self.context.storage.get_tokens()
        self.context.update_token_expiry(self.context.current_tokens)
        self._initialized = True

    def _add_auth_header(self, request):
        if self.context.is_token_valid():
            request.headers["Authorization"] = "Bearer " + self.context.current_tokens.access_token
        else:
            request.headers.pop("Authorization", None)

    async def _validate_resource_match(self, metadata):
        supplied, expected = urlsplit(metadata.resource), urlsplit(self.context.get_resource_url())
        if ((_origin(supplied), supplied.path.rstrip("/"), supplied.query)
                != (_origin(expected), expected.path.rstrip("/"), expected.query)):
            raise OAuthFlowError("OAuth protected resource does not match the configured MCP server")

    def _token_request(self, data):
        context = self.context
        endpoint = (context.oauth_metadata.token_endpoint if context.oauth_metadata is not None
                    else context.get_authorization_base_url(context.server_url) + "/token")
        data, headers = context.prepare_token_auth({**data, "resource": context.get_resource_url()}, {})
        return httpx.Request("POST", oauth_url(str(endpoint)), data=data, headers=headers,
                             extensions={"hermes_oauth_no_redirect": True})

    async def _refresh_token(self):
        if not self.context.can_refresh_token():
            raise OAuthTokenError("No refresh grant is available")
        return self._token_request({"grant_type": "refresh_token", "refresh_token": self.context.current_tokens.refresh_token})

    async def _exchange_token_authorization_code(self, code, verifier):
        return self._token_request({"grant_type": "authorization_code", "code": code,
                                    "code_verifier": verifier, "redirect_uri": self.context.client_metadata.redirect_uris[0]})

    async def _store_tokens(self, tokens):
        self.context.current_tokens = tokens
        self.context.update_token_expiry(tokens)
        await self.context.storage.set_tokens(tokens)

    async def _handle_token_response(self, response):
        await self._store_tokens(await handle_token_response_scopes(response))

    async def _handle_refresh_response(self, response):
        if not response.is_success:
            self.context.clear_tokens()
            return False
        tokens = await handle_token_response_scopes(response)
        prior = self.context.current_tokens
        if prior is not None:
            tokens.refresh_token = tokens.refresh_token if tokens.refresh_token is not None else prior.refresh_token
            tokens.scope = tokens.scope if tokens.scope is not None else prior.scope
        await self._store_tokens(tokens)
        return True

    async def _perform_authorization(self):
        context = self.context
        if context.redirect_handler is None or context.callback_handler is None:
            raise OAuthFlowError("OAuth requires an explicit authorization callback")
        metadata = context.oauth_metadata
        if not context.client_metadata.redirect_uris:
            raise OAuthRegistrationError("OAuth client has no redirect URI")
        endpoint = (metadata.authorization_endpoint if metadata is not None
                    else context.get_authorization_base_url(context.server_url) + "/authorize")
        if not endpoint:
            raise OAuthFlowError("Authorization server does not advertise a browser authorization endpoint")
        methods = metadata.code_challenge_methods_supported if metadata is not None else None
        if methods is not None and "S256" not in methods:
            raise OAuthFlowError("Authorization server does not support S256 PKCE")
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode().rstrip("=")
        parameters = {"response_type": "code", "client_id": context.client_info.client_id,
                      "redirect_uri": context.client_metadata.redirect_uris[0], "state": state,
                      "code_challenge": challenge, "code_challenge_method": "S256", "resource": context.get_resource_url()}
        if context.client_metadata.scope:
            parameters["scope"] = context.client_metadata.scope
        parts = urlsplit(oauth_url(str(endpoint)))
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.update(parameters)
        await context.redirect_handler(urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), "")))
        result = await context.callback_handler()
        if isinstance(result, tuple) and len(result) == 2:
            result = AuthorizationCodeResult(code=result[0], state=result[1])
        else:
            result = AuthorizationCodeResult.model_validate(result)
        if result.state is None or not secrets.compare_digest(result.state, state):
            raise OAuthFlowError("OAuth authorization state mismatch")
        issuer = str(metadata.issuer).rstrip("/") if metadata is not None else None
        if result.iss is not None and result.iss.rstrip("/") != issuer:
            raise OAuthFlowError("OAuth authorization response issuer mismatch")
        if metadata is not None and metadata.authorization_response_iss_parameter_supported and result.iss is None:
            raise OAuthFlowError("OAuth authorization response omitted the required issuer")
        return result.code, verifier

    async def _authorization_requests(self, challenge_response):
        context = self.context
        challenge = bearer_challenge(challenge_response)
        for url in build_protected_resource_metadata_discovery_urls(challenge.get("resource_metadata"), context.server_url):
            response = yield create_oauth_metadata_request(url)
            resource = await handle_protected_resource_response(response)
            if resource is not None:
                await self._validate_resource_match(resource)
                context.protected_resource_metadata = resource
                discovered = resource.authorization_servers[0] if resource.authorization_servers else None
                if discovered != context.auth_server_url:
                    context.oauth_metadata = None
                context.auth_server_url = discovered
                break
        for url in build_oauth_authorization_server_metadata_discovery_urls(context.auth_server_url, context.server_url):
            response = yield create_oauth_metadata_request(url)
            again, metadata = await handle_auth_metadata_response(response, context.auth_server_url)
            if metadata is not None:
                context.oauth_metadata = metadata
                break
            if not again:
                break
        if context.auth_server_url and context.oauth_metadata is None:
            raise OAuthFlowError("Could not read metadata for the advertised authorization server")
        scope = challenge.get("scope")
        if scope is None:
            scope = context.client_metadata.scope
        if scope is None and context.protected_resource_metadata is not None:
            scopes = context.protected_resource_metadata.scopes_supported
            scope = " ".join(scopes) if scopes is not None else None
        context.client_metadata.scope = scope
        metadata = context.oauth_metadata
        issuer = str(context.auth_server_url or (metadata.issuer if metadata else context.get_authorization_base_url(context.server_url)))
        info = context.client_info
        if info is not None and info.issuer and info.issuer.rstrip("/") != issuer.rstrip("/"):
            context.client_info = None
            context.clear_tokens()
            info = None
        if info is None:
            data = context.client_metadata.model_dump(exclude_none=True)
            if context.client_metadata_url and metadata is not None and metadata.client_id_metadata_document_supported:
                data.update(client_id=context.client_metadata_url, token_endpoint_auth_method="none")
            else:
                endpoint = metadata.registration_endpoint if metadata is not None else context.get_authorization_base_url(context.server_url) + "/register"
                if not endpoint:
                    raise OAuthRegistrationError("Authorization server has no registration endpoint; configure oauth.client_id")
                response = yield httpx.Request("POST", oauth_url(str(endpoint)), json=data,
                                               extensions={"hermes_oauth_no_redirect": True})
                if not response.is_success:
                    raise OAuthRegistrationError(f"OAuth registration failed (HTTP {response.status_code})")
                data = {**data, **await oauth_payload(response, label="Registration")}
            data["issuer"] = issuer
            info = context.client_info = OAuthClientInformationFull.model_validate(data)
        if not info.issuer:
            info.issuer = issuer
        check_registration_usable(info)
        if any(uri not in info.redirect_uris for uri in context.client_metadata.redirect_uris):
            raise OAuthRegistrationError("OAuth registration did not retain the configured redirect URI")
        await context.storage.set_client_info(info)
        code, verifier = await self._perform_authorization()
        response = yield await self._exchange_token_authorization_code(code, verifier)
        await self._handle_token_response(response)

    async def async_auth_flow(self, request):
        context = self.context
        requested = urlsplit(str(request.url))
        resource = urlsplit(context.server_url)
        if _origin(requested) != _origin(resource):
            raise OAuthFlowError("OAuth provider cannot authorize a different resource origin")
        async with context.lock:
            if not self._initialized:
                await self._initialize()
            refreshed = False
            if not context.is_token_valid() and context.can_refresh_token():
                response = yield await self._refresh_token()
                await self._handle_refresh_response(response)
                refreshed = True
            seen_scopes = set()
            for _attempt in range(3):
                self._add_auth_header(request)
                response = yield request
                challenge = bearer_challenge(response)
                if response.status_code != 401 and not (response.status_code == 403 and challenge.get("error") == "insufficient_scope"):
                    return
                await response.aread()
                challenge_response = response
                if response.status_code == 401 and context.can_refresh_token() and not refreshed:
                    response = yield await self._refresh_token()
                    refreshed = True
                    if await self._handle_refresh_response(response):
                        continue
                scope = challenge.get("scope")
                if scope in seen_scopes:
                    raise OAuthFlowError("MCP authorization was rejected after a completed grant")
                seen_scopes.add(scope)
                flow = self._authorization_requests(challenge_response)
                incoming = None
                try:
                    while True:
                        outgoing = await flow.asend(incoming)
                        incoming = yield outgoing
                except StopAsyncIteration:
                    pass
                finally:
                    await flow.aclose()
            raise OAuthFlowError("MCP authorization retry limit exceeded")
