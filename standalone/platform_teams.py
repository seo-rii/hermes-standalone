"""Qualified Bot Connector boundary for Hermes' existing Teams adapter.

The consumed activity, card, OAuth and JWT contracts were reviewed against
microsoft-teams-apps 2.0.13.4 and microsoft-teams-api/cards 2.0.16, and:
https://learn.microsoft.com/azure/bot-service/rest-api/bot-framework-rest-connector-authentication
https://learn.microsoft.com/azure/bot-service/rest-api/bot-framework-rest-connector-send-and-receive-messages

Hermes retains its webhook listener, admission/dedup, attachment and approval
logic. This module supplies its private protocol records and HTTP transport;
it never installs aliases for microsoft_teams, Pydantic, or dependency-injector.

Reviewed wheel SHA256 values from uv.lock:
apps: db16f714ec658b592929c6386a29792e90bb73840732f8ae65a198cda1fea96c
api: 96b3a55420aadaaeed6d33a6be2c406bdb0cd4b8246b56de587ba3ed322cff35
cards: 2c7e909fd805d67656a24ef54b1d5f23e221666417a50cbe39bb9b973796c3e8
"""

import asyncio
from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Callable, Literal
from urllib.parse import quote, urlsplit

from standalone.web_schema import RequestModel as BaseModel, ConfigDict, Field


JWKS_URL = "https://login.botframework.com/v1/.well-known/keys"
TOKEN_ISSUER = "https://api.botframework.com"
AUTHORITY_URL = "https://login.microsoftonline.com"
SERVICE_URL = "https://smba.trafficmanager.net/teams/"
TOKEN_SCOPE = "https://api.botframework.com/.default"


class _Record(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, strict=True)


class Account(_Record):
    id: str
    name: str | None = None
    aad_object_id: str | None = Field(None, alias="aadObjectId")
    is_targeted: bool | None = Field(None, alias="isTargeted")


class ConversationAccount(_Record):
    id: str
    name: str | None = None
    conversation_type: str | None = Field(None, alias="conversationType")
    tenant_id: str | None = Field(None, alias="tenantId")
    is_group: bool | None = Field(None, alias="isGroup")


class Entity(_Record):
    type: str
    mentioned: Account | None = None


class Attachment(_Record):
    content_type: str = Field(alias="contentType")
    content_url: str | None = Field(None, alias="contentUrl")
    content: Any = None
    name: str | None = None


class _Action(_Record):
    type: str = "Action.Execute"
    data: dict[str, Any] | None = None
    verb: str | None = None


class _ActionValue(_Record):
    action: _Action


class MessageActivity(_Record):
    type: str
    id: str | None = None
    service_url: str = Field(alias="serviceUrl")
    channel_id: str = Field(alias="channelId")
    from_: Account = Field(alias="from")
    recipient: Account
    conversation: ConversationAccount
    text: str | None = None
    locale: str | None = None
    reply_to_id: str | None = Field(None, alias="replyToId")
    attachments: list[Attachment] | None = None
    entities: list[Entity] | None = None
    name: str | None = None
    value: _ActionValue | None = None
    channel_data: dict[str, Any] | None = Field(None, alias="channelData")


class AdaptiveCardInvokeActivity(MessageActivity):
    type: Literal["invoke"] = "invoke"
    name: Literal["adaptiveCard/action"] = "adaptiveCard/action"
    value: _ActionValue


class ConversationReference(_Record):
    bot: Account
    conversation: ConversationAccount
    channel_id: str = Field(alias="channelId")
    service_url: str = Field(alias="serviceUrl")
    user: Account | None = None
    locale: str | None = None
    activity_id: str | None = Field(None, alias="activityId")


class TextBlock(_Record):
    type: Literal["TextBlock"] = "TextBlock"
    text: str
    wrap: bool = False
    weight: str | None = None


class ExecuteAction(_Record):
    type: Literal["Action.Execute"] = "Action.Execute"
    title: str | None = None
    data: dict[str, Any] | None = None
    verb: str | None = None
    style: str | None = None


class AdaptiveCard(_Record):
    type: Literal["AdaptiveCard"] = "AdaptiveCard"
    schema: str = Field("http://adaptivecards.io/schemas/adaptive-card.json", alias="$schema")
    version: str = "1.5"
    body: list[Any] = Field(default_factory=list)
    actions: list[Any] = Field(default_factory=list)

    def with_version(self, version):
        self.version = version
        return self

    def with_body(self, body):
        self.body = list(body)
        return self

    def with_actions(self, actions):
        self.actions = list(actions)
        return self


class AdaptiveCardActionMessageResponse(_Record):
    status_code: Literal[200] = Field(200, alias="statusCode")
    type: Literal["application/vnd.microsoft.activity.message"] = "application/vnd.microsoft.activity.message"
    value: str


class AdaptiveCardActionCardResponse(_Record):
    status_code: Literal[200] = Field(200, alias="statusCode")
    type: Literal["application/vnd.microsoft.card.adaptive"] = "application/vnd.microsoft.card.adaptive"
    value: AdaptiveCard


class InvokeResponse(_Record):
    status: int = 200
    body: Any = None

    @classmethod
    def __class_getitem__(cls, _annotation):
        return cls


AdaptiveCardInvokeResponse = AdaptiveCardActionMessageResponse | AdaptiveCardActionCardResponse


class MessageActivityInput(_Record):
    type: Literal["message"] = "message"
    text: str | None = None
    id: str | None = None
    from_: Account | None = Field(None, alias="from")
    recipient: Account | None = None
    conversation: ConversationAccount | None = None
    attachments: list[Attachment] | None = None
    reply_to_id: str | None = Field(None, alias="replyToId")

    def add_text(self, text):
        self.text = (self.text or "") + text
        return self

    def add_attachments(self, *attachments):
        self.attachments = [*(self.attachments or []), *attachments]
        return self

    def add_card(self, card):
        return self.add_attachments(Attachment(content_type="application/vnd.microsoft.card.adaptive", content=card))


class TypingActivityInput(_Record):
    type: Literal["typing"] = "typing"
    from_: Account | None = Field(None, alias="from")
    conversation: ConversationAccount | None = None


class _SentActivity(_Record):
    id: str


@dataclass
class ClientOptions:
    headers: dict[str, str] = field(default_factory=dict)


HttpRequest = HttpResponse = dict
HttpMethod = str
HttpRouteHandler = Callable


def _plain(value):
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True, exclude_none=True, mode="json")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _https_url(value):
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Bot Connector endpoints must use HTTPS without user information or fragments")
    return value.rstrip("/")


class ActivityContext:
    def __init__(self, app, activity):
        self.activity, self.app_id = activity, app.id
        self.conversation_ref = ConversationReference(bot=activity.recipient, user=activity.from_,
            conversation=activity.conversation, channel_id=activity.channel_id, service_url=activity.service_url,
            activity_id=activity.id, locale=activity.locale)
        self._app = app
        self.storage = app.storage

    async def send(self, message, conversation_ref=None):
        return await self._app.activity_sender.send(message, conversation_ref or self.conversation_ref)

    async def reply(self, message):
        activity = self._app._activity(message)
        if isinstance(activity, MessageActivityInput):
            activity.reply_to_id = self.activity.id
        return await self.send(activity)


class _ActivitySender:
    def __init__(self, app):
        self.app = app

    async def send(self, activity, reference):
        import httpx

        activity = self.app._activity(activity)
        activity.from_, activity.conversation = reference.bot, reference.conversation
        service = _https_url(reference.service_url)
        conversation = quote(reference.conversation.id, safe="@:-_.;=")
        url = service + "/v3/conversations/" + conversation + "/activities"
        existing = getattr(activity, "id", None)
        method = "PUT" if existing else "POST"
        if existing:
            url += "/" + quote(existing, safe="")
        if getattr(getattr(activity, "recipient", None), "is_targeted", False):
            if reference.conversation.conversation_type == "personal":
                raise ValueError("Targeted messages are not supported in 1:1 (personal) chats.")
            url += "?isTargetedActivity=true"
        headers = {**self.app.client.headers, "Authorization": "Bearer " + await self.app._token()}
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            response = await client.request(method, url, headers=headers, json=_plain(activity))
            response.raise_for_status()
            value = response.json() if response.content else {}
        return _SentActivity(id=value.get("id", "DO_NOT_USE_PLACEHOLDER_ID"))


class App:
    """Hermes' consumed Teams protocol surface, with authenticated webhook dispatch."""

    def __init__(self, *, client_id, client_secret, tenant_id, http_server_adapter, client=None):
        import jwt

        self.id, self._secret, self._tenant = client_id, client_secret, tenant_id
        self.client = client or ClientOptions()
        self._bridge = http_server_adapter
        self._jwk = jwt.PyJWKClient(JWKS_URL)
        self._handlers = {}
        self._token_value, self._token_expires = None, 0
        self._token_lock = asyncio.Lock()
        self._initialized = False
        self.storage = {}
        self.activity_sender = _ActivitySender(self)

    def on_message(self, handler):
        self._handlers["message"] = handler
        return handler

    def on_card_action(self, handler):
        self._handlers["adaptiveCard/action"] = handler
        return handler

    async def initialize(self):
        if not self._initialized:
            self._bridge.register_route("POST", "/api/messages", self._request)
            self._initialized = True

    async def _request(self, request):
        import jwt

        body, headers = request["body"], request["headers"]
        authorization = headers.get("authorization") or headers.get("Authorization") or ""
        if not authorization.startswith("Bearer "):
            return {"status": 401, "body": {"error": "Unauthorized"}}
        try:
            token = authorization.removeprefix("Bearer ")
            key = await asyncio.to_thread(self._jwk.get_signing_key_from_jwt, token)
            claims = jwt.decode(token, key.key, algorithms=["RS256"], issuer=TOKEN_ISSUER,
                audience=[self.id, "api://" + self.id, "api://botid-" + self.id], leeway=300,
                options={"verify_signature": True, "verify_aud": True, "verify_iss": True,
                         "verify_exp": True, "verify_iat": True})
            service = body.get("serviceUrl")
            if service:
                if not isinstance(service, str) or not isinstance(claims.get("serviceurl"), str) or claims["serviceurl"].rstrip("/") != service.rstrip("/"):
                    raise jwt.InvalidTokenError("Service URL claim mismatch")
                _https_url(service)
        except Exception:
            return {"status": 401, "body": {"error": "Unauthorized"}}
        try:
            model = AdaptiveCardInvokeActivity if body.get("type") == "invoke" and body.get("name") == "adaptiveCard/action" else MessageActivity
            activity = model.model_validate(body)
            handler = self._handlers.get(activity.name if activity.type == "invoke" else activity.type)
            result = handler(ActivityContext(self, activity)) if handler else None
            if inspect.isawaitable(result):
                result = await result
            result = _plain(result) if result else {}
            return {"status": result.get("status", 200), "body": result.get("body")}
        except Exception:
            return {"status": 500, "body": {"error": "Internal server error"}}

    async def _token(self):
        import httpx

        async with self._token_lock:
            if self._token_value and self._token_expires > time.monotonic() + 300:
                return self._token_value
            authority = _https_url(AUTHORITY_URL)
            url = authority + "/" + quote(self._tenant, safe="") + "/oauth2/v2.0/token"
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
                response = await client.post(url, data={"grant_type": "client_credentials", "client_id": self.id,
                    "client_secret": self._secret, "scope": TOKEN_SCOPE})
                response.raise_for_status()
                value = response.json()
            token = value.get("access_token")
            if not isinstance(token, str) or not token:
                raise ValueError("Bot Connector token response has no access_token")
            self._token_value, self._token_expires = token, time.monotonic() + float(value.get("expires_in", 3600))
            return token

    @staticmethod
    def _activity(value):
        if isinstance(value, str):
            return MessageActivityInput(text=value)
        if isinstance(value, AdaptiveCard):
            return MessageActivityInput().add_card(value)
        return value

    async def send(self, conversation_id, activity):
        if not self._initialized:
            raise ValueError("app not initialized - call app.initialize() first")
        reference = ConversationReference(bot=Account(id=self.id), conversation=ConversationAccount(id=conversation_id),
            channel_id="msteams", service_url=SERVICE_URL)
        return await self.activity_sender.send(activity, reference)

    async def reply(self, conversation_id, message_id="", activity=None):
        if activity is None:
            return await self.send(conversation_id, message_id)
        return await self.send(conversation_id.split(";messageid=", 1)[0] + ";messageid=" + str(message_id), activity)


def bindings():
    names = ("App", "ActivityContext", "ClientOptions", "MessageActivity", "ConversationReference",
        "TypingActivityInput", "AdaptiveCardInvokeActivity", "AdaptiveCardActionCardResponse",
        "AdaptiveCardActionMessageResponse", "AdaptiveCardInvokeResponse", "InvokeResponse",
        "HttpMethod", "HttpRequest", "HttpResponse", "HttpRouteHandler", "AdaptiveCard", "ExecuteAction", "TextBlock")
    return {name: globals()[name] for name in names}
