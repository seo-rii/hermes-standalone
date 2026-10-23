"""Qualified DingTalk OpenAPI records/HTTP calls consumed by Hermes.

Wire reviewed against alibabacloud-dingtalk 2.2.42 (uv.lock), wheel SHA256
5f5c2ef3351b7926eb870af11089e14f802e4caa51d5f72920ad79a67f03d3e4.
The original dingtalk-stream SDK still owns credentials, access-token caching,
WebSocket framing and acknowledgements. These six endpoints replace its
optional Alibaba SDK/cryptography chain, without a global SDK namespace.
"""

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from standalone.web_schema import ConfigDict, Field, RequestModel


API_URL = "https://api.dingtalk.com"


class _Record(RequestModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)


class RuntimeOptions:
    pass


class _Headers(_Record):
    x_acs_dingtalk_access_token: str


CreateCardHeaders = DeliverCardHeaders = StreamingUpdateHeaders = _Headers
RobotMessageFileDownloadHeaders = RobotReplyEmotionHeaders = RobotRecallEmotionHeaders = _Headers


class CreateCardRequestCardData(_Record):
    card_param_map: dict[str, Any] = Field(alias="cardParamMap")


class CreateCardRequestImGroupOpenSpaceModel(_Record):
    support_forward: bool = Field(alias="supportForward")


CreateCardRequestImRobotOpenSpaceModel = CreateCardRequestImGroupOpenSpaceModel


class CreateCardRequest(_Record):
    card_template_id: str = Field(alias="cardTemplateId")
    out_track_id: str = Field(alias="outTrackId")
    callback_type: str = Field(alias="callbackType")
    card_data: CreateCardRequestCardData = Field(alias="cardData")
    im_group_open_space_model: CreateCardRequestImGroupOpenSpaceModel = Field(alias="imGroupOpenSpaceModel")
    im_robot_open_space_model: CreateCardRequestImRobotOpenSpaceModel = Field(alias="imRobotOpenSpaceModel")


class DeliverCardRequestImGroupOpenDeliverModel(_Record):
    robot_code: str = Field(alias="robotCode")


class DeliverCardRequestImRobotOpenDeliverModel(_Record):
    space_type: str = Field(alias="spaceType")


class DeliverCardRequest(_Record):
    out_track_id: str = Field(alias="outTrackId")
    user_id_type: int = Field(alias="userIdType")
    open_space_id: str = Field(alias="openSpaceId")
    im_group_open_deliver_model: DeliverCardRequestImGroupOpenDeliverModel | None = Field(None, alias="imGroupOpenDeliverModel")
    im_robot_open_deliver_model: DeliverCardRequestImRobotOpenDeliverModel | None = Field(None, alias="imRobotOpenDeliverModel")


class StreamingUpdateRequest(_Record):
    out_track_id: str = Field(alias="outTrackId")
    guid: str
    key: str
    content: str
    is_full: bool = Field(alias="isFull")
    is_finalize: bool = Field(alias="isFinalize")
    is_error: bool = Field(alias="isError")


class RobotMessageFileDownloadRequest(_Record):
    download_code: str = Field(alias="downloadCode")
    robot_code: str = Field(alias="robotCode")


class _TextEmotion(_Record):
    emotion_id: str = Field(alias="emotionId")
    emotion_name: str = Field(alias="emotionName")
    text: str
    background_id: str = Field(alias="backgroundId")


RobotReplyEmotionRequestTextEmotion = RobotRecallEmotionRequestTextEmotion = _TextEmotion


class RobotReplyEmotionRequest(_Record):
    robot_code: str = Field(alias="robotCode")
    open_msg_id: str = Field(alias="openMsgId")
    open_conversation_id: str = Field(alias="openConversationId")
    emotion_type: int = Field(alias="emotionType")
    emotion_name: str = Field(alias="emotionName")
    text_emotion: _TextEmotion = Field(alias="textEmotion")


RobotRecallEmotionRequest = RobotReplyEmotionRequest


class _Body(RequestModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, strict=True)
    download_url: str | None = Field(None, alias="downloadUrl")


@dataclass
class _Response:
    body: _Body
    status_code: int


class DingTalkAPIError(RuntimeError):
    def __init__(self, response, payload):
        self.status_code = response.status_code
        self.code = payload.get("code", str(response.status_code))
        self.request_id = payload.get("requestid", payload.get("requestId", ""))
        self.data = payload
        super().__init__(f"code: {self.code}, message: {payload.get('message', response.text)} request id: {self.request_id}")


class _Client:
    async def _call(self, method, path, request, headers):
        import httpx

        parsed = urlsplit(API_URL)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("DingTalk OpenAPI endpoint must use HTTPS without credentials, query or fragment")
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            response = await client.request(method, API_URL.rstrip("/") + path,
                headers={"x-acs-dingtalk-access-token": headers.x_acs_dingtalk_access_token,
                         "User-Agent": "hermes-standalone/dingtalk-2.2.42"},
                json=request.model_dump(by_alias=True, exclude_none=True, mode="json"))
            try:
                value = response.json()
            except ValueError:
                response.raise_for_status()
                raise ValueError("DingTalk OpenAPI returned invalid JSON") from None
            if not isinstance(value, dict):
                raise ValueError("DingTalk OpenAPI returned a non-object response")
            if not response.is_success:
                raise DingTalkAPIError(response, value)
            return _Response(body=_Body.model_validate(value), status_code=response.status_code)


class CardClient(_Client):
    async def create_card_with_options_async(self, request, headers, _runtime):
        return await self._call("POST", "/v1.0/card/instances", request, headers)

    async def deliver_card_with_options_async(self, request, headers, _runtime):
        return await self._call("POST", "/v1.0/card/instances/deliver", request, headers)

    async def streaming_update_with_options_async(self, request, headers, _runtime):
        return await self._call("PUT", "/v1.0/card/streaming", request, headers)


class RobotClient(_Client):
    async def robot_message_file_download_with_options_async(self, request, headers, _runtime):
        return await self._call("POST", "/v1.0/robot/messageFiles/download", request, headers)

    async def robot_reply_emotion_with_options_async(self, request, headers, _runtime):
        return await self._call("POST", "/v1.0/robot/emotion/reply", request, headers)

    async def robot_recall_emotion_with_options_async(self, request, headers, _runtime):
        return await self._call("POST", "/v1.0/robot/emotion/recall", request, headers)
