"""Bedrock Messages using HTTPX and the embedded pure Python botocore signer.

Wire routing follows Anthropic's public Bedrock client contract; AWS event CRCs
and binary frames are decoded by botocore, not by an approximate SSE parser.
https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/lib/bedrock
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from urllib.parse import quote

from standalone.transport import UNSET, split_request_options
from standalone.transport_chat import response_object
from standalone.transport_messages import Anthropic, Messages, MessageStream, _merge_headers
from standalone.transport_types import APIResponseValidationError, status_error, wire_object


@lru_cache(maxsize=1)
def _response_stream_shape():
    from botocore.loaders import Loader
    from botocore.model import ServiceModel
    return ServiceModel(Loader().load_service_model("bedrock-runtime", "service-2")).shape_for("ResponseStream")


class BedrockEvents:
    def __init__(self, response):
        self.response = response

    def __iter__(self):
        from botocore.eventstream import EventStreamBuffer
        from botocore.parsers import EventStreamJSONParser
        buffer, parser = EventStreamBuffer(), EventStreamJSONParser()
        try:
            for chunk in self.response.iter_bytes():
                buffer.add_data(chunk)
                for event in buffer:
                    envelope = event.to_response_dict()
                    parsed = parser.parse(envelope, _response_stream_shape())
                    if envelope["status_code"] != 200:
                        error = parsed.get("Error", {})
                        code = error.get("Code")
                        status = {"throttlingException": 429, "validationException": 400,
                                  "modelTimeoutException": 408, "modelStreamErrorException": 424,
                                  "serviceUnavailableException": 503}.get(code, 500)
                        raise status_error(self.response, {"error": {"type": code, "message": error.get("Message")}},
                                           status_code=status)
                    payload = parsed.get("chunk", {}).get("bytes")
                    if payload is None:
                        continue
                    try:
                        value = json.loads(payload)
                        if not isinstance(value, dict):
                            raise ValueError("Expected an object")
                    except (ValueError, UnicodeError) as exc:
                        raise APIResponseValidationError("Invalid Bedrock event JSON", response=self.response,
                                                         body=None) from exc
                    if value.get("type") == "error":
                        raise status_error(self.response, value, status_code=500)
                    yield wire_object(value)
        finally:
            self.close()

    def close(self):
        self.response.close()


class BedrockMessages(Messages):
    def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        model = body.pop("model", None)
        if not isinstance(model, str) or not model:
            raise ValueError("Bedrock Messages requires a model identifier")
        streaming = bool(body.pop("stream", False))
        body.setdefault("anthropic_version", "bedrock-2023-05-31")
        headers = _merge_headers(self._client._custom_headers, options["headers"])
        if headers.get("anthropic-beta"):
            body.setdefault("anthropic_beta", headers["anthropic-beta"].split(","))
        suffix = "invoke-with-response-stream" if streaming else "invoke"
        response = self._client.post(f"model/{quote(model, safe=':')}/{suffix}", body=body,
                                     stream=streaming, **options)
        if streaming:
            return MessageStream(response, BedrockEvents(response))
        try:
            return response_object(response)
        finally:
            response.close()

    def count_tokens(self, **kwargs):
        raise NotImplementedError("The upstream Anthropic Bedrock client does not expose token counting")


class AnthropicBedrock(Anthropic):
    def __init__(self, *, aws_secret_key=None, aws_access_key=None, aws_region=None,
                 aws_profile=None, aws_session_token=None, api_key=None, base_url=None,
                 timeout=UNSET, max_retries=2, default_headers=None, default_query=None,
                 http_client=None):
        import boto3
        from agent.secret_scope import get_secret
        supplied = any(value is not None for value in (aws_access_key, aws_secret_key, aws_session_token, aws_profile))
        if api_key is None and not supplied:
            api_key = get_secret("AWS_BEARER_TOKEN_BEDROCK")
        if api_key and supplied:
            raise ValueError("Specify either a Bedrock bearer token or AWS credentials")
        self._aws_session = boto3.Session(aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key,
                                         aws_session_token=aws_session_token, profile_name=aws_profile,
                                         region_name=aws_region)
        self.aws_region = aws_region or os.environ.get("AWS_REGION") or self._aws_session.region_name
        if not self.aws_region:
            raise ValueError("An AWS region is required")
        if base_url is None:
            endpoint = self._aws_session._session.get_component("endpoint_resolver").construct_endpoint(
                "bedrock-runtime", self.aws_region)
            if not endpoint:
                raise ValueError("Cannot resolve the Bedrock endpoint for this AWS region")
            base_url = os.environ.get("ANTHROPIC_BEDROCK_BASE_URL") or "https://" + endpoint["hostname"]
        self._bedrock_token = api_key
        super().__init__(base_url=base_url, timeout=timeout, max_retries=max_retries,
                         default_headers=default_headers, default_query=default_query, http_client=http_client)

    @property
    def messages(self):
        return BedrockMessages(self)

    def _build_request(self, method, path, **kwargs):
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.exceptions import NoCredentialsError
        request = super()._build_request(method, path, **kwargs)
        if self._bedrock_token:
            request.headers["Authorization"] = "Bearer " + self._bedrock_token
        else:
            credentials = self._aws_session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()
            signed = AWSRequest(method=request.method, url=str(request.url), data=request.content,
                                headers=dict(request.headers))
            SigV4Auth(credentials.get_frozen_credentials(), "bedrock", self.aws_region).add_auth(signed)
            request.headers.update(dict(signed.headers))
        return request
