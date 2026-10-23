"""Provider request/response codecs used by Hermes' original Relay adapters.

Edits are applied against the decoded original baseline. Unchanged provider
fields, multimodal blocks and unknown extensions retain their exact JSON
values. Contracts derive from the pinned sources in relay-provenance.json.
"""

from copy import deepcopy
import json

from standalone.relay_types import AnnotatedLLMRequest, AnnotatedLLMResponse, LLMRequest


_COMMON = {"model", "messages", "input", "instructions", "system", "temperature", "top_p", "stop",
           "stop_sequences", "max_tokens", "max_completion_tokens", "max_output_tokens", "tools", "tool_choice"}


def _arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return deepcopy(value)


class _Codec:
    mode = "chat"

    def decode(self, request):
        if not isinstance(request, LLMRequest):
            raise TypeError("Relay codecs require LLMRequest")
        body = request.content
        messages = deepcopy(body.get("input" if self.mode == "responses" else "messages", []))
        if isinstance(messages, str) and self.mode == "responses":
            messages = [{"role": "user", "content": messages}]
        if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
            raise ValueError("Provider conversation must contain JSON message objects")
        if self.mode == "responses":
            for item in messages:
                if item.get("type") == "function_call":
                    item.pop("type")
                    item["role"] = "tool_call"
                    item["arguments"] = _arguments(item.get("arguments", "{}"))
                elif item.get("type") == "function_call_output":
                    item.pop("type")
                    item["role"] = "tool_result"
        params = {key: body[key] for key in ("temperature", "top_p") if body.get(key) is not None}
        limit = body.get("max_completion_tokens", body.get("max_tokens", body.get("max_output_tokens")))
        if limit is not None:
            params["max_tokens"] = limit
        stop = body.get("stop_sequences" if self.mode == "anthropic" else "stop")
        if stop is not None:
            params["stop"] = [stop] if isinstance(stop, str) else deepcopy(stop)
        tools = deepcopy(body.get("tools"))
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("Provider tools must be an array")
            if self.mode == "anthropic":
                tools = [{"type": "function", "function": {"name": item["name"],
                          **({"description": item["description"]} if "description" in item else {}),
                          "parameters": item.get("input_schema", {})}} if "name" in item else item for item in tools]
            elif self.mode == "responses":
                tools = [{"type": "function", "function": {key: value for key, value in item.items() if key != "type"}}
                         if item.get("type") == "function" and "function" not in item else item for item in tools]
        return AnnotatedLLMRequest(messages, instructions=body.get("system" if self.mode == "anthropic" else "instructions"),
                                   model=body.get("model"), params=params or None, tools=tools,
                                   tool_choice=body.get("tool_choice"),
                                   extra={key: value for key, value in body.items() if key not in _COMMON})

    def encode(self, annotated, original):
        if not isinstance(annotated, AnnotatedLLMRequest):
            raise TypeError("Relay codec requires AnnotatedLLMRequest")
        baseline, body = self.decode(original), original.content
        if annotated.messages != baseline.messages:
            messages = deepcopy(annotated.messages)
            if self.mode == "responses":
                for item in messages:
                    if item.get("role") in {"tool_call", "tool_result"}:
                        role = item.pop("role")
                        item["type"] = "function_call" if role == "tool_call" else "function_call_output"
                        if role == "tool_call" and not isinstance(item.get("arguments"), str):
                            item["arguments"] = json.dumps(item.get("arguments", {}), ensure_ascii=False, separators=(",", ":"))
            body["input" if self.mode == "responses" else "messages"] = messages
        for key in ("model", "tool_choice"):
            if getattr(annotated, key) != getattr(baseline, key):
                self._set(body, key, getattr(annotated, key))
        if annotated.instructions != baseline.instructions:
            self._set(body, "system" if self.mode == "anthropic" else "instructions", annotated.instructions)
        old, new = baseline.params or {}, annotated.params or {}
        for key in ("temperature", "top_p", "max_tokens", "stop"):
            if new.get(key) == old.get(key):
                continue
            value, wire_key = new.get(key), key
            if key == "max_tokens":
                wire_key = "max_output_tokens" if self.mode == "responses" else (
                    "max_completion_tokens" if self.mode == "chat" and ("max_completion_tokens" in body or "max_tokens" not in body) else "max_tokens")
                if value is None:
                    body.pop("max_tokens", None)
                    body.pop("max_completion_tokens", None)
            if key == "stop":
                wire_key = "stop_sequences" if self.mode == "anthropic" else "stop"
                if isinstance(body.get(wire_key), str) and isinstance(value, list) and len(value) == 1:
                    value = value[0]
            self._set(body, wire_key, value)
        if annotated.tools != baseline.tools:
            tools = deepcopy(annotated.tools)
            if tools is not None and self.mode == "anthropic":
                tools = [{"name": item["function"]["name"],
                          **({"description": item["function"]["description"]} if "description" in item["function"] else {}),
                          "input_schema": item["function"].get("parameters", {})}
                         if item.get("type") == "function" else item for item in tools]
            elif tools is not None and self.mode == "responses":
                tools = [{"type": "function", **item["function"]} if item.get("type") == "function" and "function" in item
                         else item for item in tools]
            self._set(body, "tools", tools)
        # Unknown fields use the same baseline-aware add/change/delete contract.
        for key in baseline.extra.keys() | annotated.extra.keys():
            if key not in annotated.extra:
                body.pop(key, None)
            elif annotated.extra[key] != baseline.extra.get(key):
                body[key] = deepcopy(annotated.extra[key])
        return LLMRequest(original.headers, body)

    @staticmethod
    def _set(body, key, value):
        if value is None:
            body.pop(key, None)
        else:
            body[key] = deepcopy(value)

    def decode_response(self, response):
        from standalone.relay_response import decode_response
        return decode_response(self.mode, response)


class OpenAIChatCodec(_Codec):
    mode = "chat"


class OpenAIResponsesCodec(_Codec):
    mode = "responses"


class AnthropicMessagesCodec(_Codec):
    mode = "anthropic"
