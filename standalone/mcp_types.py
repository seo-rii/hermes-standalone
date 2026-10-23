"""Concrete JSON wire records for the standalone MCP boundary (no SDK shim)."""

from dataclasses import dataclass, field
import re

from standalone.transport_types import WireObject, json_value


LATEST_PROTOCOL_VERSION = "2026-07-28"
LATEST_HANDSHAKE_VERSION = "2025-11-25"
METHOD_NOT_FOUND = -32601


def mcp_object(value):
    if isinstance(value, MCPObject):
        return value
    if isinstance(value, dict):
        return MCPObject(value)
    if isinstance(value, (tuple, list)):
        return [mcp_object(item) for item in value]
    return value


class MCPObject(WireObject):
    def __init__(self, value=None, **kwargs):
        dict.__init__(self, {key: mcp_object(item) for key, item in dict(value or {}, **kwargs).items()})

    def __getattr__(self, name):
        if name in self:
            return self[name]
        camel = re.sub(r"_([a-z])", lambda match: match[1].upper(), name)
        if camel in self:
            return self[camel]
        if name == "meta" and "_meta" in self:
            return self["_meta"]
        raise AttributeError(name)

    def model_copy(self, *, update=None, deep=False):
        import copy
        value = type(self)(copy.deepcopy(dict(self)) if deep else dict(self))
        for name, item in (update or {}).items():
            camel = re.sub(r"_([a-z])", lambda match: match[1].upper(), name)
            value[camel if camel in value else name] = mcp_object(item)
        return value


class McpError(Exception):
    def __init__(self, error):
        self.error = mcp_object(error)
        self.code = self.error.get("code", -32603)
        super().__init__(f"MCP error {self.code}: {self.error.get('message', 'Unknown protocol error')}")


class ErrorData(MCPObject):
    pass


class TextContent(MCPObject):
    pass


class ToolUseContent(MCPObject):
    pass


class CreateMessageResult(MCPObject):
    pass


class CreateMessageResultWithTools(MCPObject):
    pass


class SamplingCapability(MCPObject):
    pass


class SamplingToolsCapability(MCPObject):
    pass


class ElicitRequestParams(MCPObject):
    pass


class ElicitResult(MCPObject):
    pass


class ToolListChangedNotification(MCPObject):
    pass


class PromptListChangedNotification(MCPObject):
    pass


class ResourceListChangedNotification(MCPObject):
    pass


@dataclass
class ServerNotification:
    root: MCPObject


@dataclass
class RequestFailure:
    identifier: str | int
    error: Exception


@dataclass
class StdioServerParameters:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    encoding: str = "utf-8"
    encoding_error_handler: str = "strict"


def validate_message(value):
    """Reject malformed envelopes before routing them to pending requests."""
    if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
        raise ValueError("MCP message must be a JSON-RPC 2.0 object")
    if "id" in value and (isinstance(value["id"], bool) or not isinstance(value["id"], (str, int))):
        raise ValueError("MCP request/response id must be a string or integer")
    if "method" in value:
        if not isinstance(value["method"], str) or not value["method"]:
            raise ValueError("MCP method must be a nonempty string")
        if "params" in value and not isinstance(value["params"], dict):
            raise ValueError("MCP params must be an object")
    elif "id" not in value or ("result" in value) == ("error" in value):
        raise ValueError("MCP response must have exactly one result or error")
    if "error" in value:
        error = value["error"]
        if (not isinstance(error, dict) or type(error.get("code")) is not int
                or not isinstance(error.get("message"), str)):
            raise ValueError("MCP error must contain numeric code and string message")
    return value
