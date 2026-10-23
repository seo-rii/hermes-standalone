"""Validate MCP 2026-07-28 tool-schema header annotations before using them."""

import re


_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")


def header_fields(schema):
    fields, names = [], set()

    def visit(node, path, allowed):
        if isinstance(node, list):
            for child in node:
                visit(child, path, False)
            return
        if not isinstance(node, dict):
            return
        if "x-mcp-header" in node:
            name, kind = node["x-mcp-header"], node.get("type")
            if (not allowed or not path or not isinstance(name, str) or not _TOKEN.fullmatch(name)
                    or name.casefold() in names or kind not in ("string", "integer", "boolean")):
                raise ValueError("invalid x-mcp-header annotation")
            names.add(name.casefold())
            fields.append((name, path, kind))
        for key, child in node.items():
            if key == "properties" and isinstance(child, dict):
                for part, nested in child.items():
                    visit(nested, (*path, part), allowed)
            elif isinstance(child, (dict, list)):
                visit(child, path, False)

    visit(schema, (), True)
    return fields


def parameter_headers(schema, arguments):
    result = {}
    for name, path, kind in header_fields(schema):
        value = arguments
        for part in path:
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            valid = ((kind == "string" and isinstance(value, str))
                     or (kind == "boolean" and isinstance(value, bool))
                     or (kind == "integer" and type(value) is int and abs(value) <= 2**53 - 1))
            if not valid:
                raise ValueError(f"MCP header parameter {'.'.join(path)} has an invalid primitive value")
            result["Mcp-Param-" + name] = str(value).lower() if kind == "boolean" else str(value)
    return result
