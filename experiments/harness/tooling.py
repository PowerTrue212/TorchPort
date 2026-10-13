"""Harness 暴露给 Agent 的工具协议与参数校验。"""

from __future__ import annotations

from typing import Any

from .protocol import FeedbackLevel, TrialProtocol

ORACLE_TOOLS = {"evaluate", "trace", "verify"}
LOCALIZED_TOOLS = {"trace", "localize"}
CODE_TOOLS = {"list_files", "read_file", "search_code", "apply_patch"}


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                key: {"type": value} if isinstance(value, (str, list)) else value
                for key, value in properties.items()
            },
            "required": list(required),
            "additionalProperties": False,
        },
    }


def tool_schemas(protocol: TrialProtocol) -> list[dict[str, Any]]:
    """返回适配 function-calling Agent 的稳定工具面。"""
    tools = [
        _tool(
            "list_files",
            "列出只读 source 或可写 target 内文件",
            {"path": "string", "side": {"type": "string", "enum": ["source", "target"]}},
        ),
        _tool(
            "read_file",
            "按行读取 source 或 target 文本文件；每次最多 200 行",
            {
                "path": "string",
                "side": {"type": "string", "enum": ["source", "target"]},
                "start_line": "integer",
                "end_line": "integer",
            },
            ("path",),
        ),
        _tool(
            "search_code",
            "在 source 或 target 内搜索纯文本",
            {
                "query": "string",
                "path": "string",
                "side": {"type": "string", "enum": ["source", "target"]},
            },
            ("query",),
        ),
        _tool(
            "apply_patch",
            "精确替换一段文本；old=null 时创建新文件",
            {"path": "string", "old": ["string", "null"], "new": "string"},
            ("path", "old", "new"),
        ),
        _tool("inspect", "查看公开 TaskSpec，不读取 oracle", {}),
        _tool(
            "probe",
            "用冻结的 target Python 运行 candidate 内的临时 .py 自测脚本",
            {"path": "string", "args": {"type": "array", "items": {"type": "string"}}},
            ("path",),
        ),
        _tool("run", "运行当前 target candidate", {}),
        _tool("evaluate", "评估最近一次 run", {"run_id": "string"}),
        _tool("verify", "按开发协议重复验证当前 candidate", {}),
        _tool("trace", "采集双端 trace", {}),
        _tool("localize", "定位最近一次 trace 的首次分歧", {}),
    ]
    if protocol.allow_network_research:
        tools.extend(
            [
                _tool("web_search", "搜索公开网络资料", {"query": "string"}, ("query",)),
                _tool("web_fetch", "读取一个公开网页的正文", {"url": "string"}, ("url",)),
            ]
        )
    if protocol.feedback != FeedbackLevel.LOCALIZED:
        tools = [tool for tool in tools if tool["name"] not in LOCALIZED_TOOLS]
    return tools


def validate_call(protocol: TrialProtocol, tool: str, arguments: dict[str, Any]) -> None:
    schemas = {item["name"]: item["parameters"] for item in tool_schemas(protocol)}
    if tool not in schemas:
        raise PermissionError(f"工具在 {protocol.feedback.value} 条件中不可用：{tool}")
    schema = schemas[tool]
    properties = schema["properties"]
    extra = sorted(set(arguments) - set(properties))
    if extra:
        raise ValueError(f"工具 {tool} 包含未声明参数：{', '.join(extra)}")
    missing = [name for name in schema["required"] if name not in arguments]
    if missing:
        raise ValueError(f"工具 {tool} 缺少参数：{', '.join(missing)}")
    for name, value in arguments.items():
        allowed = properties[name].get("type")
        allowed = [allowed] if isinstance(allowed, str) else list(allowed or [])
        actual = (
            "null" if value is None else "string" if isinstance(value, str) else
            "integer" if isinstance(value, int) and not isinstance(value, bool) else
            "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else
            "boolean" if isinstance(value, bool) else "object" if isinstance(value, dict) else
            "array" if isinstance(value, list) else "unknown"
        )
        if allowed and actual not in allowed:
            raise TypeError(f"工具 {tool} 参数 {name} 类型应为 {allowed}，实际为 {actual}")
        choices = properties[name].get("enum")
        if choices and value not in choices:
            raise ValueError(f"工具 {tool} 参数 {name} 必须是 {choices}")


def required(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"缺少字符串参数：{name}")
    return value


def event_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool != "apply_patch":
        return arguments
    return {
        "path": arguments.get("path"),
        "old_bytes": len((arguments.get("old") or "").encode("utf-8")),
        "new_bytes": len((arguments.get("new") or "").encode("utf-8")),
    }
