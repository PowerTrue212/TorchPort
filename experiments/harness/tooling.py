"""Harness 提供给成熟代码 Agent 的 TorchPort 工具协议。"""

from __future__ import annotations

from typing import Any

from .protocol import FeedbackLevel, TrialProtocol

ORACLE_TOOLS = {"evaluate", "trace", "verify"}
LOCALIZED_TOOLS = {"trace", "localize"}


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


def tool_schemas(
    protocol: TrialProtocol, task_names: tuple[str, ...] = (), *, traceable: bool = True
) -> list[dict[str, Any]]:
    """只返回 TorchPort 能力；文件、Shell、Git 和网络由代码 Agent 自己提供。

    traceable=False（没有支持 trace 的推理任务）时不暴露 trace/localize，避免必然失败的查询。
    """
    tools = [
        _tool("inspect", "查看任务契约", {}),
        _tool("run", "运行当前 target candidate", {}),
        _tool("evaluate", "评估最近一次 run", {"run_id": "string"}),
        _tool("verify", "按开发协议重复验证当前 candidate", {}),
        _tool("trace", "采集双端 trace", {}),
        _tool("localize", "定位最近一次 trace 的首次分歧", {}),
    ]
    if task_names:
        selector = {"type": "string", "enum": list(task_names)}
        for item in tools:
            if item["name"] in {"run", "evaluate"}:
                item["parameters"]["properties"]["task"] = selector
    if protocol.feedback != FeedbackLevel.LOCALIZED or not traceable:
        tools = [tool for tool in tools if tool["name"] not in LOCALIZED_TOOLS]
    return tools


def validate_call(
    protocol: TrialProtocol, tool: str, arguments: dict[str, Any], task_names: tuple[str, ...] = (),
    *, traceable: bool = True,
) -> None:
    schemas = {
        item["name"]: item["parameters"]
        for item in tool_schemas(protocol, task_names, traceable=traceable)
    }
    if tool in LOCALIZED_TOOLS and protocol.feedback == FeedbackLevel.LOCALIZED and not traceable:
        raise PermissionError(f"工具 {tool} 在本任务中不可用：没有支持 trace 的推理任务")
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


def event_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    del tool
    return arguments
