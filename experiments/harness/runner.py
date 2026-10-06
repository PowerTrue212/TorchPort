"""与具体模型供应商无关的 trial 生命周期。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .gateway import HarnessGateway

AgentLoop = Callable[[list[dict[str, Any]], Callable[[str, dict[str, Any] | None], dict]], str]


class InfrastructureError(RuntimeError):
    """模型供应商、网络或控制器错误；不计为方法失败。"""


def run_trial(gateway: HarnessGateway, agent: AgentLoop) -> dict:
    """把工具交给 Agent；Agent 返回停止原因后，controller 才执行隐藏认证。"""
    gateway.record_agent_event("agent_started")
    try:
        stop_reason = agent(gateway.tool_schemas(), gateway.call) or "agent_stopped"
    except TimeoutError as exc:
        stop_reason = "wall_budget_exhausted"
        gateway.record_agent_event("agent_error", error_type=type(exc).__name__)
    except InfrastructureError as exc:
        stop_reason = "infrastructure_error"
        gateway.record_agent_event("agent_error", error_type=type(exc).__name__, phase="provider")
    except Exception as exc:
        # Tool-call Agents express ordinary failure through responses or a stop reason.
        # An exception escaping the trusted adapter is a provider/controller failure and
        # must not be counted as a valid method failure.
        stop_reason = "infrastructure_error"
        gateway.record_agent_event("agent_error", error_type=type(exc).__name__, phase="controller")
    gateway.record_agent_event("agent_stopped", stop_reason=stop_reason)
    try:
        return gateway.certify(stop_reason)
    except Exception as exc:
        gateway.record_agent_event(
            "official_error", error_type=type(exc).__name__, phase="official_verify"
        )
        raise InfrastructureError(f"official verify failed: {type(exc).__name__}: {exc}") from exc
