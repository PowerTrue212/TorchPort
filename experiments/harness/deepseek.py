"""DeepSeek Chat Completions adapter，只把受控 Gateway tools 暴露给模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .gateway import HarnessGateway
from .protocol import FeedbackLevel, TrialProtocol
from .runner import InfrastructureError, run_trial

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def load_env(path: str | Path) -> None:
    """加载简单 KEY=VALUE 文件；已有进程变量优先，API key 不写日志。"""
    path = Path(path).expanduser()
    if not path.is_file():
        return
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"无效环境变量行：{raw!r}")
        name, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    request_timeout: float = 120.0
    max_tokens: int = 32768
    temperature: float = 0.0
    max_model_turns: int = 100
    thinking: bool = True
    reasoning_effort: str = "max"

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> DeepSeekConfig:
        env = os.environ if values is None else values
        key = env.get("DEEPSEEK_API_KEY", "").strip()
        if not key or key == "replace_me":
            raise ValueError("请在 .env 中设置 DEEPSEEK_API_KEY")
        config = cls(
            api_key=key,
            base_url=env.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=env.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
            request_timeout=float(env.get("DEEPSEEK_REQUEST_TIMEOUT", "120")),
            max_tokens=int(env.get("DEEPSEEK_MAX_TOKENS", "32768")),
            temperature=float(env.get("DEEPSEEK_TEMPERATURE", "0")),
            max_model_turns=int(env.get("DEEPSEEK_MAX_MODEL_TURNS", "100")),
            thinking=env.get("DEEPSEEK_THINKING", "enabled").strip().lower()
            not in {"0", "false", "disabled", "none"},
            reasoning_effort=env.get("DEEPSEEK_REASONING_EFFORT", "max").strip().lower(),
        )
        if config.request_timeout <= 0 or config.max_tokens <= 0 or config.max_model_turns <= 0:
            raise ValueError("DeepSeek timeout、max tokens 和 max turns 必须为正数")
        if not 0 <= config.temperature <= 2:
            raise ValueError("DEEPSEEK_TEMPERATURE 必须位于 0 到 2")
        if config.reasoning_effort not in {"none", "low", "high", "max"}:
            raise ValueError("DEEPSEEK_REASONING_EFFORT 必须是 none/low/high/max")
        if config.thinking and config.reasoning_effort == "none":
            raise ValueError("启用 thinking 时 reasoning effort 不能为 none")
        return config

    def public(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_model_turns": self.max_model_turns,
            "thinking": "enabled" if self.thinking else "disabled",
            "reasoning_effort": self.reasoning_effort if self.thinking else "none",
            "api_key_configured": True,
        }


Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]
Recorder = Callable[..., None]
BudgetReader = Callable[[], dict[str, Any]]


def post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[-2000:]
        raise InfrastructureError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise InfrastructureError(f"DeepSeek 请求失败：{type(exc).__name__}: {exc}") from exc


class DeepSeekAgent:
    def __init__(
        self,
        config: DeepSeekConfig,
        prompt: str,
        *,
        system_prompt: str | None = None,
        recorder: Recorder | None = None,
        budget: BudgetReader | None = None,
        max_total_tokens: int | None = None,
        transport: Transport = post_json,
    ) -> None:
        self.config = config
        self.prompt = prompt
        self.system_prompt = system_prompt or (
            "You are a PyTorch-to-Jittor migration agent in a controlled experiment. "
            "Use only the supplied tools to inspect and edit the candidate and to obtain "
            "verification feedback. Do not claim access to a shell or to hidden certification. "
            "Finish when the candidate is ready or the tool budget stops the trial."
        )
        self.recorder = recorder or (lambda *_args, **_kwargs: None)
        self.budget = budget
        self.max_total_tokens = max_total_tokens
        self.transport = transport

    def __call__(
        self,
        tool_schemas: list[dict[str, Any]],
        call: Callable[[str, dict[str, Any] | None], dict],
    ) -> str:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in tool_schemas
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt},
        ]
        cumulative_tokens = 0
        for turn in range(1, self.config.max_model_turns + 1):
            budget = self.budget() if self.budget else {}
            if budget.get("budget_stop_reason"):
                return str(budget["budget_stop_reason"])
            timeout = min(
                self.config.request_timeout,
                float(budget.get("wall_seconds_remaining", self.config.request_timeout)),
            )
            if timeout <= 0:
                return "wall_budget_exhausted"
            started = time.monotonic()
            try:
                request_body = {
                    "model": self.config.model,
                    "messages": list(messages),
                    "tools": tools,
                    "tool_choice": "auto",
                    "thinking": {"type": "enabled" if self.config.thinking else "disabled"},
                    "reasoning_effort": (
                        self.config.reasoning_effort if self.config.thinking else "none"
                    ),
                    "max_tokens": self.config.max_tokens,
                    "stream": False,
                }
                if not self.config.thinking:
                    request_body["temperature"] = self.config.temperature
                response = self.transport(
                    f"{self.config.base_url}/chat/completions",
                    {
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    request_body,
                    timeout,
                )
            except InfrastructureError:
                current = self.budget() if self.budget else {}
                if current.get("budget_stop_reason") == "wall_budget_exhausted":
                    return "wall_budget_exhausted"
                raise
            duration = time.monotonic() - started
            try:
                choice = response["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise InfrastructureError("DeepSeek 响应缺少 choices[0].message") from exc
            usage = _usage(response.get("usage") or {})
            total = usage.get("total_tokens")
            if total is None:
                total = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
            cumulative_tokens += int(total)
            self.recorder(
                "model_response",
                turn=turn,
                model=response.get("model", self.config.model),
                response_id=response.get("id"),
                finish_reason=choice.get("finish_reason"),
                duration_seconds=duration,
                usage=usage,
                cumulative_total_tokens=cumulative_tokens,
            )
            messages.append(_assistant_message(message))
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return "agent_completed" if message.get("content") else "agent_stopped"
            for item in tool_calls:
                function = item.get("function") or {}
                name = str(function.get("name") or "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {"_invalid_json": function.get("arguments")}
                if not isinstance(arguments, dict):
                    arguments = {"_invalid_arguments": arguments}
                result = call(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("id"),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                reason = (result.get("budget") or {}).get("budget_stop_reason")
                if reason:
                    return str(reason)
            if self.max_total_tokens is not None and cumulative_tokens >= self.max_total_tokens:
                return "token_budget_exhausted"
        return "turn_budget_exhausted"


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        key: message[key]
        for key in ("role", "content", "reasoning_content", "tool_calls")
        if key in message
    }


def _usage(value: dict[str, Any]) -> dict[str, int | None]:
    prompt_details = value.get("prompt_tokens_details") or {}
    completion_details = value.get("completion_tokens_details") or {}
    cached = prompt_details.get("cached_tokens", value.get("prompt_cache_hit_tokens"))
    return {
        "input_tokens": value.get("prompt_tokens"),
        "cached_input_tokens": cached,
        "output_tokens": value.get("completion_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "total_tokens": value.get("total_tokens"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a controlled TorchPort trial with DeepSeek")
    parser.add_argument("--env", type=Path, default=Path("experiments/harness/.env"))
    parser.add_argument("--check", action="store_true", help="只检查配置，不发送 API 请求")
    parser.add_argument("--task", type=Path)
    parser.add_argument("--hidden-task", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--feedback", choices=[item.value for item in FeedbackLevel], default="diagnostic")
    parser.add_argument("--query-budget", type=int, default=10)
    parser.add_argument("--wall-seconds", type=float, default=3600)
    parser.add_argument("--token-budget", type=int, default=100000)
    parser.add_argument("--turn-budget", type=int)
    parser.add_argument("--allow-network-research", action="store_true")
    parser.add_argument("--experiment-id", default="manual")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--trial-id", default="t0")
    parser.add_argument("--evidence-root", type=Path, default=Path(".torchport/p2"))
    parser.add_argument("--trial-dir", type=Path, default=Path("experiments/p2/runs/manual"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env(args.env)
    try:
        config = DeepSeekConfig.from_env()
    except ValueError as exc:
        raise SystemExit(f"deepseek: {exc}") from exc
    if args.check:
        print(json.dumps(config.public(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.task is None or args.prompt_file is None:
        raise SystemExit("--task 和 --prompt-file 是运行 trial 的必需参数")
    if args.token_budget <= 0 or (args.turn_budget is not None and args.turn_budget <= 0):
        raise SystemExit("--token-budget 和 --turn-budget 必须为正数")
    prompt = args.prompt_file.read_text("utf-8")
    turn_budget = args.turn_budget or config.max_model_turns
    if turn_budget != config.max_model_turns:
        config = DeepSeekConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            request_timeout=config.request_timeout,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            max_model_turns=turn_budget,
            thinking=config.thinking,
            reasoning_effort=config.reasoning_effort,
        )
    gateway = HarnessGateway(
        args.task,
        args.evidence_root / args.trial_id,
        args.trial_dir / args.trial_id,
        TrialProtocol(
            feedback=FeedbackLevel(args.feedback),
            max_oracle_queries=args.query_budget,
            max_wall_seconds=args.wall_seconds,
            allow_network_research=args.allow_network_research,
        ),
        hidden_task=args.hidden_task,
        trial_id=args.trial_id,
        metadata={
            "experiment_id": args.experiment_id,
            "condition": args.feedback,
            "replicate": args.replicate,
            "agent": {"provider": "deepseek", **config.public()},
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "token_budget": args.token_budget,
            "turn_budget": turn_budget,
        },
    )
    agent = DeepSeekAgent(
        config,
        prompt,
        recorder=gateway.record_agent_event,
        budget=gateway.budget_status,
        max_total_tokens=args.token_budget,
    )
    result = run_trial(gateway, agent)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["success"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
