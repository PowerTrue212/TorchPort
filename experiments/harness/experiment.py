"""准备、恢复并运行一个冻结的 P2 实验计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from torchport.runtime import write_json

from .agents.deepseek import DeepSeekAgent, DeepSeekConfig, load_env
from .gateway import HarnessGateway
from .orchestration import AttemptRunner, execute_run_plan, preflight, prepare_run_plan
from .orchestration.common import inside as _inside
from .orchestration.common import read_json as _json
from .protocol import FeedbackLevel, TrialProtocol
from .reporting.aggregate import aggregate
from .runner import run_trial

__all__ = ["execute_run_plan", "preflight", "prepare_run_plan"]


def _deepseek_runner(
    plan: dict[str, Any], root: Path, config: DeepSeekConfig, prompt: str
) -> AttemptRunner:
    budget = plan["budget"]

    def run(
        item: dict[str, Any],
        attempt_id: str,
        public_task: Path,
        hidden_task: Path,
        attempt_path: Path,
        runtime_sha256: str,
    ) -> dict[str, Any]:
        condition = FeedbackLevel(str(item["condition"]))
        gateway = HarnessGateway(
            public_task,
            root / ".evidence" / str(plan["experiment_id"]) / attempt_id,
            attempt_path,
            TrialProtocol(
                feedback=condition,
                max_oracle_queries=int(budget["oracle_queries"]),
                max_wall_seconds=float(budget["wall_seconds"]),
                allow_network_research=bool(plan.get("network_research", False)),
            ),
            hidden_task=hidden_task,
            trial_id=attempt_id,
            metadata={
                "experiment_id": plan["experiment_id"],
                "condition": condition.value,
                "replicate": int(item.get("replicate", 0)),
                "agent": {"provider": "deepseek", **config.public()},
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "token_budget": int(budget["total_tokens"]),
                "turn_budget": int(budget["model_turns"]),
                "trial_runtime_snapshot_sha256": runtime_sha256,
            },
        )
        agent = DeepSeekAgent(
            config,
            prompt,
            recorder=gateway.record_agent_event,
            budget=gateway.budget_status,
            max_total_tokens=int(budget["total_tokens"]),
        )
        return run_trial(gateway, agent)

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--env", type=Path, default=Path("experiments/harness/.env"))
    parser.add_argument("--worker-attestation", type=Path)
    parser.add_argument("--allow-trusted-pilot", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-infrastructure-retries", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan_path = prepare_run_plan(
            args.manifest,
            worker_attestation=args.worker_attestation,
            allow_trusted_pilot=args.allow_trusted_pilot,
        )
        if args.prepare_only:
            print(json.dumps({"run_plan": str(plan_path)}, indent=2))
            return 0
        load_env(args.env)
        config = DeepSeekConfig.from_env()
        plan = _json(plan_path)
        if config.model != plan.get("model"):
            raise ValueError(
                f"DeepSeek model 与 run-plan 不一致：{config.model} != {plan.get('model')}"
            )
        config = DeepSeekConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            request_timeout=config.request_timeout,
            max_tokens=config.max_tokens,
            temperature=float(plan.get("temperature", 0.0)),
            max_model_turns=int(plan["budget"]["model_turns"]),
            thinking=str(plan.get("thinking", "enabled")).lower() != "disabled",
            reasoning_effort=str(plan.get("reasoning_effort", "max")),
        )
        prompt_path = _inside(plan_path.parent, str(plan["prompt"]))
        prompt = prompt_path.read_text("utf-8")
        state = execute_run_plan(
            plan_path,
            _deepseek_runner(plan, plan_path.parent, config, prompt),
            max_infrastructure_retries=args.max_infrastructure_retries,
        )
        report = aggregate(plan_path)
        write_json(plan_path.parent / "summary.json", report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"experiment: {exc}") from exc
    print(json.dumps({"run_state": state, "summary": report}, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
