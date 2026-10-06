from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .doctor import build_doctor_report
from .parity import TOLERANCE_PROFILES, compare_tensor_archives


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchport",
        description="供迁移 Agent 调用的可复现执行与验证工具",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="检查框架、工具和硬件运行环境")
    doctor.add_argument("--runtime", action="store_true", help="实际导入 Jittor 和 PyTorch")

    compare = subparsers.add_parser("compare", help="比较两份张量证据并定位首次分歧")
    compare.add_argument("reference", type=Path, help="源框架 NPZ 证据")
    compare.add_argument("candidate", type=Path, help="目标框架 NPZ 证据")
    compare.add_argument(
        "--profile",
        choices=tuple(TOLERANCE_PROFILES),
        default="fp32-inference",
        help="容差配置",
    )
    compare.add_argument("--rtol", type=float, help="覆盖配置中的相对误差容差")
    compare.add_argument("--atol", type=float, help="覆盖配置中的绝对误差容差")
    compare.add_argument("--output", type=Path, help="可选的 JSON 报告路径")

    inspect = subparsers.add_parser("inspect", help="查看公开任务契约，不读取 oracle")
    inspect.add_argument("task", type=Path, help="task.yaml / task.json")

    for name, help_text in (
        ("run", "按冻结 task 执行一端，输出 RunManifest，不判定成功"),
        ("evaluate", "只读既有 oracle 与 candidate 产物并判定，不启动模型"),
        ("trace", "以同一输入和状态采集双端 trace"),
        ("verify", "冻结 oracle 并重复运行 candidate，给出唯一结果"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("task", type=Path, help="task.yaml / task.json")
        command.add_argument("--root", type=Path, default=Path(".torchport/harness"))
        command.add_argument("--trial", default="t0")
        if name == "run":
            command.add_argument("side", choices=("source", "target"))
            command.add_argument("--mode", choices=("normal", "trace"), default="normal")
        if name == "evaluate":
            command.add_argument("manifest", type=Path, help="target 端 RunManifest")

    localize = subparsers.add_parser("localize", help="从既有 trace 报告提取首次分歧")
    localize.add_argument("report", type=Path, help="torchport trace 生成的 JSON 报告")

    internal = subparsers.add_parser("_capture", help=argparse.SUPPRESS)
    internal.add_argument("job", type=Path)
    return parser


def _harness(args: Any) -> tuple[dict[str, Any], int]:
    from .execution import read_json
    from .oracle import build_oracle, evaluate, load_oracle, run_side, verify
    from .task import load_task

    task = load_task(args.task)
    root = args.root.resolve()
    if args.command == "verify":
        report = verify(task, root, trial=args.trial)
        return report, 0 if report["passed"] else 3
    if args.command == "trace":
        from .diagnosis import trace

        report = trace(task, root, trial=args.trial)
        return report, 0 if report["passed"] else 3
    if args.command == "evaluate":
        oracle = load_oracle(task, root)
        report = evaluate(oracle, read_json(args.manifest), task["protocol"], task["target_identity"])
        return report, 0 if report["passed"] else 3
    oracle = None
    if args.side == "target":
        meta = build_oracle(task, root)
        if not meta.get("ok"):
            return meta, 3
        oracle = Path(meta["dir"])
    manifest = run_side(task, args.side, root, trial=args.trial, mode=args.mode, oracle=oracle)
    return manifest, 0 if manifest["ok"] else 3


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "_capture":
            from .capture import main as capture_main

            return capture_main(str(args.job))

        if args.command in {"run", "evaluate", "trace", "verify"}:
            report, code = _harness(args)
            _print_json(report)
            return code

        if args.command == "inspect":
            from .diagnosis import inspect_task
            from .task import load_task

            _print_json(inspect_task(load_task(args.task)))
            return 0

        if args.command == "localize":
            from .diagnosis import load_and_localize

            report = load_and_localize(args.report)
            _print_json(report)
            return 0 if report["passed"] else 3

        if args.command == "doctor":
            _print_json(build_doctor_report(runtime=args.runtime))
            return 0

        if args.command == "compare":
            report = compare_tensor_archives(
                args.reference,
                args.candidate,
                profile=args.profile,
                rtol=args.rtol,
                atol=args.atol,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            _print_json(report)
            return 0 if report["passed"] else 3
    except (OSError, ValueError) as exc:
        raise SystemExit(f"torchport: {exc}") from exc
    return 2
