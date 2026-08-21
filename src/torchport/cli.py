from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .auto import AutoRun
from .doctor import build_doctor_report
from .lab import build_lab
from .migrate import RepoCheck, RepositoryMigrator
from .models import MigrationTarget
from .parity import TOLERANCE_PROFILES, compare_tensor_archives
from .planner import MigrationPlanner
from .probe.cli import add_probe, run_probe
from .probe.parse import parse_dict, parse_list, parse_shape
from .scanner import RepositoryScanner
from .verification import MODEL_VERIFICATION_CONFIGS, VerificationWorkflow


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_env(value: str) -> tuple[str, str]:
    name, separator, content = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("环境变量必须使用 NAME=VALUE 格式")
    return name, content


def _add_repo_args(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--target",
        choices=(MigrationTarget.JTORCH.value, MigrationTarget.JITTOR_NATIVE.value),
        default=MigrationTarget.JTORCH.value,
        help="JTorch 覆盖更广；原生 Jittor 会执行确定性源代码改写",
    )
    command.add_argument("--entrypoint", help="相对源仓库的 Python 运行入口")
    command.add_argument("--model", help="覆盖自动发现的模型类或工厂")
    command.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        help="输入形状；多输入时重复传入",
    )
    command.add_argument("--dtype", choices=("float32", "int64"), action="append")
    command.add_argument(
        "--input",
        type=parse_dict,
        action="append",
        help="完整输入描述，可包含 shape、dtype、arg、key、pack 和 item",
    )
    command.add_argument("--model-args", type=parse_list)
    command.add_argument("--model-kw", type=parse_dict)
    command.add_argument(
        "--scope",
        choices=("repo", "model"),
        default="repo",
        help="repo 检查全仓库；model 只把模型依赖切片作为阻塞范围",
    )
    command.add_argument(
        "--env",
        type=_parse_env,
        action="append",
        default=[],
        help="固定模型构造使用的环境变量，可重复传入",
    )


def _repo_kwargs(args: Any) -> dict[str, Any]:
    return {
        "target": MigrationTarget(args.target),
        "entrypoint": args.entrypoint,
        "env": dict(args.env),
        "model": args.model,
        "inputs": args.input,
        "shapes": tuple(args.shape) if args.shape else None,
        "dtypes": tuple(args.dtype) if args.dtype else None,
        "model_args": args.model_args,
        "model_kw": args.model_kw,
        "scope": args.scope,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchport",
        description="可验证的跨框架到 Jittor 自动迁移平台",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="检查框架、工具和硬件运行环境")
    doctor.add_argument("--runtime", action="store_true", help="实际导入 Jittor 和 PyTorch")

    scan = subparsers.add_parser("scan", help="扫描项目结构和迁移风险")
    scan.add_argument("repository", type=Path)

    plan = subparsers.add_parser("plan", help="生成初步迁移路线")
    plan.add_argument("repository", type=Path)
    plan.add_argument(
        "--target",
        choices=[target.value for target in MigrationTarget],
        default=MigrationTarget.JITTOR_NATIVE.value,
    )

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

    verify = subparsers.add_parser("verify", help="执行双端模型验证并生成统一证书")
    verify.add_argument("model", choices=tuple(MODEL_VERIFICATION_CONFIGS))
    verify.add_argument("--project-root", type=Path, default=Path.cwd())
    verify.add_argument("--run-root", type=Path, help="验证证据与报告目录")
    verify.add_argument("--source-device", choices=("cpu", "cuda"), default="cpu")
    verify.add_argument("--target-device", choices=("cpu", "cuda"), default="cpu")
    verify.add_argument(
        "--reuse-existing",
        action="store_true",
        help="复用已有双端张量证据，只重新比较并签发证书",
    )

    migrate = subparsers.add_parser("migrate", help="自动迁移完整 PyTorch 仓库")
    migrate.add_argument("repository", type=Path, help="源 PyTorch 仓库")
    migrate.add_argument("--output", type=Path, required=True, help="新的隔离输出目录")
    _add_repo_args(migrate)

    auto = subparsers.add_parser("auto", help="迁移仓库、检查运行并在可用时自动数值对齐")
    auto.add_argument("repository", type=Path, help="源 PyTorch 仓库")
    auto.add_argument("--output", type=Path, required=True, help="新的隔离输出目录")
    _add_repo_args(auto)
    auto.add_argument("--run-root", type=Path, help="对齐证据目录，默认写入输出目录")
    auto.add_argument("--check-run", action="store_true", help="执行入口；默认只导入入口")
    auto.add_argument("--no-runtime", action="store_true", help="只做静态检查")
    auto.add_argument("--timeout", type=int, default=120, help="运行超时秒数")
    auto.add_argument("--no-align", action="store_true", help="只迁移和检查，不执行数值对齐")
    auto.add_argument("--try-hints", action="store_true", help="用高置信静态建议自动重跑一次")
    auto.add_argument("--weights", type=Path, help="覆盖自动发现的 PyTorch 检查点")
    auto.add_argument("--seed", type=int, default=7)
    auto.add_argument("--source-python", type=Path)
    auto.add_argument("--target-python", type=Path)
    auto.add_argument("--source-path", type=Path, action="append", default=[])
    auto.add_argument("--target-path", type=Path, action="append", default=[])
    auto.add_argument("--agent", type=parse_dict, help="记录 DSH/LLM 本轮决策依据")

    check = subparsers.add_parser("check", help="检查迁移产物并执行目标端冒烟")
    check.add_argument("repository", type=Path, help="torchport migrate 的输出目录")
    check.add_argument("--run", action="store_true", help="执行入口；默认只导入入口")
    check.add_argument("--no-runtime", action="store_true", help="只做静态检查")
    check.add_argument("--timeout", type=int, default=120, help="运行超时秒数")

    lab = subparsers.add_parser("lab", help="汇总真实仓库迁移实验报告")
    lab.add_argument("reports", type=Path, nargs="+", help="auto.json 文件或包含报告的目录")
    lab.add_argument("--name", action="append", default=[], help="只统计指定仓库名")
    lab.add_argument("--output", type=Path, help="可选的 JSON 汇总输出")
    add_probe(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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

        if args.command == "verify":
            run_root = args.run_root or Path("artifacts/runs") / f"{args.model}-verification"
            certificate = VerificationWorkflow(args.project_root).verify(
                args.model,
                run_root=run_root,
                source_device=args.source_device,
                target_device=args.target_device,
                reuse_existing=args.reuse_existing,
            )
            _print_json(certificate)
            return 0 if certificate["passed"] else 3

        if args.command == "migrate":
            migration = RepositoryMigrator().migrate(
                args.repository,
                args.output,
                **_repo_kwargs(args),
            )
            _print_json(migration)
            return 0 if migration["status"] == "converted" else 4

        if args.command == "auto":
            report = AutoRun().run(
                args.repository,
                args.output,
                **_repo_kwargs(args),
                check_run=args.check_run,
                runtime=not args.no_runtime,
                timeout=args.timeout,
                align=not args.no_align,
                run_root=args.run_root,
                weights=args.weights,
                seed=args.seed,
                source_python=args.source_python,
                target_python=args.target_python,
                source_path=args.source_path,
                target_path=args.target_path,
                try_hints=args.try_hints,
                agent=args.agent,
            )
            _print_json(report)
            return 0 if report["passed"] else 6

        if args.command == "check":
            report = RepoCheck().check(
                args.repository,
                run=args.run,
                runtime=not args.no_runtime,
                timeout=args.timeout,
            )
            _print_json(report)
            return 0 if report["passed"] else 5

        if args.command == "lab":
            report = build_lab(args.reports, names=args.name)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            _print_json(report)
            return 0

        if args.command in {"capture", "align"}:
            report, code = run_probe(args)
            _print_json(report)
            return code

        inspection = RepositoryScanner().scan(args.repository)
        if args.command == "scan":
            _print_json(inspection.to_dict())
            return 0
        if args.command == "plan":
            migration_plan = MigrationPlanner().create(
                inspection,
                MigrationTarget(args.target),
            )
            _print_json(
                {
                    "inspection": inspection.to_dict(),
                    "plan": migration_plan.to_dict(),
                }
            )
            return 0
    except (OSError, ValueError) as exc:
        raise SystemExit(f"torchport: {exc}") from exc
    return 2
