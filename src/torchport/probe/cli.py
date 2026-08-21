from __future__ import annotations

from pathlib import Path
from typing import Any

from .parse import parse_dict, parse_list, parse_shape
from .run import AlignRun
from .side import capture


def add_probe(subparsers: Any) -> None:
    align = subparsers.add_parser("align", help="采集任意仓库的双端模块输出并数值对齐")
    align.add_argument("source", type=Path, help="源 PyTorch 仓库")
    align.add_argument("target", type=Path, help="迁移后的 Jittor 仓库")
    align.add_argument("--run-root", type=Path, required=True, help="对齐证据目录")
    align.add_argument("--entry", help="覆盖自动发现的入口")
    align.add_argument("--model", help="覆盖自动发现的模型或工厂")
    align.add_argument("--weights", type=Path, help="覆盖自动发现的 PyTorch 检查点")
    align.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        help="输入形状；多输入时重复传入",
    )
    align.add_argument("--dtype", choices=("float32", "int64"), action="append")
    align.add_argument(
        "--input",
        type=parse_dict,
        action="append",
        help="完整输入描述，可包含 shape、dtype、arg、key、pack 和 item",
    )
    align.add_argument("--model-args", type=parse_list)
    align.add_argument("--model-kw", type=parse_dict)
    align.add_argument("--seed", type=int, default=7)
    align.add_argument("--source-python", type=Path)
    align.add_argument("--target-python", type=Path)
    align.add_argument("--source-path", type=Path, action="append", default=[])
    align.add_argument("--target-path", type=Path, action="append", default=[])

    capture_cmd = subparsers.add_parser("capture", help="内部证据采集命令")
    capture_cmd.add_argument("kind", choices=("source", "target"))
    capture_cmd.add_argument("repository", type=Path)
    capture_cmd.add_argument("--entry", required=True)
    capture_cmd.add_argument("--model", required=True)
    capture_cmd.add_argument("--input", type=parse_dict, action="append", required=True)
    capture_cmd.add_argument("--model-args", type=parse_list, required=True)
    capture_cmd.add_argument("--model-kw", type=parse_dict, required=True)
    capture_cmd.add_argument("--seed", type=int, required=True)
    capture_cmd.add_argument("--output", type=Path, required=True)
    capture_cmd.add_argument("--state", type=Path, required=True)
    capture_cmd.add_argument("--source", type=Path)
    capture_cmd.add_argument("--weights", type=Path)


def run_probe(args: Any) -> tuple[dict[str, Any], int]:
    if args.command == "capture":
        report = capture(
            args.kind,
            args.repository,
            entry=args.entry,
            model_name=args.model,
            inputs=args.input,
            model_args=args.model_args,
            model_kw=args.model_kw,
            seed=args.seed,
            output=args.output,
            state=args.state,
            source=args.source,
            weights=args.weights,
        )
        return report, 0
    report = AlignRun().align(
        args.source,
        args.target,
        run_root=args.run_root,
        entry=args.entry,
        model=args.model,
        weights=args.weights,
        inputs=args.input,
        shapes=tuple(args.shape) if args.shape else None,
        dtypes=tuple(args.dtype) if args.dtype else None,
        model_args=args.model_args,
        model_kw=args.model_kw,
        seed=args.seed,
        source_python=args.source_python,
        target_python=args.target_python,
        source_path=args.source_path,
        target_path=args.target_path,
    )
    return report, 0 if report["passed"] else 3
