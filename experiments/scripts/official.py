from __future__ import annotations

import argparse
import ast
import json
import shutil
import time
from pathlib import Path
from typing import Any


def main() -> int:
    args = _args()
    start = time.time()
    out = args.output
    result: dict[str, Any] = {
        "schema_version": 1,
        "method": "official_jittor",
        "case": args.case,
        "source": str(args.source),
        "entry": args.entry,
        "status": "failed",
        "stages": {},
        "seconds": None,
        "error": None,
    }
    try:
        _copy(args.source, out)
        converted = _convert((args.source / args.entry).read_text(encoding="utf-8"))
        target = out / args.entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(converted, encoding="utf-8")
        ast.parse(converted, filename=args.entry)
        result["stages"]["convert"] = True
        result["stages"]["syntax"] = True
        result["status"] = "converted"
    except Exception as exc:  # noqa: BLE001 - baseline must record exact failure
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["seconds"] = round(time.time() - start, 3)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if result["status"] == "converted" else 1


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _copy(source: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache")
    shutil.copytree(source, output, ignore=ignore)


def _convert(text: str) -> str:
    try:
        from jittor.utils.pytorch_converter import convert
    except Exception as exc:
        raise RuntimeError("jittor official converter is unavailable") from exc
    value = convert(text)
    if not isinstance(value, str):
        raise TypeError("official converter did not return source text")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
