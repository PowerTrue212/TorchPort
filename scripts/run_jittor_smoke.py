#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 Jittor 最小算术执行链路")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cpu":
        # 必须在导入 Jittor 前关闭 nvcc 探测。
        os.environ["nvcc_path"] = ""

    import numpy as np
    import jittor as jt

    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        raise SystemExit(
            "Jittor 1.3.11.0 需要项目隔离的 NumPy 1.x；"
            "请将 artifacts/jittor-deps 放在 PYTHONPATH 首位。"
        )

    if args.device == "cuda":
        jt.flags.use_cuda = 1

    # 使用显式 NumPy dtype，避免旧版 Jittor 对 Python 列表的推断差异。
    source = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    value = float((jt.array(source) ** 2).sum().item())
    expected = 14.0
    passed = math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6)
    print(
        json.dumps(
            {
                "jittor": jt.__version__,
                "numpy": np.__version__,
                "device": args.device,
                "use_cuda": bool(jt.flags.use_cuda),
                "expected": expected,
                "actual": value,
                "passed": passed,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
