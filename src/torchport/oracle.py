"""兼容入口：oracle 与 verify 编排位于 :mod:`torchport.verification`。"""

from .verification import (
    MANIFEST,
    ORACLE_KIND,
    build_envelope,
    build_oracle,
    evaluate,
    load_oracle,
    run_side,
    verify,
    verify_envelope,
)

__all__ = [
    "MANIFEST",
    "ORACLE_KIND",
    "build_envelope",
    "build_oracle",
    "evaluate",
    "load_oracle",
    "run_side",
    "verify",
    "verify_envelope",
]
