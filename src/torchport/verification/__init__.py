"""Oracle 生命周期、candidate 运行和唯一验收判定。"""

from .judge import judge, judge_repeats
from .oracle import ORACLE_KIND, build_oracle, load_oracle
from .runner import MANIFEST, run_side
from .service import build_envelope, evaluate, verify, verify_envelope

__all__ = [
    "MANIFEST",
    "ORACLE_KIND",
    "build_envelope",
    "build_oracle",
    "evaluate",
    "judge",
    "judge_repeats",
    "load_oracle",
    "run_side",
    "verify",
    "verify_envelope",
]

