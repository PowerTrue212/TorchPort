"""兼容入口：唯一判定实现位于 :mod:`torchport.verification.judge`。"""

from .verification.judge import _finish, _gating_outputs, _structural, judge, judge_repeats

__all__ = ["_finish", "_gating_outputs", "_structural", "judge", "judge_repeats"]
