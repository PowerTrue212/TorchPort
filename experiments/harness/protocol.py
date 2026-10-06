"""实验条件与预算；这里只定义控制变量，不实现 TorchPort 判断。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

FEEDBACK_SCHEMA_VERSION = 1
DIAGNOSTIC_FIELDS = (
    "passed",
    "aligned",
    "strict_aligned",
    "stage",
    "failure_code",
    "category",
    "max_abs",
    "mismatch_ratio",
    "mismatched_elements",
    "mismatched_tensors",
    "structural",
    "weight_load",
)
LOCALIZATION_FIELDS = (
    "kind",
    "passed",
    "stage",
    "scope",
    "invocation",
    "operator",
    "first_divergence",
    "evidence",
    "error_jump",
    "metrics",
)


class FeedbackLevel(str, Enum):
    VERDICT = "verdict"
    DIAGNOSTIC = "diagnostic"
    LOCALIZED = "localized"


@dataclass(frozen=True)
class TrialProtocol:
    feedback: FeedbackLevel
    max_oracle_queries: int
    max_wall_seconds: float
    allow_network_research: bool = False

    def __post_init__(self) -> None:
        if self.max_oracle_queries < 1:
            raise ValueError("max_oracle_queries 必须为正数")
        if self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds 必须为正数")

    def as_dict(self) -> dict:
        value = asdict(self)
        value["feedback"] = self.feedback.value
        value["feedback_schema_version"] = FEEDBACK_SCHEMA_VERSION
        return value
