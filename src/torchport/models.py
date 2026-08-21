from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class SourceFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CAFFE = "caffe"
    ONNX = "onnx"
    JITTOR = "jittor"
    UNKNOWN = "unknown"


class MigrationTarget(str, Enum):
    JTORCH = "jtorch"
    JITTOR_NATIVE = "jittor-native"
    JITTOR_ASCEND = "jittor-ascend"


@dataclass(frozen=True)
class ProjectInspection:
    root: Path
    frameworks: tuple[SourceFramework, ...]
    framework_signals: dict[str, int]
    source_files: tuple[str, ...]
    dependency_files: tuple[str, ...]
    entrypoints: tuple[str, ...]
    weight_files: tuple[str, ...]
    custom_operator_files: tuple[str, ...]
    risks: tuple[str, ...] = field(default_factory=tuple)

    @property
    def primary_framework(self) -> SourceFramework:
        return self.frameworks[0] if self.frameworks else SourceFramework.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True)
class MigrationPlan:
    source_framework: SourceFramework
    target: MigrationTarget
    route: str
    steps: tuple[str, ...]
    blockers: tuple[str, ...]
    required_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [to_primitive(item) for item in value]
    return value

