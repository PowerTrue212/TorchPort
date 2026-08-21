from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import ProjectInspection, SourceFramework


class FrameworkAdapter(ABC):
    """源框架或目标框架需要实现的稳定边界。"""

    framework: SourceFramework

    @abstractmethod
    def detect(self, repository: Path) -> bool:
        """判断仓库是否使用该框架。"""

    @abstractmethod
    def inspect(self, repository: Path) -> ProjectInspection:
        """提取项目结构、模型入口、权重和风险。"""

    @abstractmethod
    def capture_reference(
        self,
        repository: Path,
        inputs: Mapping[str, Any],
        probes: Sequence[str],
    ) -> Mapping[str, Any]:
        """在源框架运行并采集可验证的参考结果。"""

    @abstractmethod
    def export_state(self, repository: Path) -> Mapping[str, Any]:
        """导出权重、训练状态和名称映射。"""


class ExecutionBackend(ABC):
    """CPU、CUDA、昇腾等硬件执行后端的稳定边界。"""

    name: str

    @abstractmethod
    def probe_environment(self) -> Mapping[str, Any]:
        """返回驱动、运行库、设备和可用能力。"""

    @abstractmethod
    def run(self, command: Sequence[str], *, cwd: Path) -> Mapping[str, Any]:
        """在受控环境执行命令并返回证据。"""

    @abstractmethod
    def synchronize(self) -> None:
        """同步异步设备，保证计时和结果采集有效。"""

    @abstractmethod
    def inspect_fallbacks(self) -> Mapping[str, Any]:
        """报告 CPU 回退、外部运行库残留和未覆盖算子。"""

