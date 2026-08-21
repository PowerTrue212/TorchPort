from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from .models import ProjectInspection, SourceFramework

_IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
    "artifacts",
}

_DEPENDENCY_FILENAMES = {
    "requirements.txt",
    "requirements-dev.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
}

_ENTRYPOINT_RE = re.compile(
    r"^(main|app|train|training|infer|inference|predict|evaluate|eval|demo)([_-].*)?\.py$",
    re.IGNORECASE,
)

_TEXT_SIGNALS: tuple[tuple[SourceFramework, re.Pattern[str]], ...] = (
    (SourceFramework.PYTORCH, re.compile(r"(^|\n)\s*(from\s+torch\b|import\s+torch\b)")),
    (
        SourceFramework.TENSORFLOW,
        re.compile(r"(^|\n)\s*(from\s+tensorflow\b|import\s+tensorflow\b)"),
    ),
    (SourceFramework.CAFFE, re.compile(r"(^|\n)\s*(from\s+caffe\b|import\s+caffe\b)")),
    (SourceFramework.ONNX, re.compile(r"(^|\n)\s*(from\s+onnx\b|import\s+onnx\b)")),
    (SourceFramework.JITTOR, re.compile(r"(^|\n)\s*(from\s+jittor\b|import\s+jittor\b)")),
)

_DEPENDENCY_SIGNALS: tuple[tuple[SourceFramework, re.Pattern[str]], ...] = (
    (SourceFramework.PYTORCH, re.compile(r"(^|[\s=<>~!])(?:torch|torchvision)(?:[\s=<>~!]|$)")),
    (SourceFramework.TENSORFLOW, re.compile(r"(^|[\s=<>~!])tensorflow(?:[\s=<>~!]|$)")),
    (SourceFramework.CAFFE, re.compile(r"(^|[\s=<>~!])caffe(?:[\s=<>~!]|$)")),
    (SourceFramework.ONNX, re.compile(r"(^|[\s=<>~!])onnx(?:[\s=<>~!]|$)")),
    (SourceFramework.JITTOR, re.compile(r"(^|[\s=<>~!])jittor(?:[\s=<>~!]|$)")),
)

_WEIGHT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".caffemodel",
    ".onnx",
    ".h5",
    ".keras",
    ".pb",
}

_CUSTOM_OPERATOR_SUFFIXES = {".cu", ".cuh", ".cc", ".cpp", ".cxx", ".so"}
_MAX_TEXT_BYTES = 1_000_000


class RepositoryScanner:
    def scan(self, repository: str | Path) -> ProjectInspection:
        root = Path(repository).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"仓库目录不存在：{root}")

        signals: Counter[SourceFramework] = Counter()
        source_files: list[str] = []
        dependency_files: list[str] = []
        entrypoints: list[str] = []
        weight_files: list[str] = []
        custom_operator_files: list[str] = []

        for path in self._iter_files(root):
            relative = path.relative_to(root).as_posix()
            suffix = path.suffix.lower()

            is_dependency_file = path.name in _DEPENDENCY_FILENAMES or (
                path.name.lower().startswith("requirements") and suffix == ".txt"
            )
            if is_dependency_file:
                dependency_files.append(relative)
            if _ENTRYPOINT_RE.match(path.name):
                entrypoints.append(relative)
            if suffix in _WEIGHT_SUFFIXES:
                weight_files.append(relative)
            if suffix in _CUSTOM_OPERATOR_SUFFIXES:
                custom_operator_files.append(relative)
            if suffix == ".prototxt":
                signals[SourceFramework.CAFFE] += 3
                source_files.append(relative)
            elif suffix == ".caffemodel":
                signals[SourceFramework.CAFFE] += 4
            elif suffix == ".onnx":
                signals[SourceFramework.ONNX] += 4

            if suffix not in {".py", ".txt", ".toml", ".yaml", ".yml", ".cfg"}:
                continue
            text = self._read_text(path)
            if text is None:
                continue
            if suffix == ".py":
                source_files.append(relative)
            for framework, pattern in _TEXT_SIGNALS:
                if pattern.search(text):
                    signals[framework] += 1
            if is_dependency_file:
                for framework, pattern in _DEPENDENCY_SIGNALS:
                    if pattern.search(text):
                        signals[framework] += 1

        frameworks = tuple(
            framework
            for framework, _ in sorted(signals.items(), key=lambda item: (-item[1], item[0].value))
        )
        risks: list[str] = []
        if custom_operator_files:
            risks.append("检测到 C++、CUDA 或动态库文件，需要自定义算子语义分析。")
        if not entrypoints:
            risks.append("未识别到标准运行入口，需要用户或智能体确认执行命令。")
        if len(frameworks) > 1:
            risks.append("检测到多个框架，需要区分源框架、导出工具和辅助依赖。")
        if not frameworks:
            risks.append("未识别到受支持的源框架。")

        return ProjectInspection(
            root=root,
            frameworks=frameworks,
            framework_signals={framework.value: count for framework, count in signals.items()},
            source_files=tuple(sorted(set(source_files))),
            dependency_files=tuple(sorted(set(dependency_files))),
            entrypoints=tuple(sorted(set(entrypoints))),
            weight_files=tuple(sorted(set(weight_files))),
            custom_operator_files=tuple(sorted(set(custom_operator_files))),
            risks=tuple(risks),
        )

    def _iter_files(self, root: Path):
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            directory_names[:] = sorted(
                name for name in directory_names if name not in _IGNORED_DIRECTORIES
            )
            base = Path(directory)
            for file_name in sorted(file_names):
                path = base / file_name
                if path.is_file() and not path.is_symlink():
                    yield path

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            if path.stat().st_size > _MAX_TEXT_BYTES:
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
