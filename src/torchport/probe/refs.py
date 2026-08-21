from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RefVal:
    name: str


@dataclass(frozen=True)
class DictVal:
    name: str


@dataclass(frozen=True)
class FuncVal:
    func: Any
    env: dict[str, Any]


class CfgVal(dict):
    def __init__(self, values: dict[str, Any] | None = None):
        super().__init__()
        self.merge(values or {})

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def clone(self) -> CfgVal:
        return CfgVal(_plain(self))

    def merge(self, values: dict[str, Any]) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            if isinstance(self.get(key), CfgVal) and isinstance(value, dict):
                self[key].merge(value)
            else:
                self[key] = _wrap(value)

    def merge_file(self, path: str | Path) -> None:
        try:
            import yaml
        except ImportError:
            return
        try:
            values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, TypeError, yaml.YAMLError):
            return
        self.merge(values or {})

    def merge_list(self, values: list[Any]) -> None:
        for index in range(0, len(values) - 1, 2):
            path = values[index]
            if isinstance(path, str):
                _set_path(self, path.split("."), _decode(values[index + 1]))


def _wrap(value: Any) -> Any:
    if isinstance(value, CfgVal):
        return value
    if isinstance(value, dict):
        return CfgVal(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        import yaml
    except ImportError:
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_path(owner: CfgVal, parts: list[str], value: Any) -> None:
    current = owner
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, CfgVal):
            child = CfgVal()
            current[part] = child
        current = child
    if parts:
        current[parts[-1]] = _wrap(value)
