from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, MutableMapping


@dataclass
class EnvVals:
    supplied: dict[str, str] = field(default_factory=dict)
    used: dict[str, str | None] = field(default_factory=dict)

    def read(self, name: Any, default: Any, missing: Any) -> Any:
        if not isinstance(name, str):
            return missing
        if name in self.supplied:
            value = self.supplied[name]
            self.used[name] = value
            return value
        if default is not missing:
            self.used[name] = None
            return default
        return missing


def apply_env(
    target: MutableMapping[str, str],
    contract: dict[str, str | None],
) -> None:
    for name, value in contract.items():
        if value is None:
            target.pop(name, None)
        else:
            target[name] = value
