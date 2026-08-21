from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArgVals:
    defaults: dict[str, Any] = field(default_factory=dict)
    hints: dict[str, Any] = field(default_factory=dict)

    def add(self, flags: list[Any], options: dict[str, Any]) -> None:
        names = [item for item in flags if isinstance(item, str)]
        dest = options.get("dest")
        if dest is None and names:
            dest = names[-1].lstrip("-").replace("-", "_")
        if not dest:
            return
        action = options.get("action")
        if "default" in options:
            value = options["default"]
        elif action == "store_true":
            value = False
        elif action == "store_false":
            value = True
        elif (hint := self._hint(str(dest), names)) is not None:
            value = hint
        else:
            return
        convert = options.get("type")
        if callable(convert) and isinstance(value, str):
            try:
                value = convert(value)
            except (TypeError, ValueError):
                return
        self.defaults[str(dest)] = value

    def _hint(self, dest: str, names: list[str]) -> Any:
        keys = [dest, *(name.lstrip("-").replace("-", "_") for name in names)]
        for key in keys:
            if key in self.hints:
                return self.hints[key]
        return None

    def set(self, values: dict[str, Any]) -> None:
        self.defaults.update(values)

    def parse(self, known: bool = False) -> Any:
        values = dict(self.defaults)
        return (values, []) if known else values
