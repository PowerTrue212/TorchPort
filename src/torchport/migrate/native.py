from __future__ import annotations

from .apis import ApiRules
from .base import BaseEdit
from .imports import ImportRules


class NativeEdit(ImportRules, ApiRules, BaseEdit):
    """把可证明等价的 PyTorch 语法改写为原生 Jittor。"""

