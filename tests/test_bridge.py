from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from torchport.probe.side import _check_bridge


class Array:
    def __init__(self, value, valid: bool) -> None:
        self.value = np.asarray(value) if valid else np.zeros_like(value)

    def numpy(self):
        return self.value


class BridgeTests(unittest.TestCase):
    def test_valid(self) -> None:
        module = SimpleNamespace(array=lambda value: Array(value, True))
        with patch.dict(sys.modules, {"jittor": module}):
            _check_bridge("target")

    def test_invalid(self) -> None:
        module = SimpleNamespace(array=lambda value: Array(value, False))
        with (
            patch.dict(sys.modules, {"jittor": module}),
            self.assertRaisesRegex(ValueError, "NumPy < 2"),
        ):
            _check_bridge("target")


if __name__ == "__main__":
    unittest.main()
