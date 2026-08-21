from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from torchport.probe.ops import OpTrace


class OpTests(unittest.TestCase):
    def test_trace(self) -> None:
        def original(value):
            return value + 1

        module = SimpleNamespace(relu=original)
        scope = {"act": original}
        values = {}
        with (
            patch.object(OpTrace, "_modules", return_value=[(module, ("relu",))]),
            patch("torchport.probe.ops.find_tensor", side_effect=lambda value, _kind: value),
            patch(
                "torchport.probe.ops.to_array",
                side_effect=lambda value, _kind: np.asarray(value),
            ),
        ):
            trace = OpTrace("source", values, [scope])
            trace.start()
            first = module.relu(2)
            second = scope["act"](3)
            trace.stop()

        self.assertEqual(first, 3)
        self.assertEqual(second, 4)
        self.assertEqual(values["op.relu"], 3)
        self.assertEqual(values["op.relu#2"], 4)
        self.assertIs(module.relu, original)
        self.assertIs(scope["act"], original)

    def test_replay(self) -> None:
        def original(*_args, **_kwargs):
            return np.asarray([1, 2], dtype=np.float32)

        module = SimpleNamespace(rand=original)
        values = {}
        replay = {"op.rand": np.asarray([3, 4], dtype=np.float32)}
        with (
            patch.object(OpTrace, "_modules", return_value=[(module, ("rand",))]),
            patch.object(OpTrace, "_tensor", side_effect=lambda value: value),
            patch("torchport.probe.ops.find_tensor", side_effect=lambda value, _kind: value),
            patch(
                "torchport.probe.ops.to_array",
                side_effect=lambda value, _kind: np.asarray(value),
            ),
        ):
            trace = OpTrace("target", values, replay=replay)
            trace.start()
            output = module.rand(2)
            trace.stop()

        np.testing.assert_array_equal(output, replay["op.rand"])
        np.testing.assert_array_equal(values["op.rand"], replay["op.rand"])
        self.assertIs(module.rand, original)


if __name__ == "__main__":
    unittest.main()
