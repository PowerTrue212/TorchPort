from __future__ import annotations

import unittest

import numpy as np

from torchport.capture import flatten, pack_values


class WalkTests(unittest.TestCase):
    def test_flatten(self) -> None:
        first = np.ones((1, 2), dtype=np.float32)
        second = np.zeros((1, 2), dtype=np.float32)

        result = flatten(
            {"logits": first, "parts": (second, "ignored")},
            "output",
            lambda value: isinstance(value, np.ndarray),
            lambda value: value,
        )

        self.assertEqual(list(result), ["output.logits", "output.parts.0"])
        self.assertIs(result["output.logits"], first)

    def test_pack(self) -> None:
        inputs = [
            {"pack": "tuple", "item": 0},
            {"pack": "tuple", "item": 1},
            {"arg": "scale"},
        ]

        args, kwargs = pack_values([1, 2, 3], inputs)

        self.assertEqual(args, [(1, 2)])
        self.assertEqual(kwargs, {"scale": 3})

if __name__ == "__main__":
    unittest.main()
