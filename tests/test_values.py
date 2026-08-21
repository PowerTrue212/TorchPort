from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from torchport.probe.envvals import EnvVals, apply_env
from torchport.probe.values import MISSING, call_args, eval_node


class ValueTests(unittest.TestCase):
    def test_env_contract(self) -> None:
        state = EnvVals({"WIDTH": "4"})
        target = {"WIDTH": "99", "HIDDEN": "99"}

        width = state.read("WIDTH", MISSING, MISSING)
        hidden = state.read("HIDDEN", "6", MISSING)
        apply_env(target, state.used)

        self.assertEqual((width, hidden), ("4", "6"))
        self.assertEqual(target, {"WIDTH": "4"})

    def test_expands_args(self) -> None:
        call = ast.parse("Net(*dims, **options)").body[0].value

        values = call_args(
            call,
            {"dims": [4], "options": {"hidden": 6, "classes": 2}},
            Path.cwd(),
        )

        self.assertEqual(values, ([4], {"hidden": 6, "classes": 2}))

    def test_invalid_yaml(self) -> None:
        node = ast.parse("yaml.safe_load('[broken')").body[0].value
        with tempfile.TemporaryDirectory() as temporary_directory:
            value = eval_node(node, {}, Path(temporary_directory))

        self.assertIs(value, MISSING)


if __name__ == "__main__":
    unittest.main()
