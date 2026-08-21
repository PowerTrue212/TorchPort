from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from torchport.probe import AlignRun, infer_probe
from torchport.probe.finder import model_choices
from torchport.probe.merge import merge_probe
from torchport.probe.model import _config_args, _lookup, load_scope


class FakeRun(AlignRun):
    @staticmethod
    def _run(python, args, paths, *, target, runtime):
        del python, paths, runtime
        output = Path(args[args.index("--output") + 1])
        state = Path(args[args.index("--state") + 1])
        values = {
            "input": np.ones((1, 4), dtype=np.float32),
            "fc": np.full((1, 2), 2 if target else 1, dtype=np.float32),
            "output": np.full((1, 2), 3, dtype=np.float32),
        }
        if args.count("--input") > 1:
            values["input.1"] = np.ones((1, 4), dtype=np.float32)
        np.savez(output, **values)
        if not target:
            np.savez(state, weight=np.ones((2, 4), dtype=np.float32))
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "weights": 1,
                    "loaded": 1,
                    "checkpoint": {
                        "requested": not target and "--weights" in args,
                        "complete": True,
                    },
                    "tensors": len(values),
                    "framework": {"name": "fake", "version": "1"},
                    "hashes": {
                        "input": "same",
                        "output": "same",
                        "archive": "target" if target else "source",
                        "weights": "weights",
                    },
                }
            ),
            encoding="utf-8",
        )


class ProbeTests(unittest.TestCase):
    def test_align(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (target / ".torchport").mkdir(parents=True)
            (target / ".torchport" / "probe.json").write_text(
                json.dumps(
                    {
                        "entry": "main.py",
                        "model": "TinyNet",
                        "shape": [1, 4],
                        "dtype": "float32",
                        "weights": "model.pth",
                    }
                ),
                encoding="utf-8",
            )

            report = FakeRun().align(source, target, run_root=root / "run")

            self.assertTrue(report["passed"])
            self.assertEqual(report["spec"]["weights"], "model.pth")
            self.assertTrue(report["checkpoint"]["requested"])
            self.assertFalse(report["strict"]["passed"])
            self.assertEqual(report["strict"]["first"]["name"], "fc")
            self.assertTrue((root / "run" / "certificate.json").is_file())

    def test_infer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "main.py").write_text(
                """from torch import nn
from torchvision import datasets

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)

    def forward(self, value):
        return self.conv(value)

data = datasets.MNIST("data")
""",
                encoding="utf-8",
            )

            spec = infer_probe(root, "main.py")

            self.assertEqual(spec["status"], "ready")
            self.assertEqual(spec["model"], "Net")
            self.assertEqual(spec["shape"], [1, 1, 28, 28])

    def test_load_scope_isolates_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "main.py").write_text(
                """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--flag', action='store_true')
args = parser.parse_args()
value = args.flag
""",
                encoding="utf-8",
            )
            original = list(sys.argv)
            sys.argv = ["torchport", "capture", "--unexpected"]
            try:
                scope = load_scope(root, "main.py")
            finally:
                restored = list(sys.argv)
                sys.argv = original

            self.assertFalse(scope["value"])
            self.assertEqual(restored, ["torchport", "capture", "--unexpected"])

    def test_multi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (target / ".torchport").mkdir(parents=True)
            (target / ".torchport" / "probe.json").write_text(
                json.dumps({"entry": "main.py", "model": "PairNet"}),
                encoding="utf-8",
            )

            report = FakeRun().align(
                source,
                target,
                run_root=root / "run",
                shapes=((1, 4), (1, 4)),
                dtypes=("float32",),
                model_args=[4],
            )

            self.assertEqual(len(report["spec"]["inputs"]), 2)
            self.assertEqual(report["spec"]["args"], [4])
            self.assertTrue(report["tensors"]["input_identical"])

    def test_weights_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (target / ".torchport").mkdir(parents=True)
            (target / ".torchport" / "probe.json").write_text(
                json.dumps(
                    {
                        "entry": "main.py",
                        "model": "Net",
                        "shape": [1, 4],
                        "weight_candidates": ["best.pth", "last.pth"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "多个检查点"):
                FakeRun().align(source, target, run_root=root / "run")

    def test_struct(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "struct"

        spec = infer_probe(root, "main.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["model"], "BatchNet")
        self.assertEqual(len(spec["inputs"]), 3)
        self.assertEqual(spec["inputs"][0]["key"], "left")
        self.assertEqual(spec["inputs"][1]["key"], "right")
        self.assertEqual(spec["inputs"][2]["arg"], "scale")

    def test_nested(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "nested"

        spec = infer_probe(root, "main.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["model"], "NestNet")
        self.assertEqual(len(spec["inputs"]), 2)
        self.assertEqual(spec["inputs"][0]["pack"], "tuple")
        self.assertEqual(spec["inputs"][0]["item"], 0)
        self.assertEqual(spec["inputs"][1]["item"], 1)

    def test_package(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "package"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "train.py")
        self.assertEqual(spec["model"], "build")
        self.assertEqual(spec["shape"], [1, 4])
        self.assertEqual(spec["args"], [])

    def test_config(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "config"

        spec = infer_probe(root, "main.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["model"], "ConfigNet")
        self.assertEqual(spec["shape"], [1, 4])
        self.assertEqual(
            spec["kwargs"],
            {"width": 4, "hidden": 6, "classes": 2},
        )
        self.assertEqual(spec["required"], [])

    def test_args(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "args"

        spec = infer_probe(root, "main.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["model"], "ArgNet")
        self.assertEqual(spec["shape"], [1, 4])
        self.assertEqual(spec["args"], [4])
        self.assertEqual(
            spec["kwargs"],
            {"hidden": 6, "classes": 2, "enabled": False},
        )

    def test_env(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "env"

        missing = infer_probe(root, "main.py")
        spec = infer_probe(root, "main.py", {"TP_WIDTH": "4"})

        self.assertEqual(missing["status"], "needs-args")
        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["args"], [4])
        self.assertEqual(spec["kwargs"], {"hidden": 6})
        self.assertEqual(spec["env"], {"TP_WIDTH": "4", "TP_HIDDEN": None})

    def test_merge(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "dynamic"

        missing = infer_probe(root, "main.py")
        spec = merge_probe(
            missing,
            entry="main.py",
            model="DynNet",
            inputs=[{"shape": [1, 4], "dtype": "float32"}],
            model_args=[4, 6],
        )

        self.assertEqual(missing["status"], "needs-args")
        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["model"], "DynNet")
        self.assertEqual(spec["args"], [4, 6])
        self.assertEqual(spec["shape"], [1, 4])

    def test_merge_factory_model(self) -> None:
        base = {
            "entry": "main.py",
            "model": "Net",
            "required": ["width"],
            "inputs": [],
            "candidates": [
                {
                    "name": "main.make",
                    "kind": "factory",
                    "required": [],
                    "input_specs": [{"shape": [1, 4], "dtype": "float32"}],
                }
            ],
        }

        spec = merge_probe(base, model="make", shapes=((1, 4),))

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["model"], "make")
        self.assertEqual(spec["required"], [])

    def test_choices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "main.py").write_text(
                """WIDTH = 4

from torch import nn

class PickNet(nn.Module):
    def __init__(self, width, hidden=6):
        super().__init__()
        self.fc = nn.Linear(width, hidden)

    def forward(self, value):
        return self.fc(value)

def make(width):
    return PickNet(width)
""",
                encoding="utf-8",
            )

            spec = infer_probe(root, "main.py")
            choices = model_choices(root, "main.py")

            names = {item["name"]: item for item in choices}
            self.assertEqual(spec["status"], "needs-args")
            self.assertIn("main.PickNet", names)
            self.assertIn("main.make", names)
            self.assertEqual(names["main.PickNet"]["required"], ["width"])
            self.assertEqual(names["main.PickNet"]["defaults"], {"hidden": 6})
            self.assertEqual(names["main.PickNet"]["hints"], {"width": 4})
            self.assertEqual(names["main.PickNet"]["example_kw"], {"width": 4})
            self.assertEqual(names["main.make"]["returns"], "main.PickNet")

    def test_lookup_module_shadow(self) -> None:
        class Net:
            pass

        module = types.ModuleType("shadowmod")
        module.Net = Net
        sys.modules["shadowmod"] = module
        try:
            found = _lookup({"shadowmod": lambda: None}, "shadowmod.Net")
        finally:
            del sys.modules["shadowmod"]

        self.assertIs(found, Net)

    def test_config_args(self) -> None:
        args, kwargs = _config_args(
            [{"model": {"type": "tiny", "width": 4}}],
            {"train": {"steps": 1}},
        )

        self.assertEqual(args[0].model.width, 4)
        self.assertEqual(args[0]["model"]["type"], "tiny")
        self.assertEqual(dict(args[0].model), {"type": "tiny", "width": 4})
        self.assertEqual(kwargs["train"].steps, 1)


if __name__ == "__main__":
    unittest.main()
