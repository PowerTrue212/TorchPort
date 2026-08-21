from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from torchport.advise import build_advice
from torchport.auto import AutoRun
from torchport.fail import classify
from torchport.models import MigrationTarget


class FakeAlign:
    def align(self, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "kind": "torchport.repository-certificate",
            "status": "verified",
            "passed": True,
            "spec": {
                "weights": str(kwargs.get("weights")),
                "seed": kwargs["seed"],
            },
        }


class SourceAlign:
    def __init__(self) -> None:
        self.source: Path | None = None

    def align(self, source: str | Path, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self.source = Path(source)
        return {
            "kind": "torchport.repository-certificate",
            "status": "verified",
            "passed": True,
            "spec": {},
        }


class AutoTests(unittest.TestCase):
    def test_auto(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "dynamic"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            run_root = root / "run"

            report = AutoRun(aligner=FakeAlign()).run(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="DynNet",
                shapes=((1, 4),),
                model_args=[4, 6],
                runtime=False,
                run_root=run_root,
                weights="model.pth",
                seed=11,
            )

            self.assertEqual(report["status"], "verified")
            self.assertTrue(report["passed"])
            self.assertTrue(report["check"]["passed"])
            self.assertEqual(report["align"]["run_root"], str(run_root.resolve()))
            self.assertEqual(report["certificate"]["spec"]["weights"], "model.pth")
            self.assertIsNone(report["failure"])
            self.assertIsNone(report["agent"])
            self.assertEqual(report["entry_cert"]["status"], "resolved")
            self.assertEqual(report["entry_cert"]["final"]["ref"], "main:DynNet")
            self.assertTrue(report["advice"]["ready"])
            saved = json.loads(
                (output / ".torchport" / "auto.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["kind"], "torchport.auto-run")
            self.assertEqual(saved["entry_cert"]["kind"], "torchport.entry-certificate")
            self.assertEqual(saved["migration"]["probe"]["status"], "ready")

    def test_agent_record(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "dynamic"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            agent = {
                "mode": "dsh",
                "round": 2,
                "reason": "源码中 Config 固定 width 和 hidden。",
            }

            report = AutoRun(aligner=FakeAlign()).run(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="DynNet",
                shapes=((1, 4),),
                model_args=[4, 6],
                runtime=False,
                align=False,
                agent=agent,
            )

            self.assertEqual(report["agent"], agent)
            self.assertEqual(report["attempts"][0]["agent"], agent)
            saved = json.loads(
                (output / ".torchport" / "auto.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["agent"]["mode"], "dsh")

    def test_no_align(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "dynamic"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "output"

            report = AutoRun(aligner=FakeAlign()).run(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="DynNet",
                shapes=((1, 4),),
                model_args=[4, 6],
                runtime=False,
                align=False,
            )

            self.assertEqual(report["status"], "checked")
            self.assertTrue(report["passed"])
            self.assertIsNone(report["certificate"])
            self.assertTrue(report["advice"]["ready"])
            self.assertEqual(report["align"]["skipped"], "align-disabled")
            self.assertIn("未请求数值对齐", report["advice"]["summary"])

    def test_advice(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "dynamic"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "output"

            report = AutoRun(aligner=FakeAlign()).run(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                runtime=False,
            )

            actions = {item["id"]: item for item in report["advice"]["actions"]}
            self.assertEqual(report["status"], "partial")
            self.assertFalse(report["passed"])
            self.assertEqual(report["failure"]["code"], "CONFIG_UNRESOLVED")
            self.assertFalse(report["advice"]["ready"])
            self.assertEqual(actions["set-args"]["flag"], "--model-kw")
            self.assertEqual(actions["set-args"]["fields"], ["width", "hidden"])
            self.assertEqual(
                json.loads(actions["set-args"]["example"]),
                {"width": 4, "hidden": 6},
            )
            self.assertEqual(actions["set-args"]["hints"], {"width": 4, "hidden": 6})
            self.assertEqual(actions["set-input"]["flag"], "--input")
            self.assertEqual(
                actions["set-input"]["hints"],
                [{"shape": [1, 4], "dtype": "float32"}],
            )
            self.assertNotIn("fix-blockers", actions)
            self.assertEqual(report["align"]["skipped"], "migration-not-converted")

    def test_try_hints(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "dynamic"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"

            report = AutoRun(aligner=FakeAlign()).run(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                runtime=False,
                try_hints=True,
            )

            self.assertEqual(report["status"], "verified")
            self.assertTrue(report["passed"])
            self.assertEqual(Path(report["output"]).name, "output-hints")
            self.assertTrue((output / ".torchport" / "auto.json").is_file())
            self.assertTrue((root / "output-hints" / ".torchport" / "auto.json").is_file())
            self.assertEqual(report["retry"]["config"]["model_kw"], {"width": 4, "hidden": 6})
            self.assertEqual(
                report["retry"]["config"]["inputs"],
                [{"shape": [1, 4], "dtype": "float32"}],
            )
            self.assertEqual(
                [item["status"] for item in report["attempts"]],
                ["partial", "verified"],
            )

    def test_model_advice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """WIDTH = 4

from torch import nn

class PickNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, 2)

    def forward(self, value):
        return self.fc(value)
""",
                encoding="utf-8",
            )

            report = AutoRun(aligner=FakeAlign()).run(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                runtime=False,
            )

            actions = {item["id"]: item for item in report["advice"]["actions"]}
            self.assertEqual(actions["set-model"]["example"], "main.PickNet")
            self.assertEqual(actions["set-model"]["candidates"][0]["name"], "main.PickNet")
            self.assertEqual(actions["set-model"]["candidates"][0]["required"], ["width"])
            self.assertEqual(actions["set-model"]["candidates"][0]["hints"], {"width": 4})
            self.assertEqual(
                actions["set-model"]["candidates"][0]["input_specs"],
                [{"shape": [1, 4], "dtype": "float32"}],
            )

    def test_model_scope_slices_source_for_align(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "main.py").write_text(
                """import argparse
import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
torch.manual_seed(args.seed)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, value):
        return self.fc(value)

def train(loss):
    loss.backward()
""",
                encoding="utf-8",
            )
            aligner = SourceAlign()

            AutoRun(aligner=aligner).run(
                source,
                root / "output",
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
                model="Net",
                shapes=((1, 2),),
                runtime=False,
                run_root=root / "run",
                scope="model",
            )

            assert aligner.source is not None
            sliced = aligner.source / "main.py"
            text = sliced.read_text(encoding="utf-8")
            self.assertNotEqual(aligner.source, source)
            self.assertIn("source_scope", str(aligner.source))
            self.assertNotIn("parse_args", text)
            self.assertNotIn("args.seed", text)
            self.assertNotIn("def train", text)

    def test_residue_advice(self) -> None:
        advice = build_advice(
            {"probe": {"status": "ready"}},
            {
                "passed": False,
                "syntax": {"passed": True, "errors": []},
                "residue": {
                    "passed": False,
                    "imports": [{"file": "model.py", "line": 3, "module": "torch.nn.init"}],
                },
                "runtime": {"attempted": True, "passed": True, "stderr": ""},
                "manifest_blockers": 0,
            },
            None,
            {"requested": False},
        )

        actions = {item["id"]: item for item in advice["actions"]}
        self.assertIn("fix-imports", actions)
        self.assertNotIn("fix-runtime", actions)
        self.assertEqual(actions["fix-imports"]["stage"], "check")

    def test_source_align_failure(self) -> None:
        failure = classify(
            {"status": "converted", "probe": {"status": "ready"}},
            {"passed": True},
            None,
            {"requested": True, "error": "源端探针子进程失败：模型构造参数不匹配"},
        )

        self.assertEqual(failure["code"], "TORCH_RUN_FAILED")


if __name__ == "__main__":
    unittest.main()
