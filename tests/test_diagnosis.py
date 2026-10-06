from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.harness.protocol import LOCALIZATION_FIELDS
from torchport import diagnosis
from torchport.task import normalize


def _task(root: Path) -> dict:
    for name in ("source", "target"):
        path = root / name
        path.mkdir(parents=True)
        (path / "main.py").write_text("class Net: pass\n", encoding="utf-8")
    return normalize(
        {
            "task_id": "diagnosis",
            "source": {"repo": "source", "entry": "main.py", "model": "Net"},
            "target": {"repo": "target", "entry": "main.py", "model": "Net"},
            "inputs": [{"shape": [1]}],
            "protocol": {"repeats": 1},
        },
        root,
    )


class DiagnosisTests(unittest.TestCase):
    def test_inspect_is_public_task_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = diagnosis.inspect_task(_task(Path(temporary)))

        self.assertEqual(report["kind"], "torchport.inspect")
        self.assertNotIn("oracle", report)
        self.assertEqual(report["inputs"], [{"shape": [1]}])

    def test_trace_and_localize_first_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root / "repos")
            source_dir, target_dir = root / "source-run", root / "target-run"
            source_dir.mkdir()
            target_dir.mkdir()
            np.savez(source_dir / "outputs.npz", output=np.asarray([1.0], dtype=np.float32))
            np.savez(target_dir / "outputs.npz", output=np.asarray([1.0], dtype=np.float32))
            np.savez(
                source_dir / "trace.npz",
                layer=np.asarray([1.0, 2.0], dtype=np.float32),
            )
            np.savez(
                target_dir / "trace.npz",
                layer=np.asarray([1.0, 9.0], dtype=np.float32),
            )
            np.savez(
                source_dir / "module_outputs.npz",
                layer=np.asarray([1.0, 2.0], dtype=np.float32),
                source_only=np.asarray([3.0], dtype=np.float32),
            )
            np.savez(
                target_dir / "isolated.npz",
                layer=np.asarray([1.0, 2.0], dtype=np.float32),
            )

            def fake_run(_task_value, kind, _root, **_kwargs):
                directory = source_dir if kind == "source" else target_dir
                return {
                    "ok": True,
                    "dir": str(directory),
                    "artifacts": {
                        "outputs": str(directory / "outputs.npz"),
                        "trace": str(directory / "trace.npz"),
                        **(
                            {"module_outputs": str(directory / "module_outputs.npz")}
                            if kind == "source"
                            else {"isolated": str(directory / "isolated.npz")}
                        ),
                    },
                    "capture": {"isolated": {"succeeded": 1}} if kind == "target" else {},
                }

            with patch.object(diagnosis, "run_side", fake_run):
                report = diagnosis.trace(task, root / "evidence", trial="trace-0")

            location = diagnosis.localize(report)

        self.assertFalse(report["passed"])
        self.assertTrue(report["isolated"]["passed"])
        self.assertEqual(report["isolated"]["missing_in_candidate"], ["source_only"])
        self.assertEqual(report["isolated"]["capture"]["succeeded"], 1)
        self.assertEqual(location["stage"], "trace")
        self.assertEqual(location["first_divergence"]["name"], "layer")
        self.assertEqual(location["scope"], "layer")
        self.assertEqual(location["invocation"], 1)
        self.assertEqual(location["evidence"]["output"]["mismatched_elements"], 1)
        self.assertEqual(tuple(location), LOCALIZATION_FIELDS)


if __name__ == "__main__":
    unittest.main()
