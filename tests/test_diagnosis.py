from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.harness.protocol import LOCALIZATION_FIELDS
from torchport import diagnosis
from torchport.diagnosis import tracing as diagnosis_tracing
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
            np.savez(
                target_dir / "isolated_repeat.npz",
                layer=np.asarray([1.0, 2.00005], dtype=np.float32),
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
                            else {
                                "isolated": str(directory / "isolated.npz"),
                                "isolated_repeat": str(directory / "isolated_repeat.npz"),
                            }
                        ),
                    },
                    "capture": {"isolated": {"succeeded": 1}} if kind == "target" else {},
                }

            with patch.object(diagnosis_tracing, "run_side", fake_run):
                report = diagnosis.trace(task, root / "evidence", trial="trace-0")

            location = diagnosis.localize(report)

        self.assertFalse(report["passed"])
        self.assertTrue(report["isolated"]["passed"])
        self.assertEqual(report["isolated"]["missing_in_candidate"], ["source_only"])
        self.assertEqual(report["isolated"]["capture"]["succeeded"], 1)
        self.assertFalse(report["isolated"]["repeatability"]["passed"])
        self.assertEqual(
            report["isolated"]["repeatability"]["first_divergence"]["name"],
            "layer",
        )
        verification_repeatability = report["isolated"][
            "repeatability_at_verification_tolerance"
        ]
        self.assertTrue(verification_repeatability["passed"])
        self.assertEqual(verification_repeatability["tolerance_basis"], "fp32-inference")
        self.assertEqual(location["stage"], "trace")
        self.assertEqual(location["first_divergence"]["name"], "layer")
        self.assertEqual(location["scope"], "layer")
        self.assertEqual(location["invocation"], 1)
        self.assertEqual(location["evidence"]["output"]["mismatched_elements"], 1)
        self.assertEqual(tuple(location), LOCALIZATION_FIELDS)


    def test_failed_source_trace_reports_reason_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root / "repos")
            failed = {
                "ok": False,
                "dir": str(root / "source-run"),
                "artifacts": {},
                "execution": {"stderr_tail": "warning: noise\ntorchport: [Errno 28] No space left on device"},
            }
            with patch.object(diagnosis_tracing, "run_side", lambda *_a, **_k: failed):
                report = diagnosis.trace(task, root / "evidence", trial="trace-0")
            location = diagnosis.localize(report)

        self.assertEqual(report["failure_code"], "RUN_FAILED")
        self.assertEqual(report["stage"], "source")
        self.assertIs(report["source_run"], failed)
        self.assertIn("No space left on device", report["reason"])
        self.assertIn("No space left on device", location["reason"])

    def test_focus_modules_window_around_first_divergence(self) -> None:
        from torchport.execution import write_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(diagnosis.focus_modules(root, "t0"))
            tensors = [{"name": f"m{i}", "position": i} for i in range(20)]
            tensors += [
                {"name": "blk.0", "position": 20},
                {"name": "output.0", "position": 21},
            ]
            e1 = {"first_divergence": {"name": "blk.0", "position": 20}, "tensors": tensors}
            write_json(root / "verify" / "t0.json", {"e1": e1})
            focus = diagnosis.focus_modules(root, "t0", window=3)

        # 窗口内的模块，多输出张量 `blk.0` 同时带上模块 `blk`；输出张量与窗口外的不进入
        self.assertEqual(focus, ["blk", "blk.0", "m17", "m18", "m19"])

    def test_focused_trace_separates_frozen_oracle_from_module_evidence(self) -> None:
        from torchport.execution import write_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root / "repos")
            evidence = root / "evidence"
            write_json(
                evidence / "verify" / "t0.json",
                {
                    "e1": {
                        "first_divergence": {"name": "block", "position": 0},
                        "tensors": [{"name": "block", "position": 0}],
                    }
                },
            )
            frozen = evidence / "oracle" / "frozen"
            source_dir, target_dir = evidence / "source-run", evidence / "target-run"
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            for directory in (source_dir, target_dir):
                np.savez(directory / "outputs.npz", output=np.asarray([1.0], dtype=np.float32))
                np.savez(directory / "trace.npz", block=np.asarray([1.0], dtype=np.float32))
            calls = []

            def fake_run(_task_value, kind, _root, **kwargs):
                calls.append((kind, kwargs))
                directory = source_dir if kind == "source" else target_dir
                return {
                    "ok": True,
                    "dir": str(directory),
                    "artifacts": {
                        "outputs": str(directory / "outputs.npz"),
                        "trace": str(directory / "trace.npz"),
                    },
                    "execution": {},
                }

            with (
                patch.object(
                    diagnosis_tracing,
                    "build_oracle",
                    return_value={"ok": True, "dir": str(frozen), "physical_cost": {}},
                ),
                patch.object(diagnosis_tracing, "run_side", fake_run),
            ):
                diagnosis.trace(task, evidence, trial="t0")

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "source")
        self.assertEqual(calls[0][1]["oracle"], frozen)
        self.assertEqual(calls[0][1]["focus"], ["block"])
        self.assertEqual(calls[1][0], "target")
        self.assertEqual(calls[1][1]["oracle"], frozen)
        self.assertEqual(calls[1][1]["module_oracle"], source_dir)

    def test_trace_refuses_to_start_when_disk_is_low(self) -> None:
        from collections import namedtuple

        usage = namedtuple("usage", "total used free")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root / "repos")
            never = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not run"))  # noqa: E731
            with (
                patch.object(diagnosis_tracing.shutil, "disk_usage", lambda _p: usage(10, 9, 1024)),
                patch.object(diagnosis_tracing, "run_side", never),
            ):
                report = diagnosis.trace(task, root / "evidence", trial="t0")
            location = diagnosis.localize(report)

        self.assertEqual(report["failure_code"], "DISK_LOW")
        self.assertIn("可用空间", location["reason"])


if __name__ == "__main__":
    unittest.main()


def test_trace_tolerance_falls_back_for_envelope_profiles() -> None:
    # 噪声包络协议没有固定容差；trace 的输出比较不能因此抛 "未知容差配置"
    for profile in ("noise-envelope", "noise-envelope-v3"):
        assert diagnosis._tolerance_profile({"protocol": {"profile": profile}}) == "fp32-inference"
    assert diagnosis._tolerance_profile({"protocol": {"profile": "operator-strict"}}) == "operator-strict"
