"""Harness 编排测试：用假子进程模拟两端采集，覆盖超时、陈旧产物、oracle 不稳定、身份不符与重复判定。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from torchport import oracle as harness
from torchport.task import normalize, tree_hash


def _task(base: Path, repeats: int = 3, target: str = "target") -> dict:
    source_root = base / "source"
    target_root = base / target
    source_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)
    if not (source_root / "main.py").exists():
        (source_root / "main.py").write_text("class Net: pass\n", encoding="utf-8")
    if not (target_root / "main.py").exists():
        (target_root / "main.py").write_text("class Net: pass\n", encoding="utf-8")
    return normalize(
        {
            "task_id": "tiny",
            "source": {"repo": "source", "entry": "main.py", "model": "Net"},
            "target": {"repo": target, "entry": "main.py", "model": "Net"},
            "inputs": [{"shape": [1, 2]}],
            "protocol": {"repeats": repeats, "timeout": 5},
        },
        base,
    )


class FakeExec:
    """按调用顺序决定每次运行的行为：ok / timeout / crash / 输出值。"""

    def __init__(self, target=(1.0,), source=(1.0,), timeout=False, crash=False):
        self.target, self.source = list(target), list(source)
        self.timeout, self.crash = timeout, crash
        self.calls = {"source": 0, "target": 0}

    def __call__(self, side, args, *, run_dir, cache, timeout):
        job = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        kind = job["kind"]
        index = self.calls[kind]
        self.calls[kind] += 1
        record = {"ok": True, "exit_code": 0, "timeout": False, "cache": str(cache)}
        if kind == "target" and (self.timeout or self.crash):
            record.update(ok=False, exit_code=None if self.timeout else 1, timeout=self.timeout)
            record["stderr_tail"] = "boom"
            return record
        values = self.source if kind == "source" else self.target
        value = values[min(index, len(values) - 1)]
        inputs = {"input": np.ones((1, 2), dtype=np.float32)}
        output = np.full((1, 2), value, dtype=np.float32)
        np.savez(run_dir / "outputs.npz", **inputs, output=output)
        if kind == "source":
            np.savez(run_dir / "inputs.npz", **inputs)
            np.savez(run_dir / "state.npz", w=np.ones(2, dtype=np.float32))
        capture = {
            "kind": kind,
            "mode": job["mode"],
            "weights": 1,
            "loaded": 1,
            "checkpoint": {"requested": False, "complete": True} if kind == "source" else None,
            "hashes": {"input": "same", "output": str(value)},
        }
        (run_dir / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
        return record


class HarnessTests(unittest.TestCase):
    def _verify(self, fake: FakeExec, repeats: int = 3) -> dict:
        with tempfile.TemporaryDirectory() as tmp, patch.object(harness, "execute", fake):
            root = Path(tmp)
            return harness.verify(_task(root / "repos", repeats), root / "harness")

    def test_correct_candidate_passes_all_repeats(self) -> None:
        fake = FakeExec()
        report = self._verify(fake)
        self.assertTrue(report["passed"])
        self.assertEqual(report["repeats"]["passed_runs"], 3)
        self.assertEqual(fake.calls, {"source": 3, "target": 3})
        self.assertEqual(report["physical_cost"]["executions"], 6)

    def test_wrong_output_fails(self) -> None:
        report = self._verify(FakeExec(target=(999.0,)))
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure_code"], "PARITY_FAILED")

    def test_one_flaky_run_fails_not_best_of(self) -> None:
        report = self._verify(FakeExec(target=(1.0, 999.0, 1.0)))
        self.assertFalse(report["passed"])
        self.assertEqual(report["repeats"]["passed_runs"], 1)

    def test_timeout_and_crash_fail(self) -> None:
        for fake in (FakeExec(timeout=True), FakeExec(crash=True)):
            report = self._verify(fake)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failure_code"], "RUN_FAILED")

    def test_unstable_oracle_rejected(self) -> None:
        report = self._verify(FakeExec(source=(1.0, 2.0)))
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure_code"], "ORACLE_UNSTABLE")

    def test_stale_artifacts_not_reused_after_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = _task(root / "repos", 1)
            harness_root = root / "harness"
            with patch.object(harness, "execute", FakeExec()):
                self.assertTrue(harness.verify(task, harness_root)["passed"])
            with patch.object(harness, "execute", FakeExec(crash=True)):
                report = harness.verify(task, harness_root, trial="t1")
            self.assertFalse(report["passed"])

    def test_oracle_reused_and_tamper_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = _task(root / "repos", 1)
            harness_root = root / "harness"
            fake = FakeExec()
            with patch.object(harness, "execute", fake):
                first = harness.build_oracle(task, harness_root)
                harness.build_oracle(task, harness_root)
            self.assertEqual(fake.calls["source"], 1)
            (Path(first["dir"]) / "outputs.npz").write_bytes(b"tampered")
            self.assertEqual(harness.load_oracle(task, harness_root)["reason"], "ORACLE_INVALID")

    def test_source_edit_rebuilds_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(harness, "execute", FakeExec()) as run:
            root = Path(tmp)
            repos, harness_root = root / "repos", root / "harness"
            before = _task(repos, 1)
            first = harness.build_oracle(before, harness_root)
            (repos / "source" / "main.py").write_text("class Net: changed = True\n", encoding="utf-8")
            after = _task(repos, 1)
            second = harness.build_oracle(after, harness_root)

        self.assertNotEqual(before["digest"], after["digest"])
        self.assertNotEqual(first["dir"], second["dir"])
        self.assertEqual(run.calls["source"], 2)

    def test_evaluate_does_not_start_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = _task(root / "repos", 1)
            meta = harness.load_oracle(task, root / "harness")
        self.assertEqual(meta["reason"], "ORACLE_MISSING")
        self.assertFalse(harness.evaluate(meta, {}, task["protocol"])["passed"])

    def test_identity_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(harness, "execute", FakeExec()):
            root = Path(tmp)
            task = _task(root / "repos", 1)
            harness_root = root / "harness"
            meta = harness.build_oracle(task, harness_root)
            run = harness.run_side(task, "target", harness_root, oracle=Path(meta["dir"]))
            run["task_digest"] = "other"
            with self.assertRaises(ValueError):
                harness.evaluate(meta, run, task["protocol"])

    def test_different_target_cannot_reuse_candidate_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(harness, "execute", FakeExec()):
            root = Path(tmp)
            repos, harness_root = root / "repos", root / "harness"
            first_task = _task(repos, 1, "target-a")
            target_b = repos / "target-b"
            target_b.mkdir(parents=True)
            (target_b / "main.py").write_text(
                "class Net: implementation = 'different'\n", encoding="utf-8"
            )
            second_task = _task(repos, 1, "target-b")
            oracle = harness.build_oracle(first_task, harness_root)
            run = harness.run_side(
                first_task, "target", harness_root, oracle=Path(oracle["dir"])
            )

            self.assertEqual(first_task["digest"], second_task["digest"])
            self.assertNotEqual(first_task["target_identity"], second_task["target_identity"])
            with self.assertRaisesRegex(ValueError, "当前 target"):
                harness.evaluate(
                    oracle, run, second_task["protocol"], second_task["target_identity"]
                )

    def test_tree_hash_prunes_generated_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
            skipped = root / ".venv" / "lib"
            skipped.mkdir(parents=True)
            (skipped / "huge.py").write_text("VALUE = 2\n", encoding="utf-8")
            before = tree_hash(root)
            (skipped / "huge.py").write_text("VALUE = 3\n", encoding="utf-8")
            after = tree_hash(root)

        self.assertEqual(before, after)

    def test_trials_use_separate_caches(self) -> None:
        a = harness._cache(Path("r"), "t0", "target", None)
        b = harness._cache(Path("r"), "t1", "target", None)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
