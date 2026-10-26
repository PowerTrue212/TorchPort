from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock

from experiments.harness.remote_executor import RemoteExecutor


class RemoteExecutorTests(unittest.TestCase):
    def _executor(self, root: Path, code: int, report: dict) -> RemoteExecutor:
        executor = RemoteExecutor.__new__(RemoteExecutor)
        executor.bundle = Mock()
        executor.bundle.remote_base.return_value = PurePosixPath("/trusted/task")
        executor.transport = Mock()
        executor.transport.run.return_value = (code, json.dumps(report), "private stderr")
        executor.connect = Mock()
        executor.settings = SimpleNamespace(source_python="/env/python", system_libstdcxx="")
        executor.evidence_dir = root
        executor.sequence = 0
        return executor

    def test_raw_verify_is_retained_only_in_controller_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executor = self._executor(Path(temporary), 3, {"passed": False, "oracle": "/secret"})
            report = executor.call("verify", Path("task.json"), trial="t0")
            command = executor.transport.run.call_args.args[1]
            raw = json.loads((Path(temporary) / "0001-verify.json").read_text("utf-8"))

        self.assertFalse(report["passed"])
        self.assertIn("-m torchport verify /trusted/task/task.json", command)
        self.assertEqual(raw["stdout"]["oracle"], "/secret")
        executor.bundle.sync.assert_called_once()
        executor.connect.return_value.close.assert_called_once()

    def test_evaluate_uses_same_task_remote_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executor = self._executor(Path(temporary), 0, {"passed": True})
            executor.call("evaluate", Path("task.json"), trial="t0", manifest={"run_id": "r1"})
            command = executor.transport.run.call_args.args[1]

        self.assertIn("/trusted/task/evidence/runs/t0/r1/manifest.json", command)

    def test_remote_error_does_not_expose_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executor = self._executor(Path(temporary), 255, {})
            with self.assertRaises(RuntimeError) as context:
                executor.call("verify", Path("task.json"), trial="t0")
            raw = json.loads((Path(temporary) / "0001-verify.json").read_text("utf-8"))

        self.assertNotIn("private stderr", str(context.exception))
        self.assertEqual(raw["exit_code"], 255)
        self.assertEqual(raw["stderr"], "private stderr")


if __name__ == "__main__":
    unittest.main()
