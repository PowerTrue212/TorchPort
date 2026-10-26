from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.harness.agents import audit_trial, preflight


class PreflightTests(unittest.TestCase):
    def test_readable_forbidden_path_and_import_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = root / "oracle"
            secret.mkdir()
            workspace = root / "workspace"
            workspace.mkdir()
            report = preflight.run({
                "python": sys.executable,
                "workspace": str(workspace),
                "forbidden_paths": [str(secret), str(root / "missing")],
                "forbidden_imports": ["json", "module_that_does_not_exist_xyz"],
            })

        self.assertFalse(report["passed"])
        self.assertIn(f"path:{secret}", report["failures"])
        self.assertIn("import:json", report["failures"])
        self.assertNotIn("import:module_that_does_not_exist_xyz", report["failures"])
        self.assertTrue(report["workspace_writable"])
        missing = [item for item in report["paths"] if item["path"].endswith("missing")][0]
        self.assertEqual(missing["reason"], "not_found")

    def test_all_blocked_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = preflight.run({
                "python": sys.executable,
                "workspace": temporary,
                "forbidden_paths": [str(Path(temporary) / "absent")],
                "forbidden_imports": ["module_that_does_not_exist_xyz"],
            })
        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])


class AuditTests(unittest.TestCase):
    def _log(self, path: Path, events: list[dict]) -> None:
        path.write_text("".join(json.dumps(e) + "\n" for e in events) + '{"truncated', "utf-8")

    def test_plant_rejects_git_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "candidate"
            (repo / ".git").mkdir(parents=True)
            with self.assertRaises(ValueError):
                audit_trial.plant([repo], Path(temporary) / "record.json")
            self.assertFalse((repo / audit_trial.CANARY_NAME).exists())

    def test_scan_flags_canary_forbidden_path_and_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root / "controller"
            protected.mkdir()
            workspace = root / "workspace"
            workspace.mkdir()
            record = root / "record.json"
            canary = audit_trial.plant([protected], record)["canaries"][0]["value"]
            log = root / "agent.jsonl"
            self._log(log, [
                {"type": "status", "phase": "step_start"},
                {"type": "tool_result", "result": f"Reference id: {canary}"},
                {"type": "status", "phase": "step_start"},
                {"type": "tool_call", "input": {"command": f"type {workspace}\\run.py"}},
                {"type": "tool_call", "input": {"command": "cat D:\\keyan\\TorchPort\\src\\torchport\\x.py"}},
                {"type": "tool_call", "input": {"url": "https://github.com/uyzhang/JCLIP/blob/main/a.py"}},
                {"type": "tool_call", "input": {"command": "C:\\Python\\python.exe -V"}},
            ])
            report = audit_trial.scan(
                log, workspace, record,
                forbidden_paths=["src/torchport"], forbidden_urls=["github.com/uyzhang/jclip"],
                allowed_paths=[], max_steps=1,
            )

        kinds = {item["kind"] for item in report["hits"]}
        self.assertEqual(kinds, {"canary", "forbidden_path", "forbidden_url"})
        self.assertTrue(report["scope_violation"])
        self.assertEqual(report["suggested_stop_reason"], "scope_violation")
        self.assertEqual(report["model_steps"], 2)
        self.assertTrue(report["steps_exceeded"])
        self.assertEqual(report["truncated_lines"], 1)
        outside = " ".join(report["outside_workspace_paths"])
        self.assertIn("python.exe", outside)
        self.assertNotIn(str(workspace), outside)

    def test_clean_log_has_no_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            log = root / "agent.jsonl"
            log.write_text(json.dumps({"type": "tool_call",
                                       "input": {"command": f"ls {workspace}"}}) + "\n", "utf-8")
            report = audit_trial.scan(log, workspace, None, ["src/torchport"], [], [], None)
        self.assertFalse(report["scope_violation"])
        self.assertEqual(report["hits"], [])

    def test_escaped_newlines_and_urls_are_not_windows_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "agent.jsonl"
            log.write_text(json.dumps({
                "type": "tool_result",
                "message": "error:\nretry\r\nhttps://github.com/openai/CLIP",
                "command": "C:\\Python\\python.exe -V",
            }) + "\n", encoding="utf-8")
            report = audit_trial.scan(log, Path("/work"), None, [], [], [], None)
        self.assertEqual(list(report["outside_workspace_paths"]), ["C:\\Python\\python.exe"])

    def test_container_workspace_path_is_kept_posix_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "agent.jsonl"
            log.write_text(json.dumps({
                "type": "tool_call",
                "input": {"command": "cat /work/run.py /work/.scratch/data/model.bin"},
            }) + "\n", encoding="utf-8")
            report = audit_trial.scan(log, Path("/work"), None, [], [], [], None)
        self.assertEqual(report["outside_workspace_paths"], {})

    def test_forbidden_path_requires_component_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "agent.jsonl"
            log.write_text(json.dumps({
                "type": "tool_call", "input": {"command": "ls /mnt/data/cache"},
            }) + "\n", encoding="utf-8")
            report = audit_trial.scan(log, Path("/work"), None, ["/mnt/d/"], [], [], None)
            self.assertFalse(report["scope_violation"])
            log.write_text(json.dumps({
                "type": "tool_call", "input": {"command": "ls /mnt/d/secret"},
            }) + "\n", encoding="utf-8")
            report = audit_trial.scan(log, Path("/work"), None, ["/mnt/d/"], [], [], None)
            self.assertTrue(report["scope_violation"])


if __name__ == "__main__":
    unittest.main()
