from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from torchport.lab import build_lab


class LabTests(unittest.TestCase):
    def test_summarizes_auto_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first" / ".torchport"
            second = root / "second" / ".torchport"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            _write(
                first / "auto.json",
                "first",
                "resolved",
                "ready",
                "main:Net",
                None,
                None,
            )
            _write(
                second / "auto.json",
                "second",
                "unresolved",
                "needs-input",
                "model:Block",
                "CONFIG_UNRESOLVED",
                {"mode": "dsh", "round": 2},
            )

            report = build_lab([root])

            self.assertEqual(report["summary"]["total"], 2)
            self.assertEqual(report["summary"]["with_agent"], 1)
            self.assertEqual(report["summary"]["entry_found"], 2)
            self.assertEqual(report["summary"]["probe_ready"], 1)
            self.assertEqual(report["summary"]["by_mode"], {"auto": 1, "dsh": 1})
            self.assertEqual(report["summary"]["groups"]["auto"]["probe_ready"], 1)
            self.assertEqual(report["summary"]["groups"]["dsh"]["with_agent"], 1)
            self.assertEqual(
                report["summary"]["by_failure"],
                {"CONFIG_UNRESOLVED": 1, "none": 1},
            )

    def test_keeps_best_attempt_per_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "out" / ".torchport"
            retry = root / "out-hints" / ".torchport"
            first.mkdir(parents=True)
            retry.mkdir(parents=True)
            _write(
                first / "auto.json",
                "same",
                "unresolved",
                "needs-input",
                "main:Net",
                "CONFIG_UNRESOLVED",
                None,
            )
            _write(
                retry / "auto.json",
                "same",
                "resolved",
                "ready",
                "main:Net",
                "JITTOR_CONVERT_FAILED",
                None,
            )

            report = build_lab([root])

            self.assertEqual(report["summary"]["total"], 1)
            self.assertEqual(report["rows"][0]["probe_status"], "ready")

    def test_filters_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first" / ".torchport"
            second = root / "second" / ".torchport"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            _write(first / "auto.json", "first", "resolved", "ready", "a:Net", None, None)
            _write(second / "auto.json", "second", "resolved", "ready", "b:Net", None, None)

            report = build_lab([root], names=["second"])

            self.assertEqual(report["summary"]["total"], 1)
            self.assertEqual(report["rows"][0]["name"], "second")


def _write(
    path: Path,
    source: str,
    entry_status: str,
    probe_status: str,
    ref: str | None,
    failure: str | None,
    agent: dict[str, object] | None,
) -> None:
    value = {
        "source": source,
        "status": "failed" if failure else "checked",
        "passed": failure is None,
        "entry_cert": {
            "status": entry_status,
            "probe_status": probe_status,
            "final": {"ref": ref},
        },
        "failure": None if failure is None else {"code": failure, "stage": "entry"},
        "agent": agent,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
