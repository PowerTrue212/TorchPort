from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.harness.aggregate import aggregate


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class P2AggregateTests(unittest.TestCase):
    def test_preserves_success_failure_and_missing_trials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "experiment.json"
            _write(
                manifest,
                {
                    "experiment_id": "smoke",
                    "trials": [
                        {"trial_id": "v0", "condition": "verdict", "path": "v0"},
                        {"trial_id": "d0", "condition": "diagnostic", "path": "d0"},
                        {"trial_id": "l0", "condition": "localized", "path": "l0"},
                    ],
                },
            )
            _write(
                root / "v0" / "result.json",
                {
                    "valid": True,
                    "success": True,
                    "passed": True,
                    "stop_reason": "agent_completed",
                    "total_wall_seconds": 12.0,
                    "physical_evaluation_cost": {
                        "total": {
                            "executions": 2,
                            "execution_wall_seconds": 5.0,
                            "cuda_execution_seconds": 5.0,
                        }
                    },
                },
            )
            (root / "v0" / "events.jsonl").write_text(
                json.dumps(
                    {
                        "event": "model_response",
                        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write(
                root / "d0" / "result.json",
                {
                    "valid": True,
                    "success": False,
                    "passed": False,
                    "stop_reason": "oracle_query_budget_exhausted",
                    "total_wall_seconds": 20.0,
                    "physical_evaluation_cost": {
                        "total": {
                            "executions": 3,
                            "execution_wall_seconds": 8.0,
                            "cuda_execution_seconds": 8.0,
                        }
                    },
                },
            )

            report = aggregate(manifest)

        self.assertFalse(report["complete"])
        self.assertEqual(report["scheduled_trials"], 3)
        self.assertEqual(report["completed_trials"], 2)
        self.assertEqual(report["success_at_budget"], {"successes": 1, "valid_trials": 2, "rate": 0.5})
        self.assertEqual(report["time_to_success_seconds"]["median"], 12.0)
        self.assertEqual(report["tokens"]["total_tokens"], 12)
        self.assertEqual(report["physical_evaluation_cost"]["executions"], 5)
        self.assertEqual(report["stop_reasons"]["not_run"], 1)
        self.assertFalse(report["performance_available"])

    def test_aggregates_tool_costs_milestones_and_candidate_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(
                root / "experiment.json",
                {
                    "experiment_id": "tool-costs",
                    "trials": [
                        {"trial_id": "v0", "condition": "verdict", "path": "v0"}
                    ],
                },
            )
            _write(
                root / "v0" / "result.json",
                {
                    "valid": True,
                    "success": True,
                    "passed": True,
                    "stop_reason": "agent_completed",
                    "total_wall_seconds": 12.0,
                    "physical_evaluation_cost": {
                        "total": {
                            "executions": 5,
                            "execution_wall_seconds": 11.0,
                            "cuda_execution_seconds": 0.0,
                        }
                    },
                },
            )
            events = [
                {"event": "agent_started", "timestamp": 100.0},
                {
                    "event": "tool_call",
                    "sequence": 1,
                    "tool": "apply_patch",
                    "timestamp": 101.0,
                    "duration_seconds": 0.2,
                    "candidate_before": "a",
                    "candidate_after": "b",
                    "physical_evaluation_cost": {
                        "executions": 0,
                        "execution_wall_seconds": 0.0,
                        "cuda_execution_seconds": 0.0,
                    },
                },
                {
                    "event": "tool_call",
                    "sequence": 2,
                    "tool": "probe",
                    "timestamp": 103.0,
                    "duration_seconds": 2.0,
                    "candidate_before": "b",
                    "candidate_after": "b",
                    "physical_evaluation_cost": {
                        "executions": 1,
                        "execution_wall_seconds": 2.0,
                        "cuda_execution_seconds": 0.0,
                    },
                },
                {
                    "event": "tool_call",
                    "sequence": 3,
                    "tool": "run",
                    "timestamp": 107.0,
                    "duration_seconds": 3.0,
                    "candidate_before": "b",
                    "candidate_after": "b",
                    "physical_evaluation_cost": {
                        "executions": 2,
                        "execution_wall_seconds": 5.0,
                        "cuda_execution_seconds": 0.0,
                    },
                },
                {
                    "event": "tool_call",
                    "sequence": 4,
                    "tool": "verify",
                    "timestamp": 110.0,
                    "duration_seconds": 0.5,
                    "candidate_before": "b",
                    "candidate_after": "b",
                    "physical_evaluation_cost": {
                        "executions": 4,
                        "execution_wall_seconds": 9.0,
                        "cuda_execution_seconds": 0.0,
                    },
                },
            ]
            (root / "v0" / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            _write(root / "v0" / "tool-results" / "0003.json", {"ok": True, "data": {"ok": True}})
            _write(
                root / "v0" / "tool-results" / "0004.json",
                {"ok": True, "data": {"passed": True}},
            )

            report = aggregate(root / "experiment.json")

        row = report["trials"][0]
        self.assertEqual(report["tool_calls_by_name"]["probe"], 1)
        self.assertEqual(report["tool_time_seconds_by_name"]["run"], 3.0)
        self.assertEqual(
            report["physical_evaluation_cost_by_tool"]["verify"]["executions"], 2
        )
        self.assertEqual(
            report["physical_evaluation_cost_by_tool"]["probe"]["execution_wall_seconds"],
            2.0,
        )
        self.assertEqual(report["candidate_revisions"], 1)
        self.assertEqual(row["unique_candidate_identities"], 2)
        self.assertEqual(row["milestones_seconds"]["first_run_success"], 7.0)
        self.assertEqual(row["milestones_seconds"]["first_development_verify_pass"], 10.0)
        self.assertEqual(report["milestone_medians_seconds"]["official_verify_complete"], 12.0)


if __name__ == "__main__":
    unittest.main()
