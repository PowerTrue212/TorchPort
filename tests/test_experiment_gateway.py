from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.harness.gateway import HarnessGateway
from experiments.harness.protocol import DIAGNOSTIC_FIELDS, FeedbackLevel, TrialProtocol


def _fixture(
    root: Path,
    feedback: FeedbackLevel = FeedbackLevel.VERDICT,
    queries: int = 2,
    *,
    candidate_entry: str | None = None,
):
    source, target = root / "source", root / "target"
    source.mkdir()
    target.mkdir()
    (source / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (target / "main.py").write_text("VALUE = 0\n", encoding="utf-8")
    if candidate_entry:
        (target / candidate_entry).write_text("from main import VALUE\n", encoding="utf-8")
    task = root / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "gateway",
                "source": {"repo": "source", "entry": "main.py", "model": "Net"},
                "target": {"repo": "target", "entry": "main.py", "model": "Net"},
                "inputs": [{"shape": [1]}],
                "protocol": {"repeats": 1, "timeout": 5},
            }
        ),
        encoding="utf-8",
    )
    gateway = HarnessGateway(
        task,
        root / "evidence",
        root / "trial",
        TrialProtocol(feedback, queries, 60, candidate_entry=candidate_entry),
    )
    return gateway, target


def _failed_report() -> dict:
    return {
        "kind": "torchport.verify",
        "passed": False,
        "status": "failed",
        "failure_code": "PARITY_FAILED",
        "summary": "InstanceNorm in encoder.layer3 caused reduction mismatch",
        "next": "edit model.py:31",
        "aligned": False,
        "strict_aligned": False,
        "mismatch": {
            "fp32": {
                "max_absolute_error": 0.25,
                "mismatch_ratio": 0.5,
                "mismatched_elements": 3,
                "mismatched_tensors": 1,
            }
        },
        "results": [
            {
                "kind": "torchport.evaluation",
                "passed": False,
                "failure_code": "PARITY_FAILED",
                "runnable": "pass",
                "structural": "pass",
                "evidence": {"weights": {"complete": True}},
                "profiles": {
                    "fp32": {
                        "first": {"name": "encoder.layer3.InstanceNorm"},
                        "metrics": {
                            "max_absolute_error": 0.25,
                            "mismatch_ratio": 0.5,
                            "mismatched_elements": 3,
                            "mismatched_tensors": 1,
                        },
                    }
                },
            }
        ],
    }


class GatewayTests(unittest.TestCase):
    def test_two_tasks_share_candidate_and_both_gate_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, target = _fixture(root, FeedbackLevel.LOCALIZED)
            second = root / "training.json"
            task = json.loads(gateway.task_path.read_text("utf-8"))
            task["task_id"] = "training"
            second.write_text(json.dumps(task), encoding="utf-8")
            paired = HarnessGateway(
                {"inference": gateway.task_path, "training": second},
                root / "paired-evidence", root / "paired-trial",
                TrialProtocol(FeedbackLevel.LOCALIZED, 2, 60),
            )
            passed = {**_failed_report(), "passed": True}
            with patch("experiments.harness.gateway.verify", side_effect=[passed, _failed_report()]):
                response = paired.call("verify")

        self.assertFalse(response["data"]["passed"])
        self.assertTrue(response["data"]["tasks"]["inference"]["passed"])
        self.assertFalse(response["data"]["tasks"]["training"]["passed"])
        self.assertEqual(response["budget"]["oracle_queries"], 1)
        self.assertEqual(Path(target).name, "target")

    def test_none_schema_has_task_selector_but_no_localization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, _ = _fixture(root, FeedbackLevel.DIAGNOSTIC)
            second = root / "second.json"
            second.write_text(gateway.task_path.read_text("utf-8"), encoding="utf-8")
            paired = HarnessGateway(
                {"inference": gateway.task_path, "training": second},
                root / "e2", root / "t2", TrialProtocol(FeedbackLevel.DIAGNOSTIC, 2, 60),
            )
            schemas = {item["name"]: item for item in paired.tool_schemas()}

        self.assertNotIn("trace", schemas)
        self.assertNotIn("localize", schemas)
        self.assertEqual(schemas["run"]["parameters"]["properties"]["task"]["enum"],
                         ["inference", "training"])

    def test_restart_restores_query_budget_and_call_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, _ = _fixture(root, FeedbackLevel.DIAGNOSTIC, queries=1)
            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                first = gateway.call("verify")
            resumed = HarnessGateway(
                gateway.task_path, root / "evidence", root / "trial",
                TrialProtocol(FeedbackLevel.DIAGNOSTIC, 1, 60),
            )
            second = resumed.call("verify")

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(resumed.calls, 2)
        self.assertEqual(second["budget"]["oracle_queries"], 1)

    def test_exposes_only_torchport_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), FeedbackLevel.LOCALIZED)
            names = {item["name"] for item in gateway.tool_schemas()}

        self.assertEqual(
            names,
            {"inspect", "run", "evaluate", "verify", "trace", "localize"},
        )
        self.assertTrue(
            {"list_files", "read_file", "apply_patch", "probe", "web_search"}.isdisjoint(names)
        )

    def test_trial_metadata_is_frozen_without_overwriting_core_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, _ = _fixture(root)
            task = gateway.task_path
            enriched = HarnessGateway(
                task,
                root / "evidence-2",
                root / "trial-2",
                TrialProtocol(FeedbackLevel.VERDICT, 2, 60),
                metadata={"experiment_id": "smoke", "condition": "verdict", "replicate": 0},
            )
            trial = json.loads((root / "trial-2" / "trial.json").read_text("utf-8"))
            with self.assertRaisesRegex(ValueError, "保留字段"):
                HarnessGateway(
                    task,
                    root / "evidence-3",
                    root / "trial-3",
                    TrialProtocol(FeedbackLevel.VERDICT, 2, 60),
                    metadata={"trial_id": "overwrite"},
                )

        self.assertEqual(enriched.trial_id, "t0")
        self.assertEqual(trial["experiment_id"], "smoke")

    def test_verdict_hides_reason_but_counts_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary))
            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                response = gateway.call("verify")

        self.assertEqual(response["data"], {"passed": False})
        self.assertEqual(response["budget"]["oracle_queries"], 1)
        self.assertNotIn("reason", json.dumps(response["data"]))

    def test_diagnostic_hides_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), FeedbackLevel.DIAGNOSTIC)
            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                response = gateway.call("verify")

        self.assertEqual(response["data"]["failure_code"], "NUMERICAL_MISMATCH")
        self.assertEqual(response["data"]["max_abs"], 0.25)
        serialized = json.dumps(response["data"])
        for secret in ("InstanceNorm", "encoder.layer3", "model.py", "first", "next"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(tuple(response["data"]), DIAGNOSTIC_FIELDS)

    def test_localized_keeps_divergence_but_not_controller_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), FeedbackLevel.LOCALIZED)
            report = {**_failed_report(), "oracle": "/private/oracle",
                      "artifacts": {"report": "/private/report"}}
            with patch("experiments.harness.gateway.verify", return_value=report):
                response = gateway.call("verify")

        serialized = json.dumps(response["data"])
        self.assertNotIn("/private", serialized)
        self.assertNotIn("edit model.py", serialized)
        self.assertIn("encoder.layer3", serialized)

    def test_trace_is_only_available_for_localized_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), FeedbackLevel.DIAGNOSTIC)
            names = {tool["name"] for tool in gateway.tool_schemas()}
            response = gateway.call("trace")

        self.assertNotIn("trace", names)
        self.assertFalse(response["ok"])
        self.assertEqual(response["budget"]["oracle_queries"], 0)

    def test_oracle_query_budget_cannot_be_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), queries=1)
            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                first = gateway.call("verify")
                second = gateway.call("verify")

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(second["budget"]["budget_stop_reason"], "oracle_query_budget_exhausted")

    def test_certification_freezes_repository_before_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, target = _fixture(root, candidate_entry="run.py")
            names = {tool["name"] for tool in gateway.tool_schemas()}
            denied = gateway.call("certify")
            official = {**_failed_report(), "artifacts": {"report": "x"}}
            verified_repos = []

            def fake_verify(task, *_args, **_kwargs):
                verified_repos.append(Path(task["target"]["repo"]))
                return official

            with patch("experiments.harness.gateway.verify", side_effect=fake_verify):
                result = gateway.certify("agent_completed")
            snapshot = Path(result["candidate_delivery"]["snapshot"])
            (target / "main.py").write_text("VALUE = 9\n", encoding="utf-8")

            self.assertNotIn("certify", names)
            self.assertFalse(denied["ok"])
            self.assertEqual(verified_repos, [snapshot])
            self.assertNotEqual(snapshot, target)
            self.assertEqual((snapshot / "main.py").read_text("utf-8"), "VALUE = 0\n")
            self.assertTrue(Path(result["candidate_delivery"]["manifest"]).is_file())

    def test_scope_violation_invalidates_trial_even_if_verify_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), candidate_entry="run.py")
            report = {**_failed_report(), "passed": True}
            with patch("experiments.harness.gateway.verify", return_value=report):
                result = gateway.certify("scope_violation")

        self.assertTrue(result["passed"])
        self.assertFalse(result["valid"])
        self.assertFalse(result["success"])

    def test_pilot_only_never_claims_valid_formal_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _ = _fixture(root, candidate_entry="run.py")
            pilot = HarnessGateway(
                base.task_path, root / "pilot-evidence", root / "pilot-trial",
                TrialProtocol(FeedbackLevel.VERDICT, 2, 60),
                metadata={"claim_status": "pilot_only"},
            )
            report = {**_failed_report(), "passed": True}
            with patch("experiments.harness.gateway.verify", return_value=report):
                result = pilot.certify("agent_completed")

        self.assertTrue(result["passed"])
        self.assertEqual(result["claim_status"], "pilot_only")
        self.assertFalse(result["valid"])
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
