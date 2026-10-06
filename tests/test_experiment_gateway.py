from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.harness.gateway import HarnessGateway
from experiments.harness.protocol import DIAGNOSTIC_FIELDS, FeedbackLevel, TrialProtocol
from experiments.harness.runner import InfrastructureError, run_trial


def _fixture(
    root: Path,
    feedback: FeedbackLevel = FeedbackLevel.VERDICT,
    queries: int = 2,
    device: str = "cpu",
):
    source, target = root / "source", root / "target"
    source.mkdir()
    target.mkdir()
    (source / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (target / "main.py").write_text("VALUE = 0\n", encoding="utf-8")
    task = root / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "gateway",
                "source": {"repo": "source", "entry": "main.py", "model": "Net"},
                "target": {
                    "repo": "target",
                    "entry": "main.py",
                    "model": "Net",
                    "device": device,
                },
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
        TrialProtocol(feedback, queries, 60),
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
        "repeats": {"runs": 1, "passed_runs": 0},
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
        self.assertEqual(trial["condition"], "verdict")

    def test_verdict_hides_reason_but_counts_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary))
            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                response = gateway.call("verify")

        self.assertTrue(response["ok"])
        self.assertEqual(response["data"], {"passed": False})
        self.assertEqual(response["budget"]["oracle_queries"], 1)
        self.assertEqual(response["budget"]["unique_candidate_queries"], 1)
        self.assertNotIn("reason", json.dumps(response["data"]))

    def test_diagnostic_exposes_metrics_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), FeedbackLevel.DIAGNOSTIC)
            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                response = gateway.call("verify")

        self.assertEqual(response["data"]["failure_code"], "NUMERICAL_MISMATCH")
        self.assertEqual(response["data"]["stage"], "numerical")
        self.assertEqual(response["data"]["category"], "numeric")
        self.assertEqual(response["data"]["max_abs"], 0.25)
        self.assertEqual(response["data"]["mismatched_elements"], 3)
        serialized = json.dumps(response["data"])
        for secret in ("InstanceNorm", "encoder.layer3", "model.py", "first", "next"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(tuple(response["data"]), DIAGNOSTIC_FIELDS)

    def test_trace_is_unavailable_without_localized_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), FeedbackLevel.DIAGNOSTIC)
            names = {tool["name"] for tool in gateway.tool_schemas()}
            response = gateway.call("trace")

        self.assertNotIn("trace", names)
        self.assertNotIn("localize", names)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["type"], "PermissionError")
        self.assertEqual(response["budget"]["oracle_queries"], 0)

    def test_trace_returns_compact_summary_but_keeps_full_report_for_localize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), FeedbackLevel.LOCALIZED)
            report = {
                "kind": "torchport.trace",
                "task_id": "gateway",
                "passed": False,
                "status": "diverged",
                "output": {
                    "passed": False,
                    "profile": "fp32-inference",
                    "first_gating_divergence": {"name": "output.0", "position": 1},
                    "metrics": {"mismatched_elements": 3},
                    "tensors": [{"name": "output.0", "values": list(range(1000))}],
                },
                "strict": {
                    "passed": False,
                    "profile": "operator-strict",
                    "first_divergence": {"name": "model.block", "position": 2},
                    "metrics": {"mismatched_tensors": 1},
                    "tensors": [{"name": "model.block", "values": list(range(1000))}],
                },
                "physical_cost": {"executions": 2},
                "artifacts": {"report": "trace.json"},
            }
            with patch("experiments.harness.gateway.trace", return_value=report):
                response = gateway.call("trace")

        self.assertTrue(response["ok"])
        self.assertNotIn("tensors", response["data"]["output"])
        self.assertNotIn("tensors", response["data"]["strict"])
        self.assertIn("tensors", gateway._last_trace["strict"])
        self.assertLess(len(json.dumps(response["data"])), 2000)

    def test_oracle_query_budget_cannot_be_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary), queries=1)
            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                first = gateway.call("verify")
                second = gateway.call("verify")

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(second["budget"]["oracle_queries"], 1)
        self.assertEqual(second["budget"]["budget_stop_reason"], "oracle_query_budget_exhausted")

    def test_logical_queries_and_physical_executions_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary))
            oracle = {
                "ok": True,
                "dir": str(Path(temporary) / "oracle"),
                "physical_cost": {
                    "executions": 1,
                    "execution_wall_seconds": 2.0,
                    "cuda_execution_seconds": 0.0,
                },
            }
            manifest = {
                "kind": "torchport.run",
                "run_id": "run-1",
                "ok": True,
                "mode": "normal",
                "execution": {"ok": True, "exit_code": 0, "timeout": False, "seconds": 3.0},
                "capture": {},
            }
            with (
                patch("experiments.harness.gateway.build_oracle", return_value=oracle),
                patch("experiments.harness.gateway.run_side", return_value=manifest),
                patch("experiments.harness.gateway.evaluate", return_value=_failed_report()),
            ):
                run_response = gateway.call("run")
                evaluate_response = gateway.call("evaluate", {"run_id": "run-1"})

        self.assertEqual(run_response["budget"]["logical_query_cost"]["total"], 0)
        self.assertEqual(run_response["budget"]["physical_evaluation_cost"]["executions"], 2)
        self.assertEqual(
            run_response["budget"]["physical_evaluation_cost"]["execution_wall_seconds"],
            5.0,
        )
        self.assertEqual(evaluate_response["budget"]["logical_query_cost"]["total"], 1)
        self.assertEqual(
            evaluate_response["budget"]["physical_evaluation_cost"]["execution_wall_seconds"],
            5.0,
        )

    def test_probe_uses_target_runtime_and_counts_only_physical_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, target = _fixture(root, device="cuda")
            (target / "probe.py").write_text(
                "import json, os, pathlib, sys\n"
                "print(json.dumps({\n"
                "    'args': sys.argv[1:],\n"
                "    'cwd': str(pathlib.Path.cwd()),\n"
                "    'jittor_home': os.environ['JITTOR_HOME'],\n"
                "    'provider_key': os.environ.get('DEEPSEEK_API_KEY'),\n"
                "}))\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "must-not-leak"}):
                response = gateway.call("probe", {"path": "probe.py", "args": ["a b", ";exit 9"]})
            payload = json.loads(response["data"]["stdout"])
            manifests = list((root / "evidence" / "probes" / "t0").glob("*/probe.json"))
            manifest = json.loads(manifests[0].read_text("utf-8"))

        self.assertTrue(response["ok"])
        self.assertTrue(response["data"]["ok"])
        self.assertEqual(payload["args"], ["a b", ";exit 9"])
        self.assertIsNone(payload["provider_key"])
        self.assertEqual(Path(payload["cwd"]), target)
        self.assertTrue(
            Path(payload["jittor_home"]).is_relative_to(root / "evidence" / "cache" / "t0")
        )
        self.assertEqual(response["budget"]["logical_query_cost"]["total"], 0)
        self.assertEqual(response["budget"]["physical_evaluation_cost"]["executions"], 1)
        self.assertGreater(
            response["budget"]["physical_evaluation_cost"]["cuda_execution_seconds"], 0
        )
        self.assertEqual(manifest["device"], "cuda")
        self.assertEqual(manifest["execution"]["command"][-2:], ["a b", ";exit 9"])

    def test_probe_rejects_paths_outside_target_and_non_string_args(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, target = _fixture(root)
            (target / "notes.txt").write_text("not executable\n", encoding="utf-8")

            escaped = gateway.call("probe", {"path": "../source/main.py"})
            wrong_suffix = gateway.call("probe", {"path": "notes.txt"})
            invalid_args = gateway.call("probe", {"path": "main.py", "args": [1]})

        self.assertEqual(escaped["error"]["type"], "PermissionError")
        self.assertEqual(wrong_suffix["error"]["type"], "ValueError")
        self.assertEqual(invalid_args["error"]["type"], "TypeError")
        self.assertEqual(invalid_args["budget"]["physical_evaluation_cost"]["executions"], 0)

    def test_probe_timeout_is_counted_and_output_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, target = _fixture(root)
            (target / "probe.py").write_text("pass\n", encoding="utf-8")

            def timed_out(_side, _args, *, run_dir, cache, timeout, module):
                del cache, timeout, module
                (run_dir / "stdout.log").write_text("x" * 20_001, encoding="utf-8")
                (run_dir / "stderr.log").write_text("timed out", encoding="utf-8")
                return {
                    "command": ["python", "probe.py"],
                    "cwd": str(target),
                    "cache": "cache",
                    "started": 1.0,
                    "finished": 2.25,
                    "seconds": 1.25,
                    "timeout": True,
                    "exit_code": None,
                    "ok": False,
                    "stderr_tail": "timed out",
                }

            with patch("experiments.harness.gateway.execute", side_effect=timed_out):
                response = gateway.call("probe", {"path": "probe.py"})

        self.assertTrue(response["ok"])
        self.assertFalse(response["data"]["ok"])
        self.assertTrue(response["data"]["timeout"])
        self.assertTrue(response["data"]["stdout_truncated"])
        self.assertEqual(len(response["data"]["stdout"]), 20_000)
        self.assertEqual(response["budget"]["logical_query_cost"]["total"], 0)
        self.assertEqual(response["budget"]["physical_evaluation_cost"]["executions"], 1)
        self.assertEqual(
            response["budget"]["physical_evaluation_cost"]["execution_wall_seconds"], 1.25
        )

    def test_patch_is_confined_and_changes_candidate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, target = _fixture(root)
            before = gateway.call("inspect")["data"]["candidate_identity"]
            source_read = gateway.call("read_file", {"side": "source", "path": "main.py"})
            updated = gateway.call(
                "apply_patch",
                {"path": "main.py", "old": "VALUE = 0", "new": "VALUE = 1"},
            )
            source_write = gateway.call(
                "apply_patch",
                {"side": "source", "path": "main.py", "old": "VALUE = 1", "new": "VALUE = 9"},
            )
            escaped = gateway.call(
                "apply_patch",
                {"path": "../source/main.py", "old": "VALUE = 1", "new": "VALUE = 9"},
            )
            target_text = (target / "main.py").read_text("utf-8")
            source_text = (root / "source" / "main.py").read_text("utf-8")

        self.assertTrue(source_read["ok"])
        self.assertEqual(source_read["data"]["side"], "source")
        self.assertTrue(updated["ok"])
        self.assertNotEqual(before, updated["data"]["candidate_identity"])
        self.assertFalse(source_write["ok"])
        self.assertFalse(escaped["ok"])
        self.assertEqual(target_text, "VALUE = 1\n")
        self.assertEqual(source_text, "VALUE = 1\n")

    def test_read_file_is_line_ranged_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, target = _fixture(root)
            (target / "large.py").write_text(
                "".join(f"line {number}\n" for number in range(1, 251)), encoding="utf-8"
            )

            first = gateway.call("read_file", {"path": "large.py"})
            tail = gateway.call(
                "read_file", {"path": "large.py", "start_line": 201, "end_line": 250}
            )
            too_large = gateway.call(
                "read_file", {"path": "large.py", "start_line": 1, "end_line": 201}
            )

        self.assertEqual(first["data"]["end_line"], 200)
        self.assertEqual(first["data"]["total_lines"], 250)
        self.assertTrue(first["data"]["truncated"])
        self.assertTrue(tail["data"]["text"].startswith("line 201\n"))
        self.assertFalse(tail["data"]["truncated"])
        self.assertFalse(too_large["ok"])
        self.assertEqual(too_large["error"]["type"], "ValueError")

    def test_official_certification_is_not_an_agent_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, _ = _fixture(root)
            names = {tool["name"] for tool in gateway.tool_schemas()}
            denied = gateway.call("certify")
            official = {**_failed_report(), "artifacts": {"report": "x"}}
            with patch("experiments.harness.gateway.verify", return_value=official):
                result = gateway.certify("agent_completed")

            saved = json.loads((root / "trial" / "official-verify.json").read_text("utf-8"))

        self.assertNotIn("certify", names)
        self.assertFalse(denied["ok"])
        self.assertFalse(result["passed"])
        self.assertEqual(saved["failure_code"], "PARITY_FAILED")

    def test_runner_certifies_after_agent_returns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary))

            def agent(tools, call):
                self.assertIn("apply_patch", {tool["name"] for tool in tools})
                self.assertTrue(call("list_files")["ok"])
                return "agent_completed"

            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                result = run_trial(gateway, agent)

        self.assertEqual(result["stop_reason"], "agent_completed")
        self.assertEqual(result["tool_calls"], 1)

    def test_runner_marks_unexpected_adapter_error_as_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gateway, _ = _fixture(root)

            def broken_agent(_tools, _call):
                raise ValueError("adapter bug")

            with patch("experiments.harness.gateway.verify", return_value=_failed_report()):
                result = run_trial(gateway, broken_agent)

            events = [
                json.loads(line)
                for line in (root / "trial" / "events.jsonl").read_text("utf-8").splitlines()
            ]

        self.assertEqual(result["stop_reason"], "infrastructure_error")
        self.assertFalse(result["valid"])
        error = next(event for event in events if event["event"] == "agent_error")
        self.assertEqual(error["phase"], "controller")

    def test_runner_raises_infrastructure_error_when_official_verify_crashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gateway, _ = _fixture(Path(temporary))

            with patch(
                "experiments.harness.gateway.verify", side_effect=RuntimeError("worker died")
            ):
                with self.assertRaisesRegex(InfrastructureError, "official verify failed"):
                    run_trial(gateway, lambda _tools, _call: "agent_completed")


if __name__ == "__main__":
    unittest.main()
