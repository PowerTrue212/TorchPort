from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.harness.orchestration import (
    execute_run_plan,
    prepare_run_plan,
    validate_experiment,
)
from experiments.harness.reporting import aggregate
from torchport.contracts import file_hash
from torchport.runtime import write_json


def _fixture(root: Path, *, formal: bool = False) -> Path:
    source = root / "source"
    seed = root / "seed"
    source.mkdir()
    seed.mkdir()
    (source / "main.py").write_text("VALUE = 1\n", "utf-8")
    (seed / "main.py").write_text("VALUE = 0\n", "utf-8")
    private = root / "private.bin"
    private.write_bytes(b"hidden")
    task = {
        "schema_version": 1,
        "task_id": "fixture",
        "source": {
            "repo": str(source),
            "entry": "main.py",
            "model": "Net",
            "python": sys.executable,
            "device": "cpu",
        },
        "target": {
            "repo": str(seed),
            "entry": "main.py",
            "model": "Net",
            "python": sys.executable,
            "device": "cpu",
        },
        "inputs": [{"shape": [1]}],
        "protocol": {"repeats": 1, "timeout": 5},
    }
    write_json(root / "public.json", task)
    write_json(root / "hidden.json", task)
    (root / "prompt.md").write_text("migrate\n", "utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_id": "fixture-experiment",
        "provider": "external-code-agent",
        "model": "fixture-agent",
        "temperature": 0.0,
        "prompt": "prompt.md",
        "runtime_packages": ["pytest"],
        "budget": {
            "oracle_queries": 2,
            "wall_seconds": 60,
            "total_tokens": 100,
            "model_turns": 2,
        },
        "replicates": 1,
        "tasks": ["fixture"],
        "conditions": ["verdict"],
        "isolation": {
            "formal_claims_allowed": formal,
            "public_research_enabled": formal,
            "host_filesystem_isolation": formal,
        },
        "frozen_artifacts": {"private.bin": file_hash(private)},
        "trials": [
            {
                "trial_id": "fixture-verdict-r0",
                "task": "fixture",
                "condition": "verdict",
                "replicate": 0,
                "path": "runs/fixture/verdict-r0",
                "public_task": "public.json",
                "hidden_task": "hidden.json",
            }
        ],
    }
    path = root / "experiment.json"
    write_json(path, manifest)
    return path


class ExperimentRunnerTests(unittest.TestCase):
    def test_preflight_freezes_runtime_and_refuses_unacknowledged_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _fixture(root)

            checked = validate_experiment(manifest)
            with self.assertRaisesRegex(ValueError, "--allow-trusted-pilot"):
                prepare_run_plan(manifest)
            plan_path = prepare_run_plan(manifest, allow_trusted_pilot=True)
            plan = json.loads(plan_path.read_text("utf-8"))
            snapshot = root / plan["trial_runtime_snapshot"]
            runtime = json.loads(snapshot.read_text("utf-8"))
            self.assertEqual(len(checked["task_records"]), 1)
            self.assertTrue(snapshot.name.endswith("runtime-snapshot.json"))
            self.assertEqual(plan["trial_runtime_snapshot_sha256"], file_hash(snapshot))
            self.assertEqual(plan["prompt_sha256"], file_hash(root / "prompt.md"))
            self.assertTrue(
                all("pytest" in item["details"]["packages"] for item in runtime["runtimes"].values())
            )
            with self.assertRaisesRegex(ValueError, "--allow-trusted-pilot"):
                prepare_run_plan(manifest)

            (root / "prompt.md").write_text("changed\n", "utf-8")
            with self.assertRaisesRegex(ValueError, "prompt"):
                prepare_run_plan(manifest, allow_trusted_pilot=True)

    def test_formal_plan_requires_complete_worker_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _fixture(root, formal=True)
            attestation = root / "worker.json"
            write_json(attestation, {"public_web_search_available": True})

            with self.assertRaisesRegex(ValueError, "worker-attestation"):
                prepare_run_plan(manifest)
            with self.assertRaisesRegex(ValueError, "隔离字段"):
                prepare_run_plan(manifest, worker_attestation=attestation)

            complete_attestation = {
                "schema_version": 1,
                "kind": "torchport.p2-worker-attestation",
                "runtime": "docker",
                "image_digest": "sha256:test",
                "agent_tool_inventory_sha256": "sha256:fixture-tools",
                "host_filesystem_isolation": True,
                "candidate_workspace_is_only_writable_mount": True,
                "source_and_frozen_inputs_read_only": True,
                "hidden_artifacts_not_mounted_for_agent": True,
                "controller_code_not_mounted_for_agent": True,
                "calibration_references_not_mounted_for_agent": True,
                "other_candidate_workspaces_not_mounted_for_agent": True,
                "agent_context_is_fresh": True,
                "agent_memory_and_history_not_mounted": True,
                "other_trial_sessions_not_accessible": True,
                "public_network_access_available": True,
                "public_web_search_available": True,
                "public_code_search_available": True,
                "public_research_event_logging_available": True,
            }
            for field in (
                "controller_code_not_mounted_for_agent",
                "calibration_references_not_mounted_for_agent",
                "other_candidate_workspaces_not_mounted_for_agent",
                "agent_context_is_fresh",
                "agent_memory_and_history_not_mounted",
                "other_trial_sessions_not_accessible",
                "public_network_access_available",
                "public_web_search_available",
                "public_code_search_available",
                "public_research_event_logging_available",
            ):
                with self.subTest(field=field):
                    write_json(attestation, {**complete_attestation, field: False})
                    with self.assertRaisesRegex(ValueError, field):
                        prepare_run_plan(manifest, worker_attestation=attestation)
            write_json(attestation, complete_attestation)
            plan = prepare_run_plan(manifest, worker_attestation=attestation)
            frozen = json.loads(plan.read_text("utf-8"))
            self.assertEqual(frozen["worker_attestation_sha256"], file_hash(attestation))

            runtime = root / frozen["trial_runtime_snapshot"]
            runtime.write_text("{}\n", "utf-8")
            with self.assertRaisesRegex(ValueError, "runtime snapshot"):
                prepare_run_plan(manifest, allow_trusted_pilot=True)

    def test_benchmark_v1_requires_read_and_url_isolation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _fixture(root, formal=True)
            data = json.loads(manifest.read_text("utf-8"))
            data["benchmark_protocol"] = "benchmark-v1"
            write_json(manifest, data)
            attestation = root / "worker.json"
            write_json(attestation, {
                "runtime": "container", "image_digest": "sha256:test",
                "agent_tool_inventory_sha256": "sha256:tools",
                **{name: True for name in (
                    "host_filesystem_isolation", "candidate_workspace_is_only_writable_mount",
                    "source_and_frozen_inputs_read_only", "hidden_artifacts_not_mounted_for_agent",
                    "controller_code_not_mounted_for_agent", "calibration_references_not_mounted_for_agent",
                    "other_candidate_workspaces_not_mounted_for_agent", "agent_context_is_fresh",
                    "agent_memory_and_history_not_mounted", "other_trial_sessions_not_accessible",
                    "public_network_access_available", "public_web_search_available",
                    "public_code_search_available", "public_research_event_logging_available",
                )},
            })

            with self.assertRaisesRegex(ValueError, "torchport_import_blocked"):
                prepare_run_plan(manifest, worker_attestation=attestation)

    def test_infrastructure_retry_uses_fresh_seed_and_resume_skips_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _fixture(root)
            plan = prepare_run_plan(manifest, allow_trusted_pilot=True)
            calls = []

            def run_attempt(item, attempt_id, public, hidden, trial_dir, runtime_sha):
                del item, hidden, runtime_sha
                task = json.loads(public.read_text("utf-8"))
                candidate = Path(task["target"]["repo"])
                text = (candidate / "main.py").read_text("utf-8")
                calls.append((attempt_id, candidate, text, (candidate / ".git").exists()))
                (candidate / "main.py").write_text("MUTATED = 1\n", "utf-8")
                valid = len(calls) > 1
                result = {
                    "valid": valid,
                    "success": valid,
                    "passed": valid,
                    "stop_reason": "agent_completed" if valid else "infrastructure_error",
                    "total_wall_seconds": 1.0,
                    "physical_evaluation_cost": {
                        "total": {
                            "executions": 0,
                            "execution_wall_seconds": 0.0,
                            "cuda_execution_seconds": 0.0,
                        }
                    },
                }
                write_json(trial_dir / "result.json", result)
                return result

            state = execute_run_plan(plan, run_attempt, max_infrastructure_retries=1)
            first_call_count = len(calls)
            resumed = execute_run_plan(plan, run_attempt, max_infrastructure_retries=1)
            report = aggregate(plan)

        trial = state["trials"]["fixture-verdict-r0"]
        self.assertEqual(first_call_count, 2)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[0][2], "VALUE = 0\n")
        self.assertEqual(calls[1][2], "VALUE = 0\n")
        self.assertTrue(calls[0][3])
        self.assertTrue(calls[1][3])
        self.assertTrue(trial["selected"].endswith("infra-r1"))
        self.assertEqual(resumed, state)
        self.assertTrue(report["complete"])
        self.assertEqual(report["successes"], 1)
        self.assertEqual(len(report["trials"][0]["attempts"]), 2)

    def test_scope_violation_is_invalid_and_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = prepare_run_plan(_fixture(root), allow_trusted_pilot=True)
            calls = []

            def run_attempt(item, attempt_id, public, hidden, trial_dir, runtime_sha):
                del item, public, hidden, runtime_sha
                calls.append(attempt_id)
                result = {
                    "valid": True,
                    "success": True,
                    "passed": True,
                    "stop_reason": "scope_violation",
                }
                write_json(trial_dir / "result.json", result)
                return result

            state = execute_run_plan(plan, run_attempt, max_infrastructure_retries=2)
            resumed = execute_run_plan(plan, run_attempt, max_infrastructure_retries=2)
            report = aggregate(plan)

        trial = state["trials"]["fixture-verdict-r0"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(resumed, state)
        self.assertFalse(trial["attempts"][0]["valid"])
        self.assertFalse(trial["attempts"][0]["success"])
        self.assertIsNone(trial["selected"])
        self.assertEqual(report["stop_reasons"]["scope_violation"], 1)
        self.assertEqual(report["valid_trials"], 0)


if __name__ == "__main__":
    unittest.main()
