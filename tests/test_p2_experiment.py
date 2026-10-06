from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.harness.aggregate import aggregate
from experiments.harness.experiment import execute_run_plan, preflight, prepare_run_plan
from torchport.execution import write_json
from torchport.task import file_hash


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
        "provider": "deepseek",
        "model": "deepseek-flash",
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
            "network_isolation": formal,
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

            checked = preflight(manifest)
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
            write_json(attestation, {"network_isolation": True})

            with self.assertRaisesRegex(ValueError, "worker-attestation"):
                prepare_run_plan(manifest)
            with self.assertRaisesRegex(ValueError, "隔离字段"):
                prepare_run_plan(manifest, worker_attestation=attestation)

            write_json(
                attestation,
                {
                    "schema_version": 1,
                    "kind": "torchport.p2-worker-attestation",
                    "runtime": "docker",
                    "image_digest": "sha256:test",
                    "network_isolation": True,
                    "host_filesystem_isolation": True,
                    "candidate_workspace_is_only_writable_mount": True,
                    "source_and_frozen_inputs_read_only": True,
                    "hidden_artifacts_not_mounted_for_agent": True,
                },
            )
            plan = prepare_run_plan(manifest, worker_attestation=attestation)
            frozen = json.loads(plan.read_text("utf-8"))
            self.assertEqual(frozen["worker_attestation_sha256"], file_hash(attestation))

            runtime = root / frozen["trial_runtime_snapshot"]
            runtime.write_text("{}\n", "utf-8")
            with self.assertRaisesRegex(ValueError, "runtime snapshot"):
                prepare_run_plan(manifest, allow_trusted_pilot=True)

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
                calls.append((attempt_id, candidate, text))
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
        self.assertTrue(trial["selected"].endswith("infra-r1"))
        self.assertEqual(resumed, state)
        self.assertTrue(report["complete"])
        self.assertEqual(report["successes"], 1)
        self.assertEqual(len(report["trials"][0]["attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
