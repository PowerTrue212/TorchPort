from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.harness.performance import measure_performance


def _run(name: str, started: float, first: float, median: float) -> dict:
    return {
        "run_id": name,
        "execution": {"started": started, "seconds": 3.0},
        "capture": {
            "performance": {
                "first_output_at": first,
                "batch_size": 2,
                "steady_latency_ms": {"median": median, "p90": median + 1},
                "throughput_per_second": 2000 / median,
                "latency_ms": [median - 1, median, median + 1],
            }
        },
    }


class P2PerformanceTests(unittest.TestCase):
    def test_uses_cold_then_warm_run_and_requires_correctness(self) -> None:
        task = {
            "task_id": "tiny",
            "digest": "task-digest",
            "target_identity": "candidate",
            "target": {"device": "cuda"},
            "protocol": {},
        }
        oracle = {"ok": True, "dir": "oracle", "physical_cost": None}
        runs = iter([_run("cold", 10, 12.5, 4.0), _run("warm", 20, 20.5, 2.0)])
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("experiments.harness.performance.build_oracle", return_value=oracle),
            patch("experiments.harness.performance.run_side", side_effect=lambda *_a, **_k: next(runs)),
            patch("experiments.harness.performance.evaluate", return_value={"passed": True}),
        ):
            report = measure_performance(
                task,
                {"passed": True, "task_digest": "task-digest"},
                Path(temporary),
                trial="p0",
                warmup=2,
                iterations=3,
            )

        self.assertEqual(report["cold_start_seconds"], 2.5)
        self.assertEqual(report["warm_start_seconds"], 0.5)
        self.assertEqual(report["steady_latency_ms"]["median"], 2.0)
        self.assertEqual(report["runs"], ["cold", "warm"])
        self.assertEqual(report["schema_version"], 2)
        self.assertNotIn("intervention", report)

    def test_rejects_unverified_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "official verify"):
            measure_performance(
                {"digest": "a"},
                {"passed": False, "task_digest": "a"},
                ".",
                trial="p0",
                warmup=1,
                iterations=1,
            )


if __name__ == "__main__":
    unittest.main()
