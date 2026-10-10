from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from torchport import noise
from torchport.execution import write_json


def _task(target_repo: str = "/repo/a") -> dict:
    return {
        "target": {"repo": target_repo, "python": "jt-python", "env": {"conv_opt": "1"}, "device": "cuda"},
        "protocol": {"timeout": 60},
    }


class NoiseTests(unittest.TestCase):
    def test_summarize_takes_max_op_spread(self) -> None:
        base = np.ones(100)
        report = noise.summarize({
            "stable": [base, base.copy()],
            "reduce": [base, base + 1e-6],
        })
        self.assertEqual(report["dominant_op"], "reduce")
        self.assertEqual(report["per_op"]["stable"], 0.0)
        self.assertAlmostEqual(report["noise_floor"], 1e-6, delta=1e-12)

    def test_identity_ignores_candidate_code(self) -> None:
        probe = dict(noise.DEFAULT_PROBE)
        a = noise.noise_identity(_task("/repo/a")["target"], probe)
        b = noise.noise_identity(_task("/repo/b")["target"], probe)
        self.assertEqual(a, b)
        other_env = {**_task()["target"], "env": {"conv_opt": "0"}}
        self.assertNotEqual(a, noise.noise_identity(other_env, probe))

    def test_calibrate_caches_and_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []

            def ok(side, args, *, run_dir, cache, timeout):
                calls.append(args)
                write_json(run_dir / "noise.json", {"noise_floor": 2e-6, "dominant_op": "sum_spatial"})
                return {"ok": True, "seconds": 1.0}

            with patch.object(noise, "execute", ok):
                first = noise.calibrate(_task("/repo/a"), root)
                second = noise.calibrate(_task("/repo/b"), root)
            self.assertTrue(first["ok"])
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(second["noise_floor"], 2e-6)
            self.assertEqual(len(calls), 1)

            def broken(side, args, *, run_dir, cache, timeout):
                return {"ok": False, "seconds": 1.0, "stderr_tail": "ImportError: jittor"}

            with patch.object(noise, "execute", broken):
                failed = noise.calibrate({**_task(), "protocol": {"timeout": 60, "noise_probe": {"repeats": 3}}}, root)
            self.assertFalse(failed["ok"])
            self.assertEqual(failed["reason"], "NOISE_PROBE_FAILED")


if __name__ == "__main__":
    unittest.main()


def test_reduce_shape_takes_largest_global_reduction_from_source() -> None:
    from torchport.noise import reduce_shape

    config = {
        "a.pool": {"class": "AdaptiveAvgPool2d", "output_size": [1, 1]},
        "b.pool": {"class": "AdaptiveAvgPool2d", "output_size": [1, 1]},
        "c.pool": {"class": "AdaptiveAvgPool2d", "output_size": [7, 7]},  # 非全局池化不计入
        "d.conv": {"class": "Conv2d"},
    }
    shapes = {
        "a.pool": [1, 64, 44, 44],
        "b.pool": [1, 256, 128, 128],
        "c.pool": [1, 64, 512, 512],
        "d.conv": [1, 64, 1024, 1024],
    }
    assert reduce_shape(config, shapes) == [256, 128, 128]
    assert reduce_shape({"d.conv": {"class": "Conv2d"}}, shapes) is None


def test_probe_spec_explicit_task_value_overrides_derived_shape(tmp_path) -> None:
    import json

    from torchport.noise import probe_spec

    config, shapes = tmp_path / "config.json", tmp_path / "shapes.json"
    config.write_text(json.dumps({"p": {"class": "GroupNorm"}}))
    shapes.write_text(json.dumps({"p": [1, 32, 256, 256]}))
    envelope = {"module_config": str(config), "module_shapes": str(shapes)}
    derived = probe_spec({"protocol": {}}, envelope)
    assert derived["reduce_shape"] == [32, 256, 256]
    explicit = probe_spec({"protocol": {"noise_probe": {"reduce_shape": [8, 16, 16]}}}, envelope)
    assert explicit["reduce_shape"] == [8, 16, 16]
