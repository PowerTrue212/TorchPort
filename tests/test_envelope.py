from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from torchport.parity import compare_envelope, compare_metric_envelope
from torchport.verify import judge


def _save(path: Path, **arrays: np.ndarray) -> Path:
    np.savez(path, **arrays)
    return path


class EnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        rng = np.random.default_rng(0)
        self.base = rng.standard_normal(1000).astype(np.float64)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _archive(self, name: str, noise: float, seed: int, *, layer_noise: float | None = None) -> Path:
        rng = np.random.default_rng(seed)
        layer = self.base + (layer_noise if layer_noise is not None else noise) * rng.standard_normal(1000)
        output = self.base + noise * rng.standard_normal(1000)
        return _save(self.root / f"{name}.npz", **{"block.0": layer, "output.0": output})

    def test_candidate_within_source_variants_passes(self) -> None:
        reference = self._archive("ref", 0.0, 1)
        variants = [self._archive(f"v{i}", 1e-6, 10 + i) for i in range(3)]
        candidate = self._archive("cand", 5e-7, 20)
        report = compare_envelope(reference, variants, candidate)
        self.assertTrue(report["passed"], report["first_divergence"])
        self.assertLessEqual(report["metrics"]["max_ratio"], 1.0)

    def test_bug_far_outside_envelope_fails_at_layer(self) -> None:
        reference = self._archive("ref", 0.0, 1)
        variants = [self._archive(f"v{i}", 1e-6, 10 + i) for i in range(3)]
        candidate = self._archive("cand", 1e-6, 20, layer_noise=1e-2)
        report = compare_envelope(reference, variants, candidate)
        self.assertFalse(report["passed"])
        self.assertEqual(report["first_divergence"]["name"], "block.0")

    def test_target_self_noise_widens_limit(self) -> None:
        reference = self._archive("ref", 0.0, 1)
        variants = [self._archive("v0", 1e-7, 10)]
        candidate = self._archive("cand", 1e-5, 20)
        peer = self._archive("peer", 1e-5, 21)
        self.assertFalse(compare_envelope(reference, variants, candidate)["passed"])
        self.assertTrue(compare_envelope(reference, variants, candidate, [peer])["passed"])

    def test_missing_output_fails_but_missing_layer_is_reported_only(self) -> None:
        reference = self._archive("ref", 0.0, 1)
        variants = [self._archive("v0", 1e-6, 10)]
        no_layer = _save(self.root / "no_layer.npz", **{"output.0": self.base})
        report = compare_envelope(reference, variants, no_layer)
        self.assertTrue(report["passed"])
        self.assertEqual(report["missing_in_candidate"], ["block.0"])
        no_output = _save(self.root / "no_output.npz", **{"block.0": self.base})
        self.assertFalse(compare_envelope(reference, variants, no_output)["passed"])

    def test_requires_at_least_one_variant(self) -> None:
        reference = self._archive("ref", 0.0, 1)
        with self.assertRaises(ValueError):
            compare_envelope(reference, [], reference)


class SketchTests(unittest.TestCase):
    def test_sketch_preserves_difference_norm_and_shape(self) -> None:
        from torchport.capture import SKETCH_DIM, sketch_layers

        rng = np.random.default_rng(3)
        a = rng.standard_normal((4, 64, 32, 32))
        b = a + 1e-6 * rng.standard_normal(a.shape)
        sa, sb = sketch_layers({"x": a}), sketch_layers({"x": b})
        self.assertEqual(sa["x"].shape, (SKETCH_DIM,))
        self.assertEqual(tuple(sa["__shape__/x"]), a.shape)
        true = np.linalg.norm((a - b).reshape(-1))
        est = np.linalg.norm(sa["x"] - sb["x"])
        self.assertLess(abs(est / true - 1), 0.05)

    def test_small_tensor_kept_exact(self) -> None:
        from torchport.capture import sketch_layers

        value = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(sketch_layers({"y": value})["y"], value.astype(np.float64))

    def test_envelope_reads_sketched_archives(self) -> None:
        from torchport.capture import sketch_layers

        root = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(4)
        base = rng.standard_normal((2, 64, 64, 32))

        def save(name: str, noise: float, seed: int) -> Path:
            r = np.random.default_rng(seed)
            arrays = {"block.0": base + noise * r.standard_normal(base.shape), "output.0": base[:1, :1]}
            path = root / f"{name}.npz"
            np.savez(path, **sketch_layers(arrays))
            return path

        reference = save("ref", 0.0, 1)
        variants = [save("v0", 1e-6, 2), save("v1", 1e-6, 3)]
        good = compare_envelope(reference, variants, save("cand", 5e-7, 4))
        self.assertTrue(good["passed"], good["first_divergence"])
        names = [item["name"] for item in good["tensors"]]
        self.assertNotIn("__shape__/block.0", names)
        bad = compare_envelope(reference, variants, save("bad", 1e-3, 5))
        self.assertEqual(bad["first_divergence"]["name"], "block.0")


class MetricEnvelopeTests(unittest.TestCase):
    def test_metric_inside_and_outside(self) -> None:
        reference = {"S": 0.8700, "MAE": 0.0230}
        variants = [{"S": 0.8702, "MAE": 0.0231}, {"S": 0.8699, "MAE": 0.0229}]
        good = compare_metric_envelope(reference, variants, [{"S": 0.8701, "MAE": 0.0230}])
        self.assertTrue(good["passed"])
        bad = compare_metric_envelope(reference, variants, [{"S": 0.8600, "MAE": 0.0230}])
        self.assertFalse(bad["passed"])


class JudgeNumericalTests(unittest.TestCase):
    def _infos(self) -> tuple[dict, dict]:
        source = {"mode": "normal", "hashes": {"input": "x"}, "weights": 2, "checkpoint": {"complete": True}}
        target = {
            "mode": "layers",
            "hashes": {"input": "x"},
            "loaded": 2,
            "state": {"target_total": 2, "extra": [], "missing": [], "value_mismatch": []},
        }
        return source, target

    def _structure(self) -> dict:
        item = {
            "name": "output.0",
            "reference_shape": [1],
            "candidate_shape": [1],
            "reference_dtype": "float32",
            "candidate_dtype": "float32",
            "passed": False,
        }
        return {"tensors": [item], "missing_in_candidate": [], "extra_in_candidate": [], "passed": False}

    def test_envelope_decides_numerical_and_layers_mode_counts_as_normal(self) -> None:
        source, target = self._infos()
        passed = judge(source, target, self._structure(), None, {}, numerical={"passed": True, "metrics": {}})
        self.assertTrue(passed["passed"], passed["summary"])
        failed = judge(
            source,
            target,
            self._structure(),
            None,
            {},
            numerical={"passed": False, "first_divergence": {"name": "block.0"}, "tensors": []},
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["failure_code"], "PARITY_FAILED")

    def test_layers_mode_without_envelope_is_not_normal(self) -> None:
        source, target = self._infos()
        report = judge(source, target, self._structure(), None, {})
        self.assertEqual(report["failure_code"], "DIAGNOSTIC_RUN")


if __name__ == "__main__":
    unittest.main()
