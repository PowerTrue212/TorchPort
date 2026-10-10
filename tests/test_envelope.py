from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from torchport.parity import _l2, compare_envelope, compare_metric_envelope
from torchport.verify import judge


def test_l2_matches_linalg_norm() -> None:
    rng = np.random.default_rng(3)
    for x in (rng.standard_normal(65536).astype(np.float32), rng.standard_normal((4, 5, 6)), np.zeros(7)):
        expected = float(np.linalg.norm(x.astype(np.float64).reshape(-1)))
        assert abs(_l2(x) - expected) <= 1e-12 * max(expected, 1.0)


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


class EnvelopeV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.base = np.random.default_rng(0).standard_normal(1000)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _save(self, name: str, noise: float, seed: int, *, output_noise: float | None = None) -> Path:
        rng = np.random.default_rng(seed)
        layer = self.base + noise * rng.standard_normal(1000)
        out_noise = noise if output_noise is None else output_noise
        output = (self.base + out_noise * rng.standard_normal(1000)).astype(np.float32)
        return _save(self.root / f"{name}.npz", **{"block.0": layer, "output.0": output})

    def _envelope(self) -> tuple[Path, list[Path]]:
        return self._save("ref", 0.0, 1), [self._save(f"v{i}", 1e-6, 10 + i) for i in range(3)]

    def test_order_of_target_runs_does_not_change_verdict(self) -> None:
        from torchport.parity import compare_envelope_v3

        reference, variants = self._envelope()
        good, bad = self._save("good", 5e-7, 20), self._save("bad", 1e-4, 21)
        forward = compare_envelope_v3(reference, variants, [good, bad])
        backward = compare_envelope_v3(reference, variants, [bad, good])
        self.assertEqual(forward["status"], "fail")
        self.assertEqual(backward["status"], "fail")
        self.assertEqual(forward["metrics"]["max_ratio"], backward["metrics"]["max_ratio"])

    def test_target_noise_is_reported_not_added_to_limit(self) -> None:
        from torchport.parity import compare_envelope_v3

        reference, variants = self._envelope()
        # v2 会因两次 target 互相差很远而放宽上限；v3 只报告 unstable
        runs = [self._save("t0", 3e-5, 30), self._save("t1", 3e-5, 31)]
        report = compare_envelope_v3(reference, variants, runs)
        self.assertEqual(report["status"], "fail")
        self.assertGreater(report["metrics"]["unstable_tensors"], 0)

    def test_three_bands(self) -> None:
        from torchport.parity import compare_envelope_v3

        reference, variants = self._envelope()
        d = compare_envelope_v3(reference, variants, [reference])["tensors"][0]["source_noise"]
        self.assertEqual(compare_envelope_v3(reference, variants, [reference])["status"], "pass")

        def scaled(name: str, factor: float) -> Path:
            ref = np.load(reference)
            direction = np.random.default_rng(7).standard_normal(1000)
            direction *= factor * d * np.linalg.norm(self.base) / np.linalg.norm(direction)
            return _save(self.root / f"{name}.npz", **{"block.0": ref["block.0"] + direction, "output.0": ref["output.0"]})

        self.assertEqual(compare_envelope_v3(reference, variants, [scaled("m", 2.5)])["status"], "marginal")
        self.assertEqual(compare_envelope_v3(reference, variants, [scaled("f", 4.0)])["status"], "fail")

    def test_output_elementwise_catches_local_error(self) -> None:
        from torchport.parity import compare_envelope_v3

        reference, variants = self._envelope()
        ref = np.load(reference)
        output = ref["output.0"].copy()
        output[0] += 1e-3  # 单元素大错：L2 相对误差很小，逐元素比值很大
        local = _save(self.root / "local.npz", **{"block.0": ref["block.0"], "output.0": output})
        report = compare_envelope_v3(reference, variants, [local])
        item = next(t for t in report["tensors"] if t["name"] == "output.0")
        self.assertGreater(item["abs_ratio"], item["l2_ratio"])
        self.assertEqual(report["status"], "fail")

    def test_noise_floor_raises_basis_without_using_candidate_noise(self) -> None:
        from torchport.parity import compare_envelope_v3

        reference, variants = self._envelope()
        # 层误差约 1e-5，约为 D 的 7 倍；输出不加噪声，只考察层判定
        candidate = self._save("cand", 1e-5, 50, output_noise=0.0)
        self.assertEqual(compare_envelope_v3(reference, variants, [candidate])["status"], "fail")
        floored = compare_envelope_v3(reference, variants, [candidate], noise_floor=2e-5)
        layer = next(t for t in floored["tensors"] if t["name"] == "block.0")
        self.assertTrue(layer["floor_applied"])
        self.assertEqual(floored["status"], "pass")
        self.assertEqual(floored["noise_floor"], 2e-5)

    def test_batch_variant_far_from_others_is_excluded(self) -> None:
        from torchport.parity import compare_envelope_v3

        reference, variants = self._envelope()
        batch = self._save("batch", 1e-2, 40)  # 批内相关导致的大差异不能撑大包络
        candidate = self._save("cand", 1e-3, 41)
        report = compare_envelope_v3(reference, [*variants, batch], [candidate], batch_variants=(3,))
        self.assertEqual(report["status"], "fail")
        self.assertGreater(report["metrics"]["batch_excluded_tensors"], 0)


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


class ModuleConfigTests(unittest.TestCase):
    def test_reports_eps_swap_and_ignores_class_and_int_tuple_forms(self) -> None:
        from torchport.parity import compare_module_config

        # PlantCamo 实例：backbone.norm1 源为 1e-6、候选写成 1e-5；类名不同不算差异；
        # Conv 的 (1, 1) 与 1 视为相同。
        source = {
            "backbone.norm1": {"class": "LayerNorm", "eps": 1e-6, "normalized_shape": [64]},
            "backbone.block1.0.attn.norm": {"class": "LayerNorm", "eps": 1e-5},
            "cr4.0": {"class": "Conv2d", "stride": [1, 1], "padding": [0, 0]},
            "source_only": {"class": "Identity"},
        }
        target = {
            "backbone.norm1": {"class": "_LayerNorm", "eps": 1e-5, "normalized_shape": [64]},
            "backbone.block1.0.attn.norm": {"class": "_LayerNorm", "eps": 1e-5},
            "cr4.0": {"class": "Conv", "stride": 1, "padding": 0},
        }
        report = compare_module_config(source, target)
        self.assertEqual(
            [(m["module"], m["key"], m["source"], m["target"]) for m in report["mismatches"]],
            [("backbone.norm1", "eps", 1e-6, 1e-5)],
        )
        self.assertEqual(report["source_only_modules"], 1)

    def test_module_config_records_scalar_hyperparameters_only(self) -> None:
        from torchport.capture import module_config

        class Norm:
            eps = 1e-6
            normalized_shape = (8,)
            weight = np.zeros(8)  # 张量属性不进入记录

        class Model:
            def named_modules(self):
                return [("", self), ("norm", Norm())]

        config = module_config(Model())
        self.assertEqual(config, {"norm": {"class": "Norm", "eps": 1e-6, "normalized_shape": [8]}})

    def test_mismatch_in_uncalled_module_is_marked_unused(self) -> None:
        from torchport.parity import called_modules, compare_module_config

        # PlantCamo 实例：fm1..4 定义了但前向中没有调用，其 align_corners 差异是死代码
        with tempfile.TemporaryDirectory() as temporary:
            layers = Path(temporary) / "layers.npz"
            np.savez(layers, **{"norm1": np.zeros(2), "head.0": np.zeros(2), "__shape__/norm1": np.asarray([2])})
            called = called_modules(layers)
        self.assertEqual(called, {"norm1", "head.0", "head"})
        source = {"norm1": {"eps": 1e-6}, "fm1.up": {"align_corners": True}}
        target = {"norm1": {"eps": 1e-5}, "fm1.up": {"align_corners": False}}
        used = {m["module"]: m["used"] for m in compare_module_config(source, target, called)["mismatches"]}
        self.assertEqual(used, {"norm1": True, "fm1.up": False})


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


class IsolatedV3Tests(unittest.TestCase):
    def test_backend_noise_below_floor_passes_and_real_error_is_first(self) -> None:
        from torchport.parity import compare_isolated_v3

        root = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(5)
        conv, norm = rng.standard_normal(4096), rng.standard_normal(4096)

        def save(name: str, conv_noise: float, norm_noise: float) -> Path:
            r = np.random.default_rng(len(name))
            return _save(
                root / f"{name}.npz",
                **{
                    "block.dwconv": conv + conv_noise * np.linalg.norm(conv) / 64 * r.standard_normal(4096),
                    "norm1": norm + norm_noise * np.linalg.norm(norm) / 64 * r.standard_normal(4096),
                },
            )

        reference = save("ref", 0.0, 0.0)
        fp64, alt = save("fp64", 1e-7, 1e-7), save("alt", 1e-7, 1e-7)
        # 卷积只有后端波动（~1e-6，固定容差会先报它）；norm1 是真实的 eps 错误
        candidate = save("candidate", 1e-6, 1e-4)
        report = compare_isolated_v3(reference, candidate, fp64, alt, noise_floor=2e-6)
        self.assertEqual(report["first_divergence"]["name"], "norm1")
        statuses = {item["name"]: item["status"] for item in report["tensors"]}
        self.assertEqual(statuses["block.dwconv"], "pass")

    def test_localize_prefers_v3_isolated_report(self) -> None:
        from torchport.diagnosis import localize

        report = {
            "status": "diverged",
            "passed": False,
            "strict": {"first_divergence": {"name": "block.dwconv", "position": 0}, "tensors": []},
            "isolated": {
                "envelope_v3": {
                    "first_divergence": {"name": "norm1#2", "position": 3},
                    "tensors": [
                        {"name": "a", "position": 2, "relative_error": 1e-7},
                        {"name": "norm1#2", "position": 3, "relative_error": 5e-5},
                    ],
                    "metrics": {"failed_tensors": 1},
                }
            },
        }
        location = localize(report)
        self.assertEqual(location["scope"], "norm1")
        self.assertEqual(location["invocation"], 2)
        self.assertAlmostEqual(location["error_jump"], 5e-5 - 1e-7)
