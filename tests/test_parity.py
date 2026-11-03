from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from torchport.comparison import compare_tensor_archives


class TensorParityTests(unittest.TestCase):
    def test_complex_imaginary_difference_is_compared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            np.savez(reference, output=np.asarray([1 + 2j], dtype=np.complex64))
            np.savez(candidate, output=np.asarray([1 + 3j], dtype=np.complex64))
            report = compare_tensor_archives(reference, candidate, profile="fp32-inference")
        self.assertFalse(report["passed"])
        self.assertEqual(report["first_divergence"]["name"], "output")

    def test_strict_trace_skips_large_sketch_but_compares_small_values_exactly(self) -> None:
        from torchport.capture import SKETCH_DIM, sketch_layers

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            large = np.arange(SKETCH_DIM + 1, dtype=np.float32)
            np.savez(
                reference,
                **sketch_layers({"large": large, "small": np.asarray([1.0], dtype=np.float32)}),
            )
            np.savez(
                candidate,
                **sketch_layers({"large": large + 100, "small": np.asarray([2.0], dtype=np.float32)}),
            )

            report = compare_tensor_archives(
                reference,
                candidate,
                profile="operator-strict",
                skip_sketched=True,
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["skipped_sketched"], ["large"])
        self.assertEqual(report["first_divergence"]["name"], "small")
        self.assertEqual(report["metrics"]["compared_tensors"], 1)
        self.assertEqual(report["metrics"]["skipped_sketched_tensors"], 1)

    def test_strict_trace_still_rejects_sketched_original_shape_mismatch(self) -> None:
        from torchport.capture import SKETCH_DIM, sketch_layers

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            values = np.arange(SKETCH_DIM + 1, dtype=np.float32)
            np.savez(reference, **sketch_layers({"large": values}))
            np.savez(candidate, **sketch_layers({"large": values.reshape(1, -1)}))

            report = compare_tensor_archives(
                reference,
                candidate,
                profile="operator-strict",
                skip_sketched=True,
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["first_divergence"]["name"], "large")
        self.assertEqual(report["skipped_sketched"], [])

    def test_nested_outputs_satisfy_fp32_output_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            arrays = {
                "output.0": np.asarray([1.0], dtype=np.float32),
                "output.1": np.asarray([2.0], dtype=np.float32),
            }
            np.savez(reference, **arrays)
            np.savez(candidate, **arrays)

            report = compare_tensor_archives(reference, candidate)

            self.assertTrue(report["passed"])

    def test_nested_output_is_gating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            same = np.ones((1, 2), dtype=np.float32)
            np.savez(reference, input=same, output=same, **{"output.aux": same})
            np.savez(
                candidate,
                input=same,
                output=same,
                **{"output.aux": np.zeros((1, 2), dtype=np.float32)},
            )

            report = compare_tensor_archives(reference, candidate)

            self.assertFalse(report["passed"])
            self.assertEqual(report["first_gating_divergence"]["name"], "output.aux")
            self.assertEqual(report["metrics"]["mismatched_tensors"], 1)
            self.assertEqual(report["metrics"]["mismatched_elements"], 2)

    def test_matching_archives_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            arrays = {
                "input": np.asarray([1.0, 2.0], dtype=np.float32),
                "probe.layer1": np.asarray([3.0, 4.0], dtype=np.float32),
                "output": np.asarray([5.0], dtype=np.float32),
            }
            np.savez(reference, **arrays)
            np.savez(candidate, **arrays)

            report = compare_tensor_archives(
                reference,
                candidate,
                profile="operator-strict",
            )

            self.assertTrue(report["passed"])
            self.assertIsNone(report["first_divergence"])

    def test_scalar_singleton_shape_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            np.savez(reference, output=np.asarray(1.0, dtype=np.float32))
            np.savez(candidate, output=np.asarray([1.0], dtype=np.float32))

            report = compare_tensor_archives(reference, candidate)

            self.assertTrue(report["passed"])
            self.assertEqual(report["tensors"][0]["shape_normalized"], "scalar-singleton")

    def test_reports_first_numeric_divergence_in_reference_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            np.savez(
                reference,
                input=np.asarray([1.0], dtype=np.float32),
                **{
                    "probe.layer1": np.asarray([2.0], dtype=np.float32),
                    "probe.layer2": np.asarray([3.0], dtype=np.float32),
                },
            )
            np.savez(
                candidate,
                input=np.asarray([1.0], dtype=np.float32),
                **{
                    "probe.layer1": np.asarray([20.0], dtype=np.float32),
                    "probe.layer2": np.asarray([30.0], dtype=np.float32),
                },
            )

            report = compare_tensor_archives(
                reference,
                candidate,
                profile="operator-strict",
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["first_divergence"]["name"], "probe.layer1")
            self.assertEqual(report["first_divergence"]["position"], 1)

    def test_reports_missing_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            np.savez(reference, input=np.asarray([1.0]), output=np.asarray([2.0]))
            np.savez(candidate, input=np.asarray([1.0]))

            report = compare_tensor_archives(reference, candidate)

            self.assertEqual(report["missing_in_candidate"], ["output"])
            self.assertEqual(report["first_divergence"]["name"], "output")
            self.assertEqual(report["metrics"]["missing_tensors"], 1)
            self.assertEqual(report["metrics"]["mismatched_tensors"], 1)

    def test_common_only_comparison_reports_coverage_without_failing_on_missing_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            np.savez(
                reference,
                first=np.asarray([1.0], dtype=np.float32),
                source_only=np.asarray([2.0], dtype=np.float32),
            )
            np.savez(candidate, first=np.asarray([1.0], dtype=np.float32))

            report = compare_tensor_archives(
                reference,
                candidate,
                profile="operator-strict",
                require_all=False,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["missing_in_candidate"], ["source_only"])
            self.assertEqual(report["metrics"]["compared_tensors"], 1)

    def test_tolerance_profiles_distinguish_inference_from_operator_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            np.savez(reference, output=np.asarray([0.0], dtype=np.float32))
            np.savez(candidate, output=np.asarray([5e-5], dtype=np.float32))

            inference_report = compare_tensor_archives(reference, candidate)
            strict_report = compare_tensor_archives(
                reference,
                candidate,
                profile="operator-strict",
            )

            self.assertTrue(inference_report["passed"])
            self.assertFalse(strict_report["passed"])

    def test_inference_profile_keeps_internal_divergence_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.npz"
            candidate = root / "candidate.npz"
            np.savez(
                reference,
                input=np.asarray([1.0], dtype=np.float32),
                **{
                    "probe.layer1.0.conv1": np.asarray([2.0], dtype=np.float32),
                    "output": np.asarray([3.0], dtype=np.float32),
                },
            )
            np.savez(
                candidate,
                input=np.asarray([1.0], dtype=np.float32),
                **{
                    "probe.layer1.0.conv1": np.asarray([20.0], dtype=np.float32),
                    "output": np.asarray([3.0], dtype=np.float32),
                },
            )

            report = compare_tensor_archives(reference, candidate)

            self.assertTrue(report["passed"])
            self.assertEqual(
                report["first_divergence"]["name"],
                "probe.layer1.0.conv1",
            )
            self.assertIsNone(report["first_gating_divergence"])


if __name__ == "__main__":
    unittest.main()


class NoiseCalibratedTests(unittest.TestCase):
    def _write(self, root: Path, **archives: dict[str, np.ndarray]) -> list[Path]:
        paths = []
        for name, arrays in archives.items():
            path = root / f"{name}.npz"
            np.savez(path, **arrays)
            paths.append(path)
        return paths

    def test_candidate_within_noise_passes_and_outlier_fails(self) -> None:
        from torchport.comparison import compare_noise_calibrated

        truth = np.linspace(-1, 1, 100)
        ref = (truth + 1e-6).astype(np.float32)
        alt = (truth - 1e-6).astype(np.float32)
        good = (truth + 2e-6).astype(np.float32)
        bad = (truth + 1e-3).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = self._write(
                root,
                reference={"leaf": ref, "leaf#2": ref, "parent": ref},
                candidate={"leaf": good, "leaf#2": bad, "parent": bad},
                fp64={"leaf": truth, "leaf#2": truth, "parent": truth},
                alt={"leaf": alt, "leaf#2": alt, "parent": alt},
            )
            report = compare_noise_calibrated(*paths, leaves={"leaf"})

        by_name = {item["name"]: item for item in report["tensors"]}
        self.assertTrue(by_name["leaf"]["passed"])
        self.assertFalse(by_name["leaf#2"]["passed"])
        self.assertFalse(by_name["parent"]["gating"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["first_gating_divergence"]["name"], "leaf#2")
        self.assertEqual(report["metrics"]["failed_reported_tensors"], 1)

    def test_composite_module_outlier_also_fails_report(self) -> None:
        from torchport.comparison import compare_noise_calibrated

        baseline = np.linspace(-1, 1, 100)
        reference = baseline.astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = self._write(
                root,
                reference={"leaf": reference, "parent": reference},
                candidate={
                    "leaf": (baseline + 1e-7).astype(np.float32),
                    "parent": (baseline + 1e-3).astype(np.float32),
                },
                fp64={"leaf": baseline, "parent": baseline},
                alt={"leaf": reference, "parent": reference},
            )
            report = compare_noise_calibrated(*paths, leaves={"leaf"})

        self.assertFalse(report["passed"])
        self.assertEqual(report["first_divergence"]["name"], "parent")
        self.assertIsNone(report["first_gating_divergence"])
        self.assertEqual(report["metrics"]["failed_tensors"], 1)

    def test_leaf_key_strips_call_index_and_output_index(self) -> None:
        from torchport.comparison import _leaf_key

        leaves = {"model.encoder.block1.0.attn"}
        self.assertTrue(_leaf_key("model.encoder.block1.0.attn#3", leaves))
        self.assertTrue(_leaf_key("model.encoder.block1.0.attn.1", leaves))
        self.assertFalse(_leaf_key("model.encoder.block1.0", leaves))
        self.assertFalse(_leaf_key("model.encoder.block1.0.attn.q", leaves))


class LeafMarkingTests(unittest.TestCase):
    def test_parent_modules_are_not_leaves(self) -> None:
        from torchport.capture import _mark_leaves

        calls = [{"module": "a"}, {"module": "a.b"}, {"module": "ab"}, {"module": "a.b"}]
        _mark_leaves(calls)
        self.assertEqual([c["leaf"] for c in calls], [False, True, True, True])
