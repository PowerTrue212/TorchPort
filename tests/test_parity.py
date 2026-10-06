from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from torchport.parity import compare_tensor_archives


class TensorParityTests(unittest.TestCase):
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
