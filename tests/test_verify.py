from __future__ import annotations

import unittest

from torchport.verify import judge, judge_repeats


def _info(weights: int = 4, loaded: int = 4, digest: str | None = "in") -> tuple[dict, dict]:
    hashes = {"input": digest}
    source = {"weights": weights, "hashes": dict(hashes)}
    target = {"loaded": loaded, "hashes": dict(hashes)}
    return source, target


def _output(**overrides) -> dict:
    item = {
        "name": "output",
        "passed": True,
        "reference_shape": [1, 2],
        "candidate_shape": [1, 2],
        "reference_dtype": "float32",
        "candidate_dtype": "float32",
    }
    item.update(overrides)
    return item


def _report(passed: bool = True, **overrides) -> dict:
    report = {
        "passed": passed,
        "first_divergence": None,
        "first_gating_divergence": None,
        "missing_in_candidate": [],
        "extra_in_candidate": [],
        "tensors": [_output()],
    }
    report.update(overrides)
    return report


class JudgeTests(unittest.TestCase):
    def test_pass(self) -> None:
        result = judge(*_info(), _report(), _report())
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["aligned"])
        self.assertTrue(result["strict_aligned"])
        self.assertEqual(result["mismatch"]["fp32"]["mismatched_elements"], 0)

    def test_run_missing_fails(self) -> None:
        result = judge(None, None, None, None)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_code"], "RUN_FAILED")

    def test_partial_weights_fail(self) -> None:
        result = judge(*_info(4, 3), _report(), _report())
        self.assertEqual(result["failure_code"], "WEIGHT_LOAD_FAILED")

    def test_partial_checkpoint_fails(self) -> None:
        source, target = _info()
        source["checkpoint"] = {"requested": True, "complete": False}
        result = judge(source, target, _report(), _report())
        self.assertEqual(result["failure_code"], "WEIGHT_LOAD_FAILED")

    def test_missing_input_digest_fails(self) -> None:
        result = judge(*_info(digest=None), _report(), _report())
        self.assertEqual(result["failure_code"], "INPUT_MISMATCH")

    def test_trace_never_certifies(self) -> None:
        source, target = _info()
        target["mode"] = "trace"
        result = judge(source, target, _report(), _report())
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_code"], "DIAGNOSTIC_RUN")

    def test_missing_output_fails(self) -> None:
        report = _report(missing_in_candidate=["output.flow"])
        result = judge(*_info(), report, _report())
        self.assertEqual(result["failure_code"], "STRUCTURE_MISMATCH")

    def test_dtype_mismatch_fails(self) -> None:
        report = _report(tensors=[_output(candidate_dtype="float64")])
        result = judge(*_info(), report, _report())
        self.assertEqual(result["structural"], "fail")

    def test_scalar_normalization_needs_protocol(self) -> None:
        report = _report(tensors=[_output(candidate_shape=[2], shape_normalized="scalar-singleton")])
        self.assertFalse(judge(*_info(), report, _report())["passed"])
        allowed = judge(*_info(), report, _report(), {"allow_scalar_singleton": True})
        self.assertTrue(allowed["passed"])

    def test_numeric_failure(self) -> None:
        report = _report(
            passed=False,
            first_gating_divergence={"name": "output", "position": 1, "reason": "数值超出容差"},
            tensors=[_output(passed=False, max_absolute_error=0.5)],
        )
        result = judge(*_info(), report, _report())
        self.assertEqual(result["failure_code"], "PARITY_FAILED")
        self.assertFalse(result["aligned"])
        # summary 必须给出首个分歧张量与误差，不能为空
        self.assertIn("output", result["summary"])
        self.assertIn("0.5", result["summary"])


class RepeatTests(unittest.TestCase):
    def test_one_failure_fails_all(self) -> None:
        ok = judge(*_info(), _report(), _report())
        bad = judge(*_info(), _report(passed=False), _report())
        result = judge_repeats([ok, bad, ok], 3)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_code"], "PARITY_FAILED")

    def test_too_few_runs(self) -> None:
        ok = judge(*_info(), _report(), _report())
        self.assertEqual(judge_repeats([ok], 3)["failure_code"], "TOO_FEW_RUNS")


if __name__ == "__main__":
    unittest.main()
