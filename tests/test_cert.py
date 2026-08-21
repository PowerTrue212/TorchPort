from __future__ import annotations

import unittest

from torchport.probe.cert import build_cert


def _info(weights: int, loaded: int) -> tuple[dict, dict]:
    hashes = {"input": "input", "archive": "archive", "weights": "weights"}
    source = {
        "weights": weights,
        "tensors": 3,
        "framework": {"name": "pytorch"},
        "hashes": hashes,
    }
    target = {
        "loaded": loaded,
        "tensors": 3,
        "framework": {"name": "jittor"},
        "hashes": hashes,
    }
    return source, target


def _report() -> dict:
    return {
        "passed": True,
        "first_divergence": None,
        "first_gating_divergence": None,
        "missing_in_candidate": [],
        "extra_in_candidate": [],
    }


class CertTests(unittest.TestCase):
    def test_partial_target_fails(self) -> None:
        source, target = _info(4, 3)

        cert = build_cert("source", "target", {}, source, target, _report(), _report())

        self.assertFalse(cert["passed"])
        self.assertFalse(cert["weights"]["complete"])

    def test_partial_checkpoint_fails(self) -> None:
        source, target = _info(4, 4)
        source["checkpoint"] = {"requested": True, "complete": False}

        cert = build_cert("source", "target", {}, source, target, _report(), _report())

        self.assertFalse(cert["passed"])
        self.assertFalse(cert["checkpoint"]["complete"])

    def test_non_gating_first_stays_diagnostic(self) -> None:
        source, target = _info(4, 4)
        fp32 = _report()
        fp32["first_divergence"] = {
            "name": "probe.relu",
            "position": 2,
            "reason": "候选证据缺少该张量",
        }

        cert = build_cert("source", "target", {}, source, target, fp32, _report())

        self.assertTrue(cert["passed"])
        self.assertIsNone(cert["fp32"]["first"])
        self.assertEqual(cert["fp32"]["first_any"]["name"], "probe.relu")


if __name__ == "__main__":
    unittest.main()
