from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from torchport.capture import (
    OpTrace,
    _check_bridge,
    _decode_call_value,
    _encode_call_value,
    _matched,
    load_state,
    make_data,
    save_state,
)
from torchport.task import normalize
from torchport.verify import judge


def _identity_patches(module, names):
    return (
        patch.object(OpTrace, "_modules", return_value=[(module, names)]),
        patch("torchport.capture.find_tensor", side_effect=lambda value, _kind: value),
        patch("torchport.capture.to_array", side_effect=lambda value, _kind: np.asarray(value)),
    )


class OpTraceTests(unittest.TestCase):
    def test_trace_records_and_restores(self) -> None:
        def original(value):
            return value + 1

        module = SimpleNamespace(relu=original)
        scope = {"act": original}
        values: dict = {}
        first, second, third = _identity_patches(module, ("relu",))
        with first, second, third:
            trace = OpTrace("source", values, [scope])
            trace.start()
            self.assertEqual(module.relu(2), 3)
            self.assertEqual(scope["act"](3), 4)
            trace.stop()

        self.assertEqual(values["op.relu"], 3)
        self.assertEqual(values["op.relu#2"], 4)
        self.assertIs(module.relu, original)
        self.assertIs(scope["act"], original)

    def test_wrong_target_op_is_executed(self) -> None:
        # 故意改错的目标 relu 必须被真实调用，不能被源端结果掩盖
        calls = []

        def broken(_value):
            calls.append(1)
            return np.asarray([999])

        module = SimpleNamespace(relu=broken)
        values: dict = {}
        first, second, third = _identity_patches(module, ("relu",))
        with first, second, third:
            trace = OpTrace("target", values)
            trace.start()
            output = module.relu(np.asarray([1]))
            trace.stop()

        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(output, [999])


class Tensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)
        self.shape = self.value.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class ModuleInputTests(unittest.TestCase):
    def test_nested_call_round_trip_preserves_tensors_and_scalars(self) -> None:
        tensor = Tensor([[1.0, 2.0]])
        arrays = {}
        identities = {}
        value = ((tensor, 4), {"mask": tensor, "enabled": True})

        with (
            patch("torchport.capture._tensor_type", return_value=Tensor),
            patch("torchport.capture._to_numpy", side_effect=lambda item, _kind: item.numpy()),
        ):
            encoded = _encode_call_value(value, "source", arrays, identities)
        with patch("torchport.capture._from_numpy", side_effect=lambda item, _kind: Tensor(item)):
            decoded = _decode_call_value(encoded, arrays, "target")

        self.assertEqual(len(arrays), 1)
        self.assertEqual(decoded[0][1], 4)
        self.assertTrue(decoded[1]["enabled"])
        np.testing.assert_array_equal(decoded[0][0].numpy(), tensor.numpy())
        np.testing.assert_array_equal(decoded[1]["mask"].numpy(), tensor.numpy())


class Model:
    def __init__(self, state, corrupt: bool = False) -> None:
        self.state = state
        self.corrupt = corrupt

    def state_dict(self):
        return self.state

    def load_parameters(self, weights):
        for name, value in weights.items():
            self.state[name] = Tensor(value + (1 if self.corrupt else 0))


class StateTests(unittest.TestCase):
    def test_scalar_state_matches_singleton_target(self) -> None:
        matched = _matched(
            {"scale": np.asarray(2.0, dtype=np.float32)},
            {"scale": np.zeros((1,), dtype=np.float32)},
        )

        self.assertEqual(matched["scale"].shape, (1,))

    def _save(self, root: Path) -> Path:
        path = root / "state.npz"
        source = Model(
            {
                "w": Tensor(np.ones((2, 2), dtype=np.float32)),
                "bn.num_batches_tracked": Tensor(np.asarray(3)),
                "text.attn_mask": Tensor(np.ones((2, 2), dtype=np.float32)),
            }
        )
        self.assertEqual(save_state(source, path), 1)  # 计数器和确定性 mask 不可移植
        return path

    def test_load_verifies_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(Path(directory))
            target = Model({"w": Tensor(np.zeros((2, 2), dtype=np.float32))})
            with patch("torchport.capture._to_numpy", side_effect=lambda t, _k: t.numpy()):
                state = load_state(target, path)
        self.assertEqual(state["loaded"], 1)
        self.assertEqual(state["value_mismatch"], [])

    def test_wrong_loaded_value_is_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(Path(directory))
            target = Model({"w": Tensor(np.zeros((2, 2), dtype=np.float32))}, corrupt=True)
            with patch("torchport.capture._to_numpy", side_effect=lambda t, _k: t.numpy()):
                state = load_state(target, path)
        self.assertEqual(state["loaded"], 0)
        self.assertEqual(state["value_mismatch"], ["w"])

    def test_shape_mismatch_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(Path(directory))
            target = Model({"w": Tensor(np.zeros((3, 2), dtype=np.float32))})
            with patch("torchport.capture._to_numpy", side_effect=lambda t, _k: t.numpy()):
                state = load_state(target, path)
        self.assertEqual(state["loaded"], 0)
        self.assertEqual(state["missing"], ["w"])

    def test_extra_target_parameter_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(Path(directory))
            target = Model(
                {
                    "w": Tensor(np.zeros((2, 2), dtype=np.float32)),
                    "extra": Tensor(np.ones((1,), dtype=np.float32)),
                }
            )
            with patch("torchport.capture._to_numpy", side_effect=lambda t, _k: t.numpy()):
                state = load_state(target, path)
        tensor = {
            "name": "output",
            "passed": True,
            "reference_shape": [1],
            "candidate_shape": [1],
            "reference_dtype": "float32",
            "candidate_dtype": "float32",
        }
        parity = {
            "passed": True,
            "first_divergence": None,
            "first_gating_divergence": None,
            "missing_in_candidate": [],
            "extra_in_candidate": [],
            "tensors": [tensor],
        }
        source = {"weights": 1, "hashes": {"input": "same"}}
        candidate = {"loaded": 1, "state": state, "hashes": {"input": "same"}}
        result = judge(source, candidate, parity, parity)

        self.assertEqual(state["extra"], ["extra"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_code"], "WEIGHT_LOAD_FAILED")


class FixedInputTests(unittest.TestCase):
    def test_npy_input_is_content_addressed_and_loaded_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("source", "target"):
                (root / name).mkdir()
                (root / name / "main.py").write_text("class Net: pass\n", encoding="utf-8")
            expected = np.arange(6, dtype=np.float32).reshape(2, 3)
            np.save(root / "input.npy", expected)
            raw = {
                "task_id": "fixed-input",
                "source": {"repo": "source", "entry": "main.py", "model": "Net"},
                "target": {"repo": "target", "entry": "main.py", "model": "Net"},
                "inputs": [
                    {
                        "path": "input.npy",
                        "shape": [2, 3],
                        "dtype": "float32",
                    }
                ],
            }

            first = normalize(raw, root)
            loaded = make_data(first["inputs"], first["seed"])["input"]
            np.save(root / "input.npy", expected + 1)
            second = normalize(raw, root)

        np.testing.assert_array_equal(loaded, expected)
        self.assertNotEqual(first["inputs"][0]["sha256"], second["inputs"][0]["sha256"])
        self.assertNotEqual(first["digest"], second["digest"])


class Array:
    def __init__(self, value, valid: bool) -> None:
        self.value = np.asarray(value) if valid else np.zeros_like(value)

    def numpy(self):
        return self.value


class BridgeTests(unittest.TestCase):
    def test_valid(self) -> None:
        module = SimpleNamespace(array=lambda value: Array(value, True))
        with patch.dict(sys.modules, {"jittor": module}):
            _check_bridge("target")

    def test_invalid(self) -> None:
        module = SimpleNamespace(array=lambda value: Array(value, False))
        with patch.dict(sys.modules, {"jittor": module}):
            with self.assertRaisesRegex(ValueError, "NumPy"):
                _check_bridge("target")


if __name__ == "__main__":
    unittest.main()
