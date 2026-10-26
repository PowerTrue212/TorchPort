"""A bounded training observation still checks the full trainable set."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from torchport.capture import training
from torchport.contracts import normalize
from torchport.verification.service import _exact_train_checks


class Parameter:
    def __init__(self) -> None:
        self.value = np.asarray([0.0], dtype=np.float32)
        self.shape = self.value.shape
        self.requires_grad = True


class Model:
    def __init__(self) -> None:
        self.parameters = {"a": Parameter(), "b": Parameter()}

    def train(self) -> None:
        return None

    def named_parameters(self):
        return self.parameters.items()


class TrainObservationTests(unittest.TestCase):
    def test_task_keeps_unique_observation_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source", "target"):
                path = root / name
                path.mkdir()
                (path / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            raw = {
                "task_id": "train-observe-test",
                "source": {"repo": "source", "entry": "model.py", "model": "Model"},
                "target": {"repo": "target", "entry": "model.py", "model": "Model"},
                "inputs": [{"shape": [2, 1]}],
                "protocol": {"profile": "train-step",
                             "train": {"optimizer": "build_optimizer", "step": "train_step",
                                       "observe": ["b"]}},
            }
            task = normalize(raw, root)
            self.assertEqual(task["protocol"]["train"]["observe"], ["b"])
            raw["protocol"]["train"]["observe"] = ["b", "b"]
            with self.assertRaisesRegex(ValueError, "observe"):
                normalize(raw, root)

    def test_selected_output_and_full_trainable_check(self) -> None:
        model = Model()
        optimizer = type("Optimizer", (), {
            "param_groups": [{"lr": 0.1, "params": list(model.parameters.values())}]
        })()

        def step(current_model, _optimizer):
            for parameter in current_model.parameters.values():
                parameter.value += 1
            return 0.5

        scope = {"build_optimizer": lambda _model: optimizer, "train_step": step}
        protocol = {"optimizer": "build_optimizer", "step": "train_step", "observe": ["b"]}
        with (
            patch.object(training, "_batches", return_value=iter([((), {}), ((), {})])),
            patch.object(training, "_scalar", side_effect=lambda value, _kind: np.asarray([value])),
            patch.object(training, "_to_numpy", side_effect=lambda value, _kind: value.value.copy()),
        ):
            outputs, info = training.run_steps(
                model, "source", "cpu", scope, protocol, {}, [], 2,
            )
        self.assertEqual(info["trainable"], ["a", "b"])
        self.assertEqual(info["trainable_elements"], 2)
        self.assertEqual(info["observed"], ["b"])
        self.assertEqual(info["observed_elements"], 1)
        self.assertEqual(sorted(outputs), [
            "output.loss.0", "output.loss.1", "output.param.0.b", "output.param.1.b",
        ])
        np.testing.assert_array_equal(outputs["output.param.1.b"], [2.0])
        self.assertEqual(_exact_train_checks(info, {**info, "observed": ["a"]})[0]["name"],
                         "train.observed")

    def test_unknown_observation_fails(self) -> None:
        model = Model()
        optimizer = type("Optimizer", (), {
            "param_groups": [{"lr": 0.1, "params": list(model.parameters.values())}]
        })()
        scope = {"build_optimizer": lambda _model: optimizer,
                 "train_step": lambda *_args: 0.5}
        with self.assertRaisesRegex(ValueError, "不在可训练集合"):
            training.run_steps(model, "source", "cpu", scope,
                               {"optimizer": "build_optimizer", "step": "train_step",
                                "observe": ["missing"]}, {}, [], 2)


if __name__ == "__main__":
    unittest.main()
