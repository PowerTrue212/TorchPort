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

    def test_ignore_state_is_validated_sorted_and_changes_oracle_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source", "target"):
                path = root / name
                path.mkdir()
                (path / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            raw = {
                "task_id": "ignore-state-test",
                "source": {"repo": "source", "entry": "model.py", "model": "Model"},
                "target": {"repo": "target", "entry": "model.py", "model": "Model"},
                "inputs": [{"shape": [1, 4]}],
            }
            plain = normalize(raw, root)
            raw["protocol"] = {"ignore_state": ["z.buf", "embeddings.position_ids"]}
            task = normalize(raw, root)
            self.assertEqual(task["protocol"]["ignore_state"],
                             ["embeddings.position_ids", "z.buf"])
            # 排除规则改变完整性判定，旧 oracle 必须失效
            self.assertNotEqual(plain["digest"], task["digest"])
            for bad in ([], ["a", "a"], [""], "embeddings.position_ids"):
                raw["protocol"] = {"ignore_state": bad}
                with self.assertRaisesRegex(ValueError, "ignore_state"):
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

    def test_sketch_dim_is_validated_and_exclusive_with_observe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source", "target"):
                path = root / name
                path.mkdir()
                (path / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            raw = {
                "task_id": "train-sketch-test",
                "source": {"repo": "source", "entry": "model.py", "model": "Model"},
                "target": {"repo": "target", "entry": "model.py", "model": "Model"},
                "inputs": [{"shape": [2, 1]}],
                "protocol": {"profile": "train-step",
                             "train": {"optimizer": "o", "step": "s", "sketch_dim": 4}},
            }
            task = normalize(raw, root)
            self.assertEqual(task["protocol"]["train"]["sketch_dim"], 4)
            plain = dict(raw, protocol={"profile": "train-step",
                                        "train": {"optimizer": "o", "step": "s"}})
            # sketch 改变采集内容，必须使用新 oracle
            self.assertNotEqual(normalize(plain, root)["digest"], task["digest"])
            for bad in (0, -1, True, 2.0):
                raw["protocol"]["train"]["sketch_dim"] = bad
                with self.assertRaisesRegex(ValueError, "sketch_dim"):
                    normalize(raw, root)
            raw["protocol"]["train"].update(sketch_dim=4, observe=["b"])
            with self.assertRaisesRegex(ValueError, "sketch_dim"):
                normalize(raw, root)

    def test_update_sketch_is_linear_and_small_tensors_are_exact(self) -> None:
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=1000), rng.normal(size=1000)
        np.testing.assert_allclose(
            training.update_sketch(a, 64) - training.update_sketch(b, 64),
            training.update_sketch(a - b, 64),
        )
        self.assertEqual(training.update_sketch(a, 64).shape, (64,))
        np.testing.assert_array_equal(training.update_sketch(a[:10], 64), a[:10])
        # 单个元素偏差必然落到某个桶上
        delta = np.zeros(1000)
        delta[123] = 1e-3
        self.assertAlmostEqual(np.abs(training.update_sketch(delta, 64)).max(), 1e-3)

    def test_sketch_mode_records_all_updates_and_coverage(self) -> None:
        model = Model()
        model.parameters["a"].value = np.arange(5, dtype=np.float32)
        model.parameters["a"].shape = (5,)
        optimizer = type("Optimizer", (), {
            "param_groups": [{"lr": 0.1, "params": list(model.parameters.values())}]
        })()

        def step(current_model, _optimizer):
            for parameter in current_model.parameters.values():
                parameter.value = parameter.value + 1
            return 0.5

        scope = {"build_optimizer": lambda _model: optimizer, "train_step": step}
        protocol = {"optimizer": "build_optimizer", "step": "train_step", "sketch_dim": 2}
        with (
            patch.object(training, "_batches", return_value=iter([((), {}), ((), {})])),
            patch.object(training, "_scalar", side_effect=lambda value, _kind: np.asarray([value])),
            patch.object(training, "_to_numpy", side_effect=lambda value, _kind: value.value.copy()),
            patch.object(training, "_grad_norm", return_value=None),
        ):
            outputs, info = training.run_steps(
                model, "source", "cpu", scope, protocol, {}, [], 2,
            )
        self.assertNotIn("output.param.0.a", outputs)
        self.assertEqual(outputs["output.update.1.a"].shape, (2,))  # 5 个元素压到 2 个桶
        np.testing.assert_array_equal(outputs["output.update.1.b"], [2.0])
        np.testing.assert_allclose(outputs["output.ustat.1.a"], [np.sqrt(20), 2.0, 10.0])
        self.assertEqual(info["sketch_dim"], 2)
        self.assertEqual(info["coverage"], {"lr_changed": False, "grad_norms": [None, None]})

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
