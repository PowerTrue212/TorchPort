"""Training contract errors are surfaced as frozen enum text in both feedback levels."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.harness.feedback import _contract, diagnostic, public_run
from experiments.harness.protocol import DIAGNOSTIC_FIELDS
from torchport.capture import training
from torchport.capture.training import CONTRACT_MESSAGES, ContractError


class Parameter:
    def __init__(self) -> None:
        self.shape = (1,)
        self.requires_grad = True


class Model:
    def __init__(self) -> None:
        self.param = Parameter()

    def named_parameters(self):
        return [("w", self.param)]


class LossTests(unittest.TestCase):
    def test_python_and_numpy_scalars_are_accepted(self) -> None:
        self.assertEqual(training._scalar(1.5, "target").dtype, np.float32)
        self.assertEqual(training._scalar(np.float64(1.5), "target").dtype, np.float32)
        # fp64 参考路径保持 float64
        self.assertEqual(training._scalar(1.5, "source", dtype=np.float64).dtype, np.float64)
        self.assertEqual(training._scalar(2, "target").shape, (1,))

    def test_bad_loss_types_are_contract_errors(self) -> None:
        for value in (None, True, "x", object()):
            with self.assertRaises(ContractError) as context:
                training._scalar(value, "target")
            self.assertEqual(context.exception.code, "LOSS_TYPE")
        with self.assertRaises(ContractError) as context:
            training._scalar(np.zeros(2), "target")
        self.assertEqual(context.exception.code, "LOSS_NOT_SCALAR")


class OptimizerTests(unittest.TestCase):
    def test_optimizer_without_param_groups(self) -> None:
        with self.assertRaises(ContractError) as context:
            training._trainable(Model(), object())
        self.assertEqual(context.exception.code, "OPTIMIZER_NO_PARAM_GROUPS")

    def test_optimizer_with_foreign_or_no_trainable_params(self) -> None:
        model = Model()
        foreign = type("O", (), {"param_groups": [{"params": [Parameter()], "lr": 1.0}]})()
        with self.assertRaises(ContractError) as context:
            training._trainable(model, foreign)
        self.assertEqual(context.exception.code, "OPTIMIZER_FOREIGN_PARAM")
        empty = type("O", (), {"param_groups": [{"params": [], "lr": 1.0}]})()
        with self.assertRaises(ContractError) as context:
            training._trainable(model, empty)
        self.assertEqual(context.exception.code, "OPTIMIZER_NO_TRAINABLE")

    def test_missing_lr_and_missing_function(self) -> None:
        with self.assertRaises(ContractError) as context:
            training._lr(type("O", (), {"param_groups": [{"params": []}]})())
        self.assertEqual(context.exception.code, "OPTIMIZER_NO_LR")
        with self.assertRaises(ContractError) as context:
            training._callable({}, "train_step")
        self.assertIn("`train_step`", context.exception.message)

    def test_dict_state_is_not_unwrapped(self) -> None:
        # r3 的写法：dict 状态不是 optimizer，也不是 (optimizer, ...) 元组
        state = {"optimizer": object()}
        with self.assertRaises(ContractError):
            training._trainable(Model(), training._optimizer(state))


class FeedbackTests(unittest.TestCase):
    def test_contract_passes_only_frozen_text(self) -> None:
        message = CONTRACT_MESSAGES["LOSS_TYPE"]
        self.assertEqual(_contract({"code": "LOSS_TYPE", "message": message}),
                         {"code": "LOSS_TYPE", "message": message})
        self.assertIsNone(_contract({"code": "LOSS_TYPE", "message": "model.encoder.layer.3"}))
        self.assertIsNone(_contract({"code": "UNKNOWN", "message": message}))
        self.assertIsNone(_contract("LOSS_TYPE"))
        named = ContractError("TRAIN_FUNCTION_MISSING", name="train_step").as_dict()
        self.assertEqual(_contract(named), named)

    def test_diagnostic_reports_contract_stage(self) -> None:
        report = {"failure_code": "CONTRACT_ERROR", "runnable": "fail", "passed": False,
                  "contract": ContractError("LOSS_TYPE").as_dict(),
                  "summary": "secret free text", "problems": ["x"]}
        result = diagnostic(report)
        self.assertEqual(tuple(result), DIAGNOSTIC_FIELDS)
        self.assertEqual(result["stage"], "contract")
        self.assertEqual(result["failure_code"], "CONTRACT_ERROR")
        self.assertEqual(result["contract"]["code"], "LOSS_TYPE")
        self.assertNotIn("summary", result)

    def test_ordinary_run_failure_has_no_contract(self) -> None:
        result = diagnostic({"failure_code": "RUN_FAILED", "runnable": "fail"})
        self.assertEqual(result["stage"], "runnable")
        self.assertIsNone(result["contract"])

    def test_public_run_carries_contract(self) -> None:
        run = {"kind": "torchport.run", "run_id": "r", "ok": False,
               "contract": ContractError("OPTIMIZER_NO_LR").as_dict()}
        self.assertEqual(public_run(run)["contract"]["code"], "OPTIMIZER_NO_LR")
        self.assertIsNone(public_run({"ok": True})["contract"])


if __name__ == "__main__":
    unittest.main()
