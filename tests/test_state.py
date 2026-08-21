from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from torchport.probe.state import load_state, save_state


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


class Model:
    def __init__(self, state) -> None:
        self.state = state
        self.loaded = {}

    def state_dict(self):
        return self.state

    def load_parameters(self, weights):
        self.loaded = weights


class StateTests(unittest.TestCase):
    def test_batch_counter_is_not_portable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.npz"
            source = Model(
                {
                    "bn.weight": Tensor([1.0, 2.0]),
                    "bn.num_batches_tracked": Tensor([3]),
                }
            )

            saved = save_state(source, path)
            target = Model({"bn.weight": Tensor([0.0, 0.0])})
            total, loaded = load_state(target, path)

        self.assertEqual(saved, 1)
        self.assertEqual((total, loaded), (1, 1))
        self.assertEqual(set(target.loaded), {"bn.weight"})


if __name__ == "__main__":
    unittest.main()
