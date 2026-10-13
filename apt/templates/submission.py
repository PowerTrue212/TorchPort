"""APT PyTorch-to-Jittor migration contract.

The migration agent may add any Jittor-native modules beside this file, but the
common evaluator imports and calls only ``run_experiment``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_experiment(
    *,
    protocol: dict[str, Any],
    data_root: Path,
    pretrained_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Train all 10 APT tasks and run final-model inference on 10,000 test images.

    Return a JSON-serializable dictionary with:

    - ``framework`` equal to ``"jittor"``;
    - ``test_samples`` equal to 10,000.

    Write ``output_dir / "accuracy_matrix.npy"`` as float32 [10, 10], where
    row ``t`` contains accuracies on tasks ``0..t`` immediately after training
    task ``t`` and the unused upper triangle is NaN. The evaluator derives the
    paper's Average Accuracy and Forgetting metrics from this matrix.

    Also write ``output_dir / "predictions.npz"`` containing float32 ``logits``
    with shape [10000, 100] and int64 ``targets``, ``predictions``, and
    ``task_ids`` with shape [10000]. Preprocessing, class order, seed, optimizer,
    and all hyperparameters must come from ``protocol``. No task identity may be
    provided to the model during inference.
    """
    raise NotImplementedError("migration agent must implement the full Jittor experiment")
