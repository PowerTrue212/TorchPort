from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from torchport.models import MigrationTarget, SourceFramework
from torchport.planner import MigrationPlanner
from torchport.scanner import RepositoryScanner


class RepositoryScannerTests(unittest.TestCase):
    def test_detects_pytorch_entrypoint_weights_and_custom_operator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "train.py").write_text(
                "import torch\nfrom torch import nn\n",
                encoding="utf-8",
            )
            (root / "model.pth").write_bytes(b"weights")
            (root / "fused.cu").write_text("// kernel", encoding="utf-8")
            (root / "requirements.txt").write_text("torch\n", encoding="utf-8")

            result = RepositoryScanner().scan(root)

            self.assertEqual(result.primary_framework, SourceFramework.PYTORCH)
            self.assertEqual(result.entrypoints, ("train.py",))
            self.assertEqual(result.weight_files, ("model.pth",))
            self.assertEqual(result.custom_operator_files, ("fused.cu",))
            self.assertTrue(any("自定义算子" in risk for risk in result.risks))

    def test_detects_caffe_from_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "deploy.prototxt").write_text("layer {}", encoding="utf-8")
            (root / "weights.caffemodel").write_bytes(b"weights")

            result = RepositoryScanner().scan(root)

            self.assertEqual(result.primary_framework, SourceFramework.CAFFE)

    def test_detects_framework_from_requirements_and_prunes_virtualenv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "requirements-cuda.txt").write_text(
                "torch==2.6.0\ntorchvision>=0.21\n",
                encoding="utf-8",
            )
            virtualenv = root / ".venv"
            virtualenv.mkdir()
            (virtualenv / "fake.py").write_text("import tensorflow\n", encoding="utf-8")

            result = RepositoryScanner().scan(root)

            self.assertEqual(result.frameworks, (SourceFramework.PYTORCH,))
            self.assertEqual(result.dependency_files, ("requirements-cuda.txt",))
            self.assertNotIn(".venv/fake.py", result.source_files)

    def test_planner_rejects_jtorch_for_tensorflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "main.py").write_text("import tensorflow as tf\n", encoding="utf-8")
            inspection = RepositoryScanner().scan(root)

            plan = MigrationPlanner().create(inspection, MigrationTarget.JTORCH)

            self.assertTrue(any("JTorch" in blocker for blocker in plan.blockers))


if __name__ == "__main__":
    unittest.main()
