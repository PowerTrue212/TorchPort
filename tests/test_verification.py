from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from torchport.verification import VerificationWorkflow


class VerificationWorkflowTests(unittest.TestCase):
    def test_reuses_evidence_and_writes_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            scripts = project / "scripts"
            scripts.mkdir()
            (scripts / "run_pytorch_reference.py").write_text("", encoding="utf-8")
            run_root = project / "artifacts" / "runs" / "sample"
            source_root = run_root / "pytorch"
            source_tensors = source_root / "tensors"
            target_root = run_root / "jittor"
            source_tensors.mkdir(parents=True)
            target_root.mkdir(parents=True)

            input_array = np.asarray([1.0, 2.0], dtype=np.float32)
            source_output = np.asarray([3.0], dtype=np.float32)
            target_output = np.asarray([3.000005], dtype=np.float32)
            np.savez(
                source_tensors / "cnn-resnet18.npz",
                input=input_array,
                **{"probe.conv1": input_array, "output": source_output},
            )
            np.savez(
                target_root / "cnn-resnet18.npz",
                input=input_array,
                **{"probe.conv1": input_array, "output": target_output},
            )
            input_hash = "same-input"
            (source_root / "evidence.json").write_text(
                json.dumps(
                    {
                        "framework": "pytorch",
                        "torch_version": "2.6.0",
                        "device": "cpu",
                        "device_name": "CPU",
                        "models": [
                            {
                                "model_id": "cnn-resnet18",
                                "model_class": "ResNet",
                                "input": {"sha256": input_hash},
                                "output": {"sha256": "source-output"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (target_root / "evidence.json").write_text(
                json.dumps(
                    {
                        "framework": "jittor",
                        "jittor_version": "1.3.11.0",
                        "numpy_version": "1.26.4",
                        "device": "cpu",
                        "model_id": "cnn-resnet18",
                        "model_class": "Resnet18",
                        "weights": {"loaded": 102},
                        "input": {"sha256": input_hash},
                        "output": {"sha256": "target-output"},
                    }
                ),
                encoding="utf-8",
            )

            certificate = VerificationWorkflow(project).verify(
                "cnn-resnet18",
                run_root=run_root,
                reuse_existing=True,
            )

            self.assertTrue(certificate["passed"])
            self.assertEqual(certificate["status"], "verified")
            self.assertEqual(certificate["evidence"]["tensor_count"], 3)
            self.assertTrue(certificate["evidence"]["input_identical"])
            self.assertEqual(certificate["target"]["weights_loaded"], 102)
            self.assertTrue((run_root / "certificate.json").is_file())
            self.assertTrue((run_root / "report-operator-strict.json").is_file())


if __name__ == "__main__":
    unittest.main()
