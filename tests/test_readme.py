from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from torchport.migrate import RepoCheck, RepositoryMigrator
from torchport.models import MigrationTarget
from torchport.probe import infer_probe


class ReadmeTests(unittest.TestCase):
    def test_readme_args(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "readme"

        spec = infer_probe(root, "main.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "model.py")
        self.assertEqual(spec["model"], "Net")
        self.assertEqual(spec["kwargs"], {"upscale_factor": 3})
        self.assertEqual(spec["shape"], [1, 1, 32, 32])

    def test_module_command(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "module"

        spec = infer_probe(root, "modnet/train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "modnet/model.py")
        self.assertEqual(spec["model"], "Net")
        self.assertEqual(spec["kwargs"], {"width": 5, "hidden": 7})
        self.assertEqual(spec["shape"], [1, 1, 32, 32])

    def test_launcher_command(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "launch"

        spec = infer_probe(root, "launchpkg/train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "launchpkg/model.py")
        self.assertEqual(spec["model"], "Net")
        self.assertEqual(spec["kwargs"], {"width": 8, "hidden": 10})
        self.assertEqual(spec["shape"], [1, 8])
        self.assertNotIn("mixed_precision", spec["kwargs"])
        self.assertNotIn("nproc_per_node", spec["kwargs"])

    def test_config_command(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "configcmd"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "model.py")
        self.assertEqual(spec["model"], "Net")
        self.assertEqual(spec["kwargs"], {"hidden": 6, "classes": 2})
        self.assertEqual(spec["shape"], [1, 1, 32, 32])

    def test_omega_command(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "omega"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "model.py")
        self.assertEqual(spec["model"], "Net")
        self.assertEqual(spec["kwargs"], {"width": 4, "hidden": 6})
        self.assertEqual(spec["shape"], [1, 4])

    def test_omega_container(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "omegac"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "model.py")
        self.assertEqual(spec["model"], "Net")
        self.assertEqual(spec["kwargs"], {"width": 4, "hidden": 6})
        self.assertEqual(spec["shape"], [1, 4])

    def test_arch_dict(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "archdict"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "models.py")
        self.assertEqual(spec["model"], "smallnet")
        self.assertEqual(spec["kwargs"], {"width": 4, "hidden": 6})
        self.assertEqual(spec["shape"], [1, 4])

    def test_arch_getattr(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "archget"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "models.py")
        self.assertEqual(spec["model"], "smallnet")
        self.assertEqual(spec["kwargs"], {"width": 4, "hidden": 6})
        self.assertEqual(spec["shape"], [1, 4])

    def test_timm_factory(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "timmfact"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "timm.py")
        self.assertEqual(spec["model"], "create_model")
        self.assertEqual(spec["args"], ["tiny"])
        self.assertEqual(spec["kwargs"], {"in_chans": 3, "num_classes": 2})
        self.assertEqual(spec["shape"], [1, 3, 32, 32])

    def test_torchvision_factory(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "tvfact"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "torchvision/models.py")
        self.assertEqual(spec["model"], "get_model")
        self.assertEqual(spec["args"], ["tiny"])
        self.assertEqual(spec["kwargs"], {"weights": None, "num_classes": 2})
        self.assertEqual(spec["shape"], [1, 3, 32, 32])

    def test_hydra_instantiate(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "hydrainst"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "model.py")
        self.assertEqual(spec["model"], "Net")
        self.assertEqual(spec["kwargs"], {"width": 4, "hidden": 6})
        self.assertEqual(spec["shape"], [1, 4])

    def test_registry_build(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "regbuild"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "registry.py")
        self.assertEqual(spec["model"], "build_model")
        self.assertEqual(spec["args"], [{"type": "tiny", "width": 4, "hidden": 6}])
        self.assertEqual(spec["kwargs"], {})
        self.assertEqual(spec["shape"], [1, 4])

    def test_decorator_registry(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "decreg"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "registry.py")
        self.assertEqual(spec["model"], "build_model")
        self.assertEqual(spec["args"], [{"type": "Net", "width": 4, "hidden": 6}])
        self.assertEqual(spec["shape"], [1, 4])

    def test_named_decorator_registry(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "deccall"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "registry.py")
        self.assertEqual(spec["model"], "build_model")
        self.assertEqual(spec["args"], [{"type": "tiny", "width": 4, "hidden": 6}])
        self.assertEqual(spec["shape"], [1, 4])

    def test_full_config_attr(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "cfgattr"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "registry.py")
        self.assertEqual(spec["model"], "build_model")
        self.assertEqual(
            spec["args"],
            [{"model": {"type": "tiny", "width": 4, "hidden": 6}}],
        )
        self.assertEqual(spec["shape"], [1, 4])

    def test_full_config_item(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "cfgitem"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "registry.py")
        self.assertEqual(spec["model"], "build_model")
        self.assertEqual(
            spec["args"],
            [{"model": {"type": "tiny", "width": 4, "hidden": 6}}],
        )
        self.assertEqual(spec["shape"], [1, 4])

    def test_yacs_merge(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "yacsmerge"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "registry.py")
        self.assertEqual(spec["model"], "build_model")
        self.assertEqual(
            spec["args"],
            [{"TYPE": "tiny", "WIDTH": 4, "HIDDEN": 6}],
        )
        self.assertEqual(spec["shape"], [1, 4])

    def test_yacs_module(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "yacsmod"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "registry.py")
        self.assertEqual(spec["model"], "build_model")
        self.assertEqual(
            spec["args"],
            [{"TYPE": "tiny", "WIDTH": 4, "HIDDEN": 6}],
        )
        self.assertEqual(spec["shape"], [1, 4])

    def test_dynamic_import(self) -> None:
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "dynimp"

        spec = infer_probe(root, "train.py")

        self.assertEqual(spec["status"], "ready")
        self.assertEqual(spec["entry"], "registry.py")
        self.assertEqual(spec["model"], "build_model")
        self.assertEqual(
            spec["args"],
            [{"module": "model", "name": "Net", "width": 4, "hidden": 6}],
        )
        self.assertEqual(spec["shape"], [1, 4])

    def test_local_torchvision_check(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "tvfact"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "output"

            RepositoryMigrator().migrate(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="train.py",
            )
            check = RepoCheck().check(output, runtime=False)

            self.assertTrue(check["passed"])
            self.assertEqual(check["residue"]["imports"], [])

    def test_migration_keeps_model_entry(self) -> None:
        source = Path(__file__).resolve().parents[1] / "benchmarks" / "repos" / "readme"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "output"

            report = RepositoryMigrator().migrate(
                source,
                output,
                target=MigrationTarget.JITTOR_NATIVE,
                entrypoint="main.py",
            )

            self.assertEqual(report["entrypoint"], "main.py")
            self.assertEqual(report["probe"]["entry"], "model.py")
            self.assertEqual(report["probe"]["kwargs"], {"upscale_factor": 3})
            self.assertEqual(report["status"], "converted")
            self.assertTrue(RepoCheck().check(output, runtime=False)["passed"])

    def test_check_uses_model_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            meta = output / ".torchport"
            meta.mkdir(parents=True)
            (output / "main.py").write_text("raise RuntimeError('training entry')\n")
            (output / "model.py").write_text("VALUE = 1\n")
            (meta / "run.py").write_text("raise RuntimeError('runner')\n")
            (meta / "manifest.json").write_text(
                json.dumps(
                    {
                        "target": "jtorch",
                        "status": "converted",
                        "entrypoint": "main.py",
                        "probe": {"entry": "model.py"},
                        "summary": {"blockers": 0},
                        "files": [
                            {"path": "main.py"},
                            {"path": "model.py"},
                        ],
                    }
                )
            )

            report = RepoCheck().check(output, run=True)

            self.assertTrue(report["passed"])
            self.assertEqual(report["runtime"]["mode"], "probe-import")


if __name__ == "__main__":
    unittest.main()
