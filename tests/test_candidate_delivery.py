from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.harness.candidate import (
    finalize_candidate,
    initialize_git_workspace,
    repository_manifest,
)


class CandidateDeliveryTests(unittest.TestCase):
    def test_manifest_is_path_independent_and_tracks_the_full_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            for repo in (first, second):
                (repo / "models").mkdir(parents=True)
                (repo / "run.py").write_text("from models.net import Net\n", "utf-8")
                (repo / "models" / "net.py").write_text("class Net: pass\n", "utf-8")
                (repo / "config.yaml").write_text("width: 32\n", "utf-8")

            before = repository_manifest(first)
            same = repository_manifest(second)
            (second / "config.yaml").write_text("width: 64\n", "utf-8")
            changed = repository_manifest(second)

        self.assertEqual(before["content_digest"], same["content_digest"])
        self.assertNotEqual(before["content_digest"], changed["content_digest"])
        self.assertEqual(before["file_count"], 3)

    def test_finalizer_saves_snapshot_manifest_and_git_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, output = root / "repo", root / "delivery"
            (repo / "models").mkdir(parents=True)
            (repo / "run.py").write_text("from models.net import Net\n", "utf-8")
            (repo / "models" / "net.py").write_text("class Net: pass\n", "utf-8")
            initialize_git_workspace(repo)
            (repo / "models" / "net.py").write_text("class Net: migrated = True\n", "utf-8")
            (repo / "models" / "new.py").write_text("VALUE = 'new module'\n", "utf-8")

            result = finalize_candidate(repo, output, required_entry="run.py")
            snapshot = Path(result["snapshot"])

            self.assertTrue(result["git"]["available"])
            diff = (output / "candidate.diff").read_text("utf-8")
            self.assertIn("migrated", diff)
            self.assertIn("new module", diff)
            self.assertEqual(
                (snapshot / "models" / "net.py").read_text("utf-8"),
                "class Net: migrated = True\n",
            )
            self.assertEqual(
                result["content_digest"], repository_manifest(snapshot)["content_digest"]
            )

    def test_repository_contract_rejects_an_entry_only_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            (repo / "run.py").write_text("VALUE = 1\n", "utf-8")

            with self.assertRaisesRegex(ValueError, "完整仓库"):
                finalize_candidate(repo, root / "delivery", required_entry="run.py")

    def test_manifest_rejects_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "main.py").write_text("VALUE = 1\n", "utf-8")
            (repo / ".env").write_text("TOKEN=secret\n", "utf-8")

            with self.assertRaisesRegex(ValueError, "凭据"):
                repository_manifest(repo)

    def test_finalizer_rejects_delivery_inside_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "run.py").write_text("VALUE = 1\n", "utf-8")
            (repo / "model.py").write_text("VALUE = 2\n", "utf-8")

            with self.assertRaisesRegex(ValueError, "仓库之外"):
                finalize_candidate(repo, repo / "delivery", required_entry="run.py")

    def test_manifest_rejects_symlink_outside_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            outside = root / "outside.py"
            outside.write_text("SECRET = True\n", "utf-8")
            try:
                (repo / "outside.py").symlink_to(outside)
            except OSError:
                self.skipTest("当前 Windows 配置不允许创建符号链接")

            with self.assertRaisesRegex(ValueError, "仓库外"):
                repository_manifest(repo)


if __name__ == "__main__":
    unittest.main()
