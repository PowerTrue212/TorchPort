from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from .advise import build_advice
from .fail import classify
from .migrate import RepoCheck, RepositoryMigrator
from .migrate.rules import COPY_SKIP
from .migrate.slice import prune_imports
from .models import MigrationTarget
from .probe import AlignRun


class AutoRun:
    def __init__(
        self,
        migrator: RepositoryMigrator | None = None,
        checker: RepoCheck | None = None,
        aligner: AlignRun | None = None,
    ) -> None:
        self.migrator = migrator or RepositoryMigrator()
        self.checker = checker or RepoCheck()
        self.aligner = aligner or AlignRun()

    def run(
        self,
        repository: str | Path,
        output: str | Path,
        *,
        target: MigrationTarget = MigrationTarget.JTORCH,
        entrypoint: str | None = None,
        env: dict[str, str] | None = None,
        model: str | None = None,
        inputs: list[dict[str, Any]] | None = None,
        shapes: tuple[tuple[int, ...], ...] | None = None,
        dtypes: tuple[str, ...] | None = None,
        model_args: list[Any] | None = None,
        model_kw: dict[str, Any] | None = None,
        check_run: bool = False,
        runtime: bool = True,
        timeout: int = 120,
        align: bool = True,
        run_root: str | Path | None = None,
        weights: str | Path | None = None,
        seed: int = 7,
        source_python: str | Path | None = None,
        target_python: str | Path | None = None,
        source_path: Iterable[str | Path] = (),
        target_path: Iterable[str | Path] = (),
        try_hints: bool = False,
        agent: dict[str, Any] | None = None,
        scope: str = "repo",
    ) -> dict[str, Any]:
        source = Path(repository).expanduser().resolve()
        target_dir = Path(output).expanduser().resolve()
        migration = self.migrator.migrate(
            source,
            target_dir,
            target=target,
            entrypoint=entrypoint,
            env=env,
            model=model,
            inputs=inputs,
            shapes=shapes,
            dtypes=dtypes,
            model_args=model_args,
            model_kw=model_kw,
            scope=scope,
        )
        check = self.checker.check(
            target_dir,
            run=check_run,
            runtime=runtime,
            timeout=timeout,
            python_executable=target_python,
        )
        root = Path(run_root).expanduser().resolve() if run_root else target_dir / ".torchport" / "align"
        certificate, skip, error = self._align(
            source,
            target_dir,
            migration,
            check,
            align,
            run_root=root,
            weights=weights,
            seed=seed,
            source_python=source_python,
            target_python=target_python,
            source_path=source_path,
            target_path=target_path,
        )
        align_info = {
            "requested": align,
            "run_root": str(root),
            "skipped": skip,
            "error": error,
        }
        entry_cert = migration.get("entry_cert")
        failure = classify(migration, check, certificate, align_info)
        report = {
            "schema_version": 1,
            "kind": "torchport.auto-run",
            "status": self._status(migration, check, certificate, align, error),
            "passed": self._passed(check, certificate, align),
            "source": str(source),
            "output": str(target_dir),
            "target": target.value,
            "migration": migration,
            "check": check,
            "entry_cert": entry_cert,
            "certificate": certificate,
            "align": align_info,
            "failure": failure,
            "agent": agent,
            "advice": build_advice(migration, check, certificate, align_info),
            "artifacts": {"report": str(target_dir / ".torchport" / "auto.json"), "run_root": str(root)},
        }
        path = target_dir / ".torchport" / "auto.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if try_hints and (config := self._hint_config(report, model, inputs, shapes, model_args, model_kw)):
            retry_dir = self._next_path(target_dir, "hints")
            retry_root = self._next_path(Path(run_root).expanduser().resolve(), "hints") if run_root else None
            retry = self.run(
                source,
                retry_dir,
                target=target,
                entrypoint=entrypoint,
                env=env,
                model=config.get("model", model),
                inputs=config.get("inputs", inputs),
                shapes=None if config.get("inputs") else shapes,
                dtypes=dtypes,
                model_args=model_args,
                model_kw=config.get("model_kw", model_kw),
                check_run=check_run,
                runtime=runtime,
                timeout=timeout,
                align=align,
                run_root=retry_root,
                weights=weights,
                seed=seed,
                source_python=source_python,
                target_python=target_python,
                source_path=source_path,
                target_path=target_path,
                agent=agent,
                scope=scope,
            )
            retry["retry"] = {
                "used": True,
                "source": report["output"],
                "config": config,
            }
            retry["attempts"] = [self._attempt(report, 1), self._attempt(retry, 2)]
            self._save(retry)
            return retry
        report["retry"] = {"used": False}
        report["attempts"] = [self._attempt(report, 1)]
        self._save(report)
        return report

    def _align(
        self,
        source: Path,
        target: Path,
        migration: dict[str, Any],
        check: dict[str, Any],
        requested: bool,
        *,
        run_root: Path,
        weights: str | Path | None,
        seed: int,
        source_python: str | Path | None,
        target_python: str | Path | None,
        source_path: Iterable[str | Path],
        target_path: Iterable[str | Path],
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        if not requested:
            return None, "align-disabled", None
        if migration["status"] != "converted":
            return None, "migration-not-converted", None
        if not check["passed"]:
            return None, "check-failed", None
        if migration["probe"].get("status") != "ready":
            return None, "probe-not-ready", None
        source = self._source_scope(source, run_root, migration)
        try:
            return (
                self.aligner.align(
                    source,
                    target,
                    run_root=run_root,
                    weights=weights,
                    seed=seed,
                    source_python=source_python or sys.executable,
                    target_python=target_python or sys.executable,
                    source_path=source_path,
                    target_path=target_path,
                ),
                None,
                None,
            )
        except ValueError as exc:
            return None, None, str(exc)

    @staticmethod
    def _source_scope(source: Path, run_root: Path, migration: dict[str, Any]) -> Path:
        if (migration.get("scope") or {}).get("mode") != "model":
            return source
        run_root.mkdir(parents=True, exist_ok=True)
        target = run_root / "source_scope"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns(*COPY_SKIP),
            symlinks=True,
        )
        prune_imports(target, migration.get("probe") or {})
        return target

    @staticmethod
    def _status(
        migration: dict[str, Any],
        check: dict[str, Any],
        certificate: dict[str, Any] | None,
        align: bool,
        error: str | None,
    ) -> str:
        if certificate is not None:
            return "verified" if certificate["passed"] else "failed"
        if error is not None:
            return "failed"
        if migration["status"] != "converted":
            return "partial"
        if not check["passed"]:
            return "failed"
        return "checked"

    @staticmethod
    def _passed(
        check: dict[str, Any],
        certificate: dict[str, Any] | None,
        align: bool,
    ) -> bool:
        if certificate is not None:
            return bool(certificate["passed"])
        return bool(check["passed"] and not align)

    @staticmethod
    def _hint_config(
        report: dict[str, Any],
        model: str | None,
        inputs: list[dict[str, Any]] | None,
        shapes: tuple[tuple[int, ...], ...] | None,
        model_args: list[Any] | None,
        model_kw: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if report["passed"]:
            return None
        actions = {item["id"]: item for item in report["advice"]["actions"]}
        if any(item["stage"] == "migration" and item["id"] == "set-entry" for item in actions.values()):
            return None
        config: dict[str, Any] = {}
        if model is None:
            if "set-model" in actions:
                config["model"] = actions["set-model"]["example"]
            elif report["migration"]["probe"].get("model"):
                config["model"] = report["migration"]["probe"]["model"]
        if model_args is None and model_kw is None and "set-args" in actions:
            value = json.loads(actions["set-args"]["example"])
            if any(item == "..." for item in value.values()):
                return None
            config["model_kw"] = value
        if inputs is None and shapes is None and "set-input" in actions:
            hinted = actions["set-input"].get("hints")
            if not hinted:
                return None
            config["inputs"] = hinted
        if any(
            item["stage"] == "migration"
            and item["id"] not in {"set-model", "set-args", "set-input"}
            for item in actions.values()
        ):
            return None
        return config or None

    @staticmethod
    def _next_path(path: Path, suffix: str) -> Path:
        candidate = path.with_name(f"{path.name}-{suffix}")
        if not candidate.exists():
            return candidate
        for index in range(2, 100):
            candidate = path.with_name(f"{path.name}-{suffix}{index}")
            if not candidate.exists():
                return candidate
        raise ValueError(f"无法选择重跑目录：{path}")

    @staticmethod
    def _attempt(report: dict[str, Any], index: int) -> dict[str, Any]:
        result = {
            "index": index,
            "status": report["status"],
            "passed": report["passed"],
            "output": report["output"],
        }
        if report.get("agent") is not None:
            result["agent"] = report["agent"]
        return result

    @staticmethod
    def _save(report: dict[str, Any]) -> None:
        path = Path(report["artifacts"]["report"])
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
