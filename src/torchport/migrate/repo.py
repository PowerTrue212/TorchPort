from __future__ import annotations

import ast
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from torchport.entry import build_entry
from torchport.models import MigrationTarget, ProjectInspection, SourceFramework
from torchport.planner import MigrationPlanner
from torchport.probe import infer_probe
from torchport.probe.merge import merge_probe
from torchport.probe.paths import module_name
from torchport.scanner import RepositoryScanner

from .deps import build_deps
from .edit import finding
from .jtorch import scan_jtorch
from .native import NativeEdit
from .report import next_cmds, render_readme, write_files
from .rules import COPY_SKIP
from .scope import model_scope
from .slice import prune_imports
from .status import probe_issue


class RepositoryMigrator:
    def __init__(self) -> None:
        self.scanner = RepositoryScanner()
        self.planner = MigrationPlanner()

    def migrate(
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
        scope: str = "repo",
    ) -> dict[str, Any]:
        source = Path(repository).expanduser().resolve()
        output = Path(output).expanduser().resolve()
        inspection = self.scanner.scan(source)
        if SourceFramework.PYTORCH not in inspection.frameworks:
            raise ValueError("仓库未检测到 PyTorch。")
        if output == source or source in output.parents:
            raise ValueError("输出目录不能位于源仓库内部。")
        if output.exists():
            raise ValueError(f"输出目录已经存在：{output}")
        if target not in {MigrationTarget.JTORCH, MigrationTarget.JITTOR_NATIVE}:
            raise ValueError(f"暂不支持目标：{target.value}")
        if scope not in {"repo", "model"}:
            raise ValueError(f"未知迁移作用域：{scope}")

        shutil.copytree(source, output, ignore=self._ignore, symlinks=True)
        plan = self.planner.create(inspection, target)
        findings = self._repo_findings(inspection, target)
        files: list[dict[str, Any]] = []
        local_modules = self._local_modules(source)
        for src_path in self._py_files(source):
            rel_path = src_path.relative_to(source).as_posix()
            result, issues = self._migrate_file(
                src_path,
                output / rel_path,
                rel_path,
                target,
                local_modules,
            )
            files.append(result)
            findings.extend(issues)

        deps, issues = build_deps(source, output, inspection.dependency_files, target)
        findings.extend(issues)
        entry = self._pick_entry(source, inspection.entrypoints, entrypoint)
        rules = {rule for item in files for rule in item["applied_rules"]}
        base_probe = infer_probe(source, entry, env)
        if entry and not base_probe.get("entry"):
            base_probe["entry"] = entry
        probe = merge_probe(
            base_probe,
            entry=entry if entrypoint is not None and model is not None else None,
            model=model,
            inputs=inputs,
            shapes=shapes,
            dtypes=dtypes,
            model_args=model_args,
            model_kw=model_kw,
        )
        issue = probe_issue(probe, entry, target)
        if issue:
            findings.append(issue)
        scope_files = model_scope(source, probe.get("entry")) if scope == "model" else set()
        slice_info = prune_imports(output, probe) if scope == "model" else {}
        if slice_info.get("rules"):
            rules.update(slice_info["rules"])
            self._mark_files(files, slice_info["files"], slice_info["rules"])
        findings = self._apply_scope(findings, scope, scope_files, slice_info)
        findings = self._dedupe(findings)
        weight_files = [
            path
            for path in inspection.weight_files
            if Path(path).suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
        ]
        if len(weight_files) == 1:
            probe["weights"] = weight_files[0]
        elif weight_files:
            probe["weight_candidates"] = weight_files
        generated = write_files(output, target, entry, rules, probe)
        blockers = sum(item["severity"] == "blocker" for item in findings)
        warnings = sum(item["severity"] == "warning" for item in findings)
        changed = sum(item["status"] == "transformed" for item in files)
        valid = sum(item["syntax_valid"] for item in files)

        inspection_data = inspection.to_dict()
        entry_cert = build_entry(source, entry, probe, inspection_data)
        report: dict[str, Any] = {
            "schema_version": 1,
            "kind": "torchport.repository-migration",
            "status": "converted" if blockers == 0 else "partial",
            "source": str(source),
            "output": str(output),
            "source_framework": "pytorch",
            "target": target.value,
            "scope": {
                "mode": scope,
                "files": sorted(scope_files),
                "lines": slice_info.get("lines", {}),
                "names": slice_info.get("names", []),
                "symbols": slice_info.get("symbols", {}),
            },
            "entrypoint": entry,
            "probe": probe,
            "entry_cert": entry_cert,
            "inspection": inspection_data,
            "plan": plan.to_dict(),
            "summary": {
                "python_files": len(files),
                "transformed_files": changed,
                "syntax_valid_files": valid,
                "blockers": blockers,
                "warnings": warnings,
                "automatic_rules": sorted(rules),
            },
            "files": files,
            "dependencies": deps,
            "findings": findings,
            "generated_files": generated,
            "next_commands": next_cmds(target, entry),
        }
        meta = output / ".torchport"
        manifest = meta / "manifest.json"
        report["manifest"] = str(manifest)
        manifest.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output / "TORCHPORT_MIGRATION.md").write_text(
            render_readme(report),
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _ignore(_path: str, names: list[str]) -> set[str]:
        return {name for name in names if name in COPY_SKIP}

    @staticmethod
    def _py_files(root: Path) -> Iterable[Path]:
        for directory, dir_names, file_names in os.walk(root, followlinks=False):
            dir_names[:] = sorted(name for name in dir_names if name not in COPY_SKIP)
            base = Path(directory)
            for file_name in sorted(file_names):
                path = base / file_name
                if path.suffix == ".py" and path.is_file() and not path.is_symlink():
                    yield path

    @staticmethod
    def _migrate_file(
        source: Path,
        output: Path,
        rel_path: str,
        target: MigrationTarget,
        local_modules: set[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        text = source.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(text, filename=rel_path)
        except SyntaxError as exc:
            issue = finding(
                code="repository.python-syntax",
                level="blocker",
                file=rel_path,
                line=exc.lineno,
                symbol=None,
                message=f"源文件无法解析：{exc.msg}",
                fix="确认 Python 版本或修复语法。",
            )
            return RepositoryMigrator._file_result(rel_path, "copied", (), False), [issue]

        if target is MigrationTarget.JTORCH:
            return RepositoryMigrator._file_result(rel_path, "copied", (), True), scan_jtorch(
                text, rel_path
            )
        try:
            rendered, rules, issues = NativeEdit(text, rel_path, local_modules).transform()
        except (SyntaxError, ValueError) as exc:
            issue = finding(
                code="native.transform-failed",
                level="blocker",
                file=rel_path,
                line=getattr(exc, "lineno", None),
                symbol=None,
                message=f"转换失败，文件保持原样：{exc}",
                fix="为该语法结构增加迁移规则和回归测试。",
            )
            return RepositoryMigrator._file_result(rel_path, "copied", (), True), [issue]
        status = "copied"
        if rendered != text:
            output.write_text(rendered, encoding="utf-8")
            status = "transformed"
        return RepositoryMigrator._file_result(rel_path, status, rules, True), issues

    @staticmethod
    def _local_modules(root: Path) -> set[str]:
        result = set()
        for path in RepositoryMigrator._py_files(root):
            name = module_name(path, root)
            if name:
                result.add(name)
        return result

    @staticmethod
    def _file_result(
        path: str,
        status: str,
        rules: Iterable[str],
        valid: bool,
    ) -> dict[str, Any]:
        return {
            "path": path,
            "status": status,
            "applied_rules": list(rules),
            "syntax_valid": valid,
        }

    @staticmethod
    def _repo_findings(
        inspection: ProjectInspection,
        target: MigrationTarget,
    ) -> list[dict[str, Any]]:
        level = "blocker" if target is MigrationTarget.JITTOR_NATIVE else "warning"
        return [
            finding(
                code="repository.custom-operator",
                level=level,
                file=path,
                line=None,
                symbol=None,
                message="检测到 C++、CUDA 或动态库。",
                fix="实现 Jittor 自定义算子并保留源端对齐测试。",
            )
            for path in inspection.custom_operator_files
        ]

    @staticmethod
    def _pick_entry(
        source: Path,
        entries: Iterable[str],
        requested: str | None,
    ) -> str | None:
        if requested is None:
            return next(iter(entries), None)
        path = Path(requested)
        if path.is_absolute() or ".." in path.parts or not (source / path).is_file():
            raise ValueError(f"入口文件不存在：{requested}")
        return path.as_posix()

    @staticmethod
    def _dedupe(findings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        result: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in findings:
            key = (item["id"], item["file"], item["line"], item["symbol"])
            result.setdefault(key, item)
        return list(result.values())

    @staticmethod
    def _apply_scope(
        findings: Iterable[dict[str, Any]],
        scope: str,
        files: set[str],
        slice_info: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if scope != "model":
            return list(findings)
        lines = {
            path: set(items)
            for path, items in (slice_info or {}).get("lines", {}).items()
        }
        symbols = {
            path: set(items)
            for path, items in (slice_info or {}).get("symbols", {}).items()
        }
        result = []
        for item in findings:
            scoped = dict(item)
            symbol = str(scoped.get("symbol") or "")
            if (
                scoped.get("severity") == "blocker"
                and not str(scoped.get("id", "")).startswith("probe.")
                and scoped.get("file") not in files
            ):
                scoped["severity"] = "warning"
                scoped["scoped"] = "outside-model"
            elif (
                scoped.get("severity") == "blocker"
                and (
                    scoped.get("line") in lines.get(str(scoped.get("file")), set())
                    or RepositoryMigrator._pruned(
                        symbol,
                        symbols.get(str(scoped.get("file")), set()),
                    )
                )
            ):
                scoped["severity"] = "warning"
                scoped["scoped"] = "outside-symbol"
            result.append(scoped)
        return result

    @staticmethod
    def _pruned(symbol: str, symbols: set[str]) -> bool:
        return any(symbol == item or symbol.startswith(f"{item}.") for item in symbols)

    @staticmethod
    def _mark_files(
        files: list[dict[str, Any]],
        paths: Iterable[str],
        rules: Iterable[str],
    ) -> None:
        selected = set(paths)
        extra = list(rules)
        for item in files:
            if item["path"] not in selected:
                continue
            item["status"] = "transformed"
            item["applied_rules"] = sorted({*item["applied_rules"], *extra})
