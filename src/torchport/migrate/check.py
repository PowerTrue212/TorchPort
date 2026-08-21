from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from torchport.probe.envvals import apply_env
from torchport.probe.paths import module_name


class RepoCheck:
    def check(
        self,
        repository: str | Path,
        *,
        run: bool = False,
        runtime: bool = True,
        timeout: int = 120,
        python_executable: str | Path | None = None,
    ) -> dict[str, Any]:
        root = Path(repository).expanduser().resolve()
        meta = root / ".torchport"
        manifest = json.loads((meta / "manifest.json").read_text(encoding="utf-8"))
        syntax = self._syntax(root, manifest)
        residue = self._imports(root, manifest)
        run_result = self._runtime(
            root, manifest, run, timeout, python_executable
        ) if runtime else {
            "attempted": False,
            "passed": True,
        }
        blockers = manifest["summary"]["blockers"]
        passed = syntax["passed"] and residue["passed"] and run_result["passed"] and blockers == 0
        report = {
            "schema_version": 1,
            "kind": "torchport.repository-check",
            "status": "passed" if passed else "failed",
            "passed": passed,
            "repository": str(root),
            "target": manifest["target"],
            "manifest_status": manifest["status"],
            "manifest_blockers": blockers,
            "syntax": syntax,
            "residue": residue,
            "runtime": run_result,
        }
        (meta / "check.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _syntax(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        errors = []
        paths = RepoCheck._files(manifest)
        paths.append(".torchport/run.py")
        for rel_path in paths:
            path = root / rel_path
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
            except SyntaxError as exc:
                errors.append({"file": rel_path, "line": exc.lineno, "message": exc.msg})
        return {"passed": not errors, "files": len(paths), "errors": errors}

    @staticmethod
    def _imports(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        if manifest["target"] != "jittor-native":
            return {"passed": True, "imports": []}
        residue = []
        roots = {"torch", "torchvision", "torchaudio", "torchtext"}
        for rel_path in RepoCheck._files(manifest):
            tree = ast.parse((root / rel_path).read_text(encoding="utf-8"), filename=rel_path)
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                for module in modules:
                    root_name = module.split(".", maxsplit=1)[0]
                    if root_name in roots and not RepoCheck._local_module(root, module):
                        residue.append(
                            {"file": rel_path, "line": node.lineno, "module": module}
                        )
        return {"passed": not residue, "imports": residue}

    @staticmethod
    def _files(manifest: dict[str, Any]) -> list[str]:
        scope = manifest.get("scope") or {}
        files = scope.get("files") or []
        if scope.get("mode") == "model" and files:
            return list(files)
        return [item["path"] for item in manifest["files"]]

    @staticmethod
    def _local_module(root: Path, module: str) -> bool:
        parts = module.split(".")
        for base in (root / "src", root):
            path = base.joinpath(*parts)
            if path.with_suffix(".py").is_file() or (path / "__init__.py").is_file():
                return True
        return False

    @staticmethod
    def _runtime(
        root: Path,
        manifest: dict[str, Any],
        run: bool,
        timeout: int,
        python_executable: str | Path | None,
    ) -> dict[str, Any]:
        entrypoint = manifest.get("entrypoint")
        entry = manifest.get("probe", {}).get("entry") or entrypoint
        if not entry:
            return {"attempted": False, "passed": False, "error": "没有入口文件"}
        python = str(python_executable or sys.executable)
        if run and entry == entrypoint:
            command = [python, str(root / ".torchport" / "run.py")]
            mode = "run"
        elif name := module_name(root / entry, root):
            code = "import importlib,sys; importlib.import_module(sys.argv[1])"
            command = [python, "-c", code, name]
            mode = "import" if not run else "probe-import"
        else:
            code = "import runpy,sys; runpy.run_path(sys.argv[1], run_name='torchport_smoke')"
            command = [python, "-c", code, str(root / entry)]
            mode = "import" if not run else "probe-import"
        env = os.environ.copy()
        apply_env(env, manifest.get("probe", {}).get("env", {}))
        py_paths = [str(root / "src"), str(root)] if (root / "src").is_dir() else [str(root)]
        for item in env.get("PYTHONPATH", "").split(os.pathsep):
            if not item:
                continue
            path = Path(item)
            py_paths.append(str(path if path.is_absolute() else Path.cwd() / path))
        env["PYTHONPATH"] = os.pathsep.join(py_paths)
        if manifest["target"] == "jittor-native":
            env["nvcc_path"] = ""
        try:
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "attempted": True,
                "mode": mode,
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "attempted": True,
                "mode": mode,
                "passed": False,
                "timeout": timeout,
                "stdout": (exc.stdout or "")[-4000:],
                "stderr": (exc.stderr or "")[-4000:],
            }
