"""Materialize paired LoRA GPT-2 Small none/full candidate workspaces.

The two workspaces copy the same frozen Microsoft source and receive the same
empty Jittor entry and prompt. Only the directory name and controller feedback differ.
Third-party Jittor references are never copied into either workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MIGRATION = ROOT / "migrations" / "LoRA"
SOURCE = MIGRATION / "source" / "microsoft-LoRA"
DEFAULT_OUT = MIGRATION / "workspaces"
UPSTREAM_REVISION = "c4593f060e6a368d7bb5af5273b8e42810cdef90"
BASE_CHECKPOINT_SHA256 = "7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421"
CONDITIONS = {"none": "diagnostic", "full": "localized"}

IGNORE = shutil.ignore_patterns(
    ".git",
    ".idea",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "pretrained_checkpoints",
    "trained_models",
)

RUN_SKELETON = '''"""Agent-owned stable entry for GPT-2 Small + LoRA on Jittor."""
from __future__ import annotations

from collections.abc import Mapping

import jittor as jt


def load_model() -> jt.Module:
    raise NotImplementedError("construct the migrated GPT-2 Small + LoRA model")


def mark_only_lora_as_trainable(model: jt.Module) -> None:
    raise NotImplementedError("freeze every parameter whose name does not contain lora_")


def lora_state_dict(model: jt.Module) -> dict[str, jt.Var]:
    raise NotImplementedError("return only lora_ parameters")


def load_lora_state_dict(model: jt.Module, state: Mapping[str, object]) -> None:
    raise NotImplementedError("strictly load the LoRA-only state")


def build_optimizer(model: jt.Module):
    raise NotImplementedError("return the source-equivalent AdamW and linear warmup scheduler")


def train_step(model: jt.Module, state, input_ids, labels, mask):
    raise NotImplementedError("perform one optimizer and scheduler update, returning mean loss")
'''

ADAPTER = '''"""Stable evaluation entry. Keep its interface unchanged."""
from __future__ import annotations

import jittor as jt

from run import build_optimizer, load_model, train_step


class LoRAGPT2SmallEval(jt.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = load_model()

    def execute(self, input_ids: jt.Var) -> tuple[jt.Var, ...]:
        output = self.model(input_ids)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        return (logits,)
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if not item.is_file() or ".git" in item.parts or "__pycache__" in item.parts:
            continue
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def render_prompt(*, tokens: str, turns: int, minutes: int) -> str:
    body = (HERE / "prompts" / "body.md").read_text(encoding="utf-8")
    values = {
        "{{BUDGET_TOKENS}}": tokens,
        "{{MAX_TURNS}}": str(turns),
        "{{WALL_MINUTES}}": str(minutes),
    }
    for key, value in values.items():
        body = body.replace(key, value)
    if "{{" in body:
        raise RuntimeError("unfilled prompt placeholder")
    return body


def initialize_git(path: Path) -> str:
    commands = (
        ("init", "--quiet"),
        ("add", "-A"),
        (
            "-c",
            "user.name=Benchmark Seed",
            "-c",
            "user.email=benchmark-seed@invalid",
            "commit",
            "--quiet",
            "-m",
            "Initial repository candidate",
        ),
    )
    for command in commands:
        completed = subprocess.run(
            ["git", "-C", str(path), *command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "failed to initialize candidate Git")
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def remove_readonly(function, path: str, _error) -> None:
    """Make a Git object writable before retrying a Windows tree removal."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--tokens", default="不限")
    parser.add_argument("--turns", type=int, default=200)
    parser.add_argument("--minutes", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not SOURCE.is_dir():
        raise FileNotFoundError(f"missing frozen source: {SOURCE}")
    checkpoint = MIGRATION / "checkpoint" / "gpt2-pytorch_model.bin"
    if not checkpoint.is_file() or sha256_file(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("GPT-2 Small/base checkpoint is missing or has the wrong SHA-256")
    fixture_manifest_path = HERE / "fixtures" / "manifest.json"
    if not fixture_manifest_path.is_file():
        raise FileNotFoundError("missing fixtures/manifest.json; run fixtures/make_fixtures.py")
    fixtures = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    states: list[dict[str, object]] = []
    code_digests: set[str] = set()
    for condition, feedback in CONDITIONS.items():
        name = f"lora-gpt2-small-{condition}-r{args.replicate}"
        workspace = (out / name).resolve()
        workspace.relative_to(out)
        if workspace == out:
            raise RuntimeError("refusing to replace the workspace root")
        if workspace.exists():
            if not args.force:
                raise FileExistsError(f"workspace already exists: {workspace}; use --force to rebuild")
            shutil.rmtree(workspace, onexc=remove_readonly)
        shutil.copytree(SOURCE, workspace, ignore=IGNORE)
        (workspace / "run.py").write_text(RUN_SKELETON, encoding="utf-8")
        (workspace / "evaluation_entry.py").write_text(ADAPTER, encoding="utf-8")
        prompt = render_prompt(tokens=args.tokens, turns=args.turns, minutes=args.minutes)
        (workspace / "PROMPT.md").write_text(prompt, encoding="utf-8")

        candidate_digest = tree_digest(workspace)
        code_digests.add(candidate_digest)
        git_commit = initialize_git(workspace)
        states.append(
            {
                "condition": condition,
                "feedback": feedback,
                "workspace": str(workspace),
                "git_commit": git_commit,
                "initial_candidate_sha256": candidate_digest,
                "prompt_sha256": sha256_file(workspace / "PROMPT.md"),
            }
        )

    if len(code_digests) != 1:
        raise RuntimeError("paired workspaces do not have identical initial candidates")

    prepared = {
        "schema_version": 1,
        "kind": "torchport.p2-prepared-state",
        "experiment_id": f"lora-gpt2-small-diagnostic-vs-localized-r{args.replicate}",
        "prepared_at": date.today().isoformat(),
        "status": "pilot_prepared_not_formal",
        "isolation": {
            "formal_claims_allowed": False,
            "local_workspaces_are_host_visible": True,
            "isolation_claim": "open-tool pilot; no host filesystem sandbox claim",
            "public_research_enabled": True,
        },
        "source_revision": UPSTREAM_REVISION,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "initial_candidate_sha256": next(iter(code_digests)),
        "common": {
            "model_card": "gpt2.sm",
            "n_layer": 12,
            "n_embd": 768,
            "n_head": 12,
            "vocab_size": 50257,
            "lora_dim": 4,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "seed": 20261017,
            "wall_minutes": args.minutes,
            "max_turns": args.turns,
            "token_budget": args.tokens,
        },
        "conditions": states,
        "data": {
            "core_equivalence_requires_e2e_dataset": False,
            "raw_e2e_present_in_source": True,
            "frozen_token_batch": fixtures["input"],
            "deterministic_lora_state": fixtures["lora_state"],
            "formatted_jsonl_prepared": False,
            "task_metrics_require_e2e_dataset": True,
        },
        "missing_before_formal_run": [
            "freeze an open-tool benchmark protocol and its claim boundary",
            "record ordinary tool use and public references for both conditions",
            "freeze hidden verification, budgets, and any task-performance thresholds in scope",
        ],
    }
    state_path = HERE / ("prepared-state.json" if args.replicate == 0 else f"prepared-state-r{args.replicate}.json")
    state_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(prepared, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
