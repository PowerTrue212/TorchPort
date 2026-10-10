"""Create agent workspaces workspaces/<repo>-<condition>-r<k>/.

Contents: copy of the PyTorch source, the PyTorch adapter as reference_torch_adapter.py,
a Jittor-side torchport_adapter.py, an empty submission.py skeleton, the rendered prompt
(PROMPT.md) and WEIGHTS.md (path reference only; no weights are copied).
No Jittor reference, oracle, hidden input or evaluator is copied.
Per-workspace task files go to workspaces/_tasks/ (outside every workspace).

Usage: python materialize.py [--out DIR] [--repos REPO ...] [--conds COND ...]
                             [--k 0] [--tokens 不限] [--turns 200] [--minutes 180]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
ROOT = KIT.parents[2]
HELD = ROOT / "heldout"

REPOS = {
    "birefnet": {
        "name": "BiRefNet-COD", "src": "source", "model": "TorchPortBiRefNet", "shape": "(1, 3, 1024, 1024)",
        "ckpt": ["BiRefNet-COD-epoch_125.pth"],
        "note": "官方配置 swin_v1_l；参数名以 BiRefNet(bb_pretrained=False).state_dict() 为准。",
    },
    "vscode-v2": {
        "name": "VSCode-v2-T stage2 (CODRGB)", "src": "source/VSCode2", "model": "TorchPortVSCodeV2", "shape": "(1, 3, 352, 352)",
        "ckpt": ["swinT.pth"],
        "note": "只迁移 CODRGB 推理路径，并返回一个张量（最终 COD logits）。checkpoint 含 64 个 CODRGB 推理未使用分支的键；不要构造这些分支，否则额外参数会导致验证失败。参数名必须与去掉 `module.` 前缀后的 788 个推理键完全一致。",
    },
    "plantcamo": {
        "name": "PlantCamo MyNet", "src": "source", "model": "TorchPortPlantCamo", "shape": "(1, 3, 704, 704)",
        "ckpt": ["Net_epoch_best.pth", "pvt_v2_b2.pth"],
        "note": "PyTorch 源码硬编码读取 ./pretrained_pvt/pvt_v2_b2.pth，Jittor 实现里不要读取任何权重文件，由评测方加载完整 checkpoint。",
    },
}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".idea", "*.pth", "*.png", "*.jpg")

SKELETON = '''"""Agent-owned {name} migration contract.

Implement ``load_model`` and return a Jittor module whose call accepts one NCHW float32
image {shape} and returns the same tuple structure as reference_torch_adapter.py.
Parameter names must equal the PyTorch state_dict keys. TorchPort loads the frozen
PyTorch state into the returned module before inference.
"""

from __future__ import annotations

import jittor as jt


def load_model() -> jt.Module:
    raise NotImplementedError("implement the native Jittor model")
'''

JT_ADAPTER = '''"""Stable TorchPort entry around the Agent-owned submission."""

from __future__ import annotations

import jittor as jt
from submission import load_model


class {model}(jt.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = load_model()

    def execute(self, image: jt.Var) -> tuple[jt.Var, ...]:
        return tuple(self.model(image))
'''


def render_prompt(cond: str, repo: str, args) -> str:
    cfg = REPOS[repo]
    text = (KIT / "prompts" / f"{cond}.md").read_text(encoding="utf-8")
    values = {
        "{{REPO_NAME}}": cfg["name"], "{{INPUT_SHAPE}}": cfg["shape"], "{{ADAPTER_NOTE}}": cfg["note"],
        "{{BUDGET_TOKENS}}": str(args.tokens), "{{MAX_TURNS}}": str(args.turns), "{{WALL_MINUTES}}": str(args.minutes),
        "{{WORKSPACE}}": f"{repo}-{cond}-r{args.k}",
    }
    for key, value in values.items():
        text = text.replace(key, value)
    assert "{{" not in text, "unfilled placeholder"
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE, help="workspace parent directory")
    ap.add_argument("--repos", nargs="+", choices=tuple(REPOS), default=tuple(REPOS))
    ap.add_argument("--conds", nargs="+", choices=("none", "full"), default=("none", "full"))
    ap.add_argument("--k", type=int, default=0)
    # 预算（用户 2026-10-09 确定）：不限 token，200 轮，180 分钟；none/full 相同。
    ap.add_argument("--tokens", default="不限")
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--minutes", type=int, default=180)
    # 只重写已有工作区的 PROMPT.md（Agent 尚未开始时用于更新提示词），不动代码与任务文件。
    ap.add_argument("--refresh-prompts", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (HERE / "_tasks").mkdir(exist_ok=True)
    for repo in args.repos:
        cfg = REPOS[repo]
        for cond in args.conds:
            ws = out / f"{repo}-{cond}-r{args.k}"
            if ws.exists():
                if args.refresh_prompts:
                    (ws / "PROMPT.md").write_text(render_prompt(cond, repo, args), encoding="utf-8")
                    print("refreshed prompt", ws.name)
                else:
                    print(f"skip existing {ws.name}")
                continue
            shutil.copytree(HELD / repo / cfg["src"], ws, ignore=IGNORE)
            shutil.copy2(HELD / repo / "adapter" / "torchport_adapter.py", ws / "reference_torch_adapter.py")
            (ws / "torchport_adapter.py").write_text(JT_ADAPTER.format(model=cfg["model"]), encoding="utf-8")
            (ws / "submission.py").write_text(SKELETON.format(name=cfg["name"], shape=cfg["shape"]), encoding="utf-8")
            (ws / "PROMPT.md").write_text(render_prompt(cond, repo, args), encoding="utf-8")
            weights = "\n".join(f"- {(HELD / repo / 'checkpoint' / c)}" for c in cfg["ckpt"])
            (ws / "WEIGHTS.md").write_text(
                "# 权重位置（仅路径引用，未复制）\n\n" + weights + "\n\n这些文件位于工作区之外，评测方负责加载。\n", encoding="utf-8")
            task = json.loads((KIT / "tasks" / f"{repo}.json").read_text(encoding="utf-8"))
            task["target"]["repo"] = str(ws)
            task["task_id"] += f"-{cond}-r{args.k}"
            (HERE / "_tasks" / f"{ws.name}.json").write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
            print("created", ws.name)


if __name__ == "__main__":
    main()
