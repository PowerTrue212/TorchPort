"""Write small-magnitude v2 negative controls from the controller training reference.

Each control copies the reference directory and applies exactly one textual edit; the
edit must match once, otherwise the script fails. The learning rate reported through
`param_groups` is never changed, so the exact lr-schedule check cannot catch these
controls and only the numerical envelope is tested.

Not included: scaling gradients by a constant. Adam is invariant to it up to eps, so it
is a behaviour-preserving change, not a negative control.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BENCH = Path("D:/科研/SimCSE-Benchmark-v1")
REFERENCE = BENCH / "controller/train-reference-v2/SimCSE"
SKIPPED = "bert.encoder.layer.11.attention.output.dense.bias"

CONTROLS = {
    # 有效学习率 +1%，上报的 lr 不变
    "lr-hidden-1p01": [('            lr = group["lr"]\n',
                        '            lr = group["lr"] * 1.01\n')],
    # 温度 +1%
    "temp-0p0505": [("        self.temperature = 0.05\n",
                     "        self.temperature = 0.0505\n")],
    # Adam epsilon 1e-8 -> 1e-6：只影响梯度接近零的元素
    "eps-1e-6": [("        self.eps = 1e-8\n", "        self.eps = 1e-6\n")],
    # 二阶矩衰减 0.999 -> 0.998
    "beta2-0p998": [("        self.beta2 = 0.999\n", "        self.beta2 = 0.998\n")],
    # 单个 768 元素参数不更新（Adam 状态照常累积）
    "skip-one-bias": [(
        "def train_step(model, state, input_ids, attention_mask, token_type_ids):\n"
        "    loss = model(input_ids, attention_mask, token_type_ids)\n"
        "    state.step(loss)\n"
        "    return loss\n",
        "def train_step(model, state, input_ids, attention_mask, token_type_ids):\n"
        f"    frozen = dict(model.named_parameters())[{SKIPPED!r}]\n"
        "    snapshot = frozen.numpy().copy()\n"
        "    loss = model(input_ids, attention_mask, token_type_ids)\n"
        "    state.step(loss)\n"
        "    frozen.update(jt.array(snapshot))\n"
        "    return loss\n",
    )],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = (REFERENCE / "run.py").read_text("utf-8")
    manifest = {}
    for name, edits in CONTROLS.items():
        text = source
        for old, new in edits:
            if text.count(old) != 1:
                raise ValueError(f"{name}: edit does not match exactly once")
            text = text.replace(old, new)
        target = args.output / name
        shutil.copytree(REFERENCE, target, ignore=shutil.ignore_patterns("__pycache__"))
        (target / "run.py").write_text(text, "utf-8")
        compile(text, str(target / "run.py"), "exec")
        manifest[name] = [{"old": old, "new": new} for old, new in edits]
    (args.output / "manifest.json").write_text(
        json.dumps({"reference": str(REFERENCE), "controls": manifest}, indent=2,
                   ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(sorted(manifest)))


if __name__ == "__main__":
    main()
