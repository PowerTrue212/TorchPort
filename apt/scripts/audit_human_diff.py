#!/usr/bin/env python3
"""Audit the real APT PyTorch -> Jittor human-migration diff.

The classification is intentionally manual at the hunk level.  The script only
recomputes hunk boundaries and counts, verifies that every hunk has exactly one
label, and emits reviewable CSV/JSON/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import io
import json
import re
import tokenize
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)


CATEGORIES = {
    "API": {
        "name": "直接 API 映射",
        "definition": "可由局部、近一一对应的名称或调用形式替换完成，不需要重构周边算法。",
    },
    "REIMPLEMENT": {
        "name": "缺失依赖/能力重实现",
        "definition": "Jittor 侧没有原依赖或等价能力，需要新增组件、兼容层或完整加载逻辑。",
    },
    "SEMANTIC": {
        "name": "框架语义/执行适配",
        "definition": "因执行、梯度、张量写入、参数状态或调度语义不同而重写逻辑。",
    },
    "ENGINEERING": {
        "name": "工程、数据、权重与协议适配",
        "definition": "涉及数据增强、checkpoint、运行配置、功能范围、设备和工程健壮性。",
    },
    "NONFUNCTIONAL": {
        "name": "非功能性改动",
        "definition": "仅注释、空白、格式或输出字符串变化，不计入功能改动主比例。",
    },
}


# Each tuple is (category, scope_only, audit note).  Hunk numbering is produced
# with difflib.unified_diff(..., n=3), equivalent to a standard three-context
# unified diff.  A hard count check below prevents silent drift.
LABELS: dict[str, list[tuple[str, bool, str]]] = {
    "dataloaders/__init__.py": [
        ("ENGINEERING", True, "只再导出 CIFAR-10/100，属于支持范围收窄。"),
    ],
    "dataloaders/dataloader.py": [
        ("API", False, "torch Dataset/import 改为 Jittor 对应入口。"),
        ("SEMANTIC", False, "补充 Jittor Dataset 批处理状态字段。"),
        ("ENGINEERING", True, "删除未迁移的 ImageNet-R 等数据集类。"),
    ],
    "dataloaders/utils.py": [
        ("API", False, "torchvision transforms import 改为 jittor.transform。"),
        ("ENGINEERING", True, "数据集统计表收窄为 CIFAR-100。"),
        ("ENGINEERING", False, "AutoAugment 改为 ColorJitter，并重写归一化与数据分支。"),
        ("API", False, "six urllib 改为标准库 urllib.request。"),
        ("ENGINEERING", False, "增加下载文件已存在时的保护逻辑。"),
    ],
    "experiments/cifar-100.sh": [
        ("ENGINEERING", False, "修改 GPU 号与覆盖运行开关。"),
    ],
    "learners/default.py": [
        ("API", False, "框架、nn、functional 和 Optimizer import 映射。"),
        ("SEMANTIC", False, "CrossEntropyLoss 的 reduction 语义不同。"),
        ("SEMANTIC", False, "增加同步并调整训练循环运行行为；其余多为注释清理。"),
        ("NONFUNCTIONAL", False, "只清理注释/空白。"),
        ("NONFUNCTIONAL", False, "只清理解释性注释。"),
        ("SEMANTIC", False, "反向传播三步改为 optimizer.step(loss)，并改用 execute。"),
        ("API", False, "no_grad 与 forward/execute 的局部映射。"),
        ("SEMANTIC", False, "重写模型 train/eval 状态恢复，并调整本地测试调用。"),
        ("ENGINEERING", False, "模型保存格式变化，并按 shape 过滤 checkpoint 参数。"),
        ("API", False, "torch.load 改为 jt.load。"),
        ("API", False, "torch.load 改为 jt.load。"),
        ("NONFUNCTIONAL", False, "只增加空白。"),
        ("API", False, "optimizer 与 scheduler 命名空间映射。"),
        ("API", False, "forward 方法及调用改为 execute。"),
        ("API", False, "CUDA 设备和 DataParallel 入口映射。"),
        ("SEMANTIC", False, "第二处优化器反向传播生命周期适配。"),
    ],
    "learners/prompt.py": [
        ("API", False, "框架 import 映射；附带迁移说明。"),
        ("SEMANTIC", False, "原地 logits 屏蔽改为 jt.where，并适配 optimizer.step(loss)。"),
        ("SEMANTIC", False, "从单/多卡分支改为按参数名筛选，并增加失败检查。"),
        ("SEMANTIC", False, "优化器映射同时移除 PyTorch 式 set_device/DataParallel 分支。"),
    ],
    "models/__init__.py": [
        ("REIMPLEMENT", False, "接入新增的 timm 兼容模块 jimm。"),
    ],
    "models/vit.py": [
        ("REIMPLEMENT", False, "用本地 jimm 替代 timm；同时完成框架 import 映射。"),
        ("API", False, "MLP forward 改为 execute。"),
        ("SEMANTIC", False, "Attention 的张量切片原地写入改为 concat 重建张量。"),
        ("API", False, "Block forward 改为 execute。"),
        ("API", False, "Parameter 初始化、zeros 和 linspace 的局部映射。"),
        ("API", False, "移除 torch.jit 装饰并映射 execute/cat。"),
        ("API", False, "prompt.forward 与 torch.cat 的局部映射。"),
        ("API", False, "no_grad 装饰器映射。"),
        ("API", False, "from_numpy 映射。"),
        ("ENGINEERING", False, "禁用原位置编码 resize 分支，改变权重加载行为。"),
        ("API", False, "权重加载中的 cat 映射。"),
        ("API", False, "位置编码插值调用与 cat 映射。"),
    ],
    "models/zoo.py": [
        ("REIMPLEMENT", False, "移除 timm/MoCo 依赖入口，接入 jimm 和 Jittor 模块。"),
        ("API", False, "prompt buffer 的 zeros 映射。"),
        ("API", False, "prompt execute 与 concat/zeros 的局部映射。"),
        ("REIMPLEMENT", False, "新增位置编码 resize、完整 checkpoint 匹配/加载和 ViT 创建逻辑。"),
        ("ENGINEERING", False, "改用本地转换权重并调整分类头加载。"),
        ("ENGINEERING", False, "改写余弦分类头归一化/偏置，同时删除未迁移的 MoCo 路径。"),
    ],
    "run.py": [
        ("NONFUNCTIONAL", False, "只增加文件迁移说明。"),
        ("API", False, "torch import 改为 Jittor。"),
        ("SEMANTIC", False, "两个 PyTorch seed API 改为 Jittor 全局 seed。"),
        ("SEMANTIC", False, "显式启用 Jittor CUDA，并移除 cuDNN deterministic 设置。"),
        ("NONFUNCTIONAL", False, "只调整空格。"),
        ("NONFUNCTIONAL", False, "只修正输出字符串转义。"),
        ("NONFUNCTIONAL", False, "只修正输出字符串转义/空白。"),
    ],
    "trainer.py": [
        ("API", False, "框架与 DataLoader import 映射；附带迁移说明。"),
        ("ENGINEERING", True, "删除除 CIFAR 外的数据集选择分支。"),
        ("NONFUNCTIONAL", False, "只调整行尾空白。"),
        ("NONFUNCTIONAL", False, "只改注释文本。"),
    ],
    "utils/metric.py": [
        ("API", False, "no_grad、topk、equal/broadcast 的局部 API 映射。"),
    ],
    "utils/schedulers.py": [
        ("API", False, "Optimizer import 映射。"),
        ("SEMANTIC", False, "Jittor optimizer 参数组状态结构不同，重取 lr 字段。"),
        ("SEMANTIC", False, "增加 K<=1 的调度边界处理。"),
    ],
    "models/jimm.py": [
        ("REIMPLEMENT", False, "新增 161 行 timm 替代层：PatchEmbed、DropPath、trunc_normal、adapt_input_conv 等。"),
    ],
}


HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def hunks_for(old_lines: list[str], new_lines: list[str]) -> list[tuple[str, list[str]]]:
    diff = difflib.unified_diff(old_lines, new_lines, n=3)
    result: list[tuple[str, list[str]]] = []
    header: str | None = None
    body: list[str] = []
    for line in diff:
        if line.startswith("@@"):
            if header is not None:
                result.append((header, body))
            header, body = line, []
        elif header is not None:
            body.append(line)
    if header is not None:
        result.append((header, body))
    return result


def changed_excerpt(body: list[str], limit: int = 240) -> str:
    parts: list[str] = []
    for line in body:
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        text = line[1:].strip()
        if not text or text.startswith("#") or text in parts:
            continue
        parts.append(text)
        if len(" | ".join(parts)) >= limit:
            break
    return " | ".join(parts)[:limit]


def code_line_numbers(text: str) -> set[int]:
    """Return non-blank, non-comment, non-docstring physical line numbers."""
    lines = text.splitlines()
    result = {index for index, line in enumerate(lines, 1) if line.strip()}
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                prefix = lines[token.start[0] - 1][: token.start[1]]
                if not prefix.strip():
                    result.discard(token.start[0])
    except (IndentationError, tokenize.TokenError):
        pass

    try:
        tree = ast.parse(text)
        containers = [tree] + [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        for container in containers:
            if not container.body:
                continue
            first = container.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                for line_number in range(first.lineno, first.end_lineno + 1):
                    result.discard(line_number)
    except SyntaxError:
        pass
    return result


def code_change_counts(
    body: list[str], old_start: int, new_start: int, old_code: set[int], new_code: set[int]
) -> tuple[int, int]:
    old_line, new_line = old_start, new_start
    additions = deletions = 0
    for line in body:
        if line.startswith("-") and not line.startswith("---"):
            deletions += int(old_line in old_code)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            additions += int(new_line in new_code)
            new_line += 1
        elif line.startswith(" "):
            old_line += 1
            new_line += 1
        elif line.startswith("\\"):
            continue
    return additions, deletions


def pct(value: int, total: int) -> float:
    return round(100.0 * value / total, 1) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(r"D:\大学\新芽\apt"))
    parser.add_argument("--target", type=Path, default=Path(r"D:\大学\新芽\APT-Jittor"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for relative, labels in LABELS.items():
        old_path, new_path = args.source / relative, args.target / relative
        old_lines = old_path.read_text(encoding="utf-8", errors="replace").splitlines() if old_path.exists() else []
        new_lines = new_path.read_text(encoding="utf-8", errors="replace").splitlines()
        old_text = "\n".join(old_lines) + ("\n" if old_lines else "")
        new_text = "\n".join(new_lines) + ("\n" if new_lines else "")
        old_code = code_line_numbers(old_text)
        new_code = code_line_numbers(new_text)
        hunks = hunks_for(old_lines, new_lines)
        if len(hunks) != len(labels):
            raise RuntimeError(f"Hunk drift in {relative}: expected {len(labels)}, found {len(hunks)}")

        for index, ((header, body), (category, scope_only, note)) in enumerate(zip(hunks, labels), 1):
            match = HUNK_RE.search(header)
            if not match:
                raise RuntimeError(f"Cannot parse hunk header: {header}")
            additions = sum(line.startswith("+") and not line.startswith("+++") for line in body)
            deletions = sum(line.startswith("-") and not line.startswith("---") for line in body)
            code_additions, code_deletions = code_change_counts(
                body, int(match.group(1)), int(match.group(3)), old_code, new_code
            )
            rows.append(
                {
                    "hunk_id": f"{relative}#H{index:02d}",
                    "file": relative,
                    "hunk": index,
                    "old_start": int(match.group(1)),
                    "new_start": int(match.group(3)),
                    "additions": additions,
                    "deletions": deletions,
                    "changed_lines": additions + deletions,
                    "code_additions": code_additions,
                    "code_deletions": code_deletions,
                    "code_changed_lines": code_additions + code_deletions,
                    "category_code": category,
                    "category": CATEGORIES[category]["name"],
                    "scope_only": scope_only,
                    "audit_note": note,
                    "excerpt": changed_excerpt(body),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "apt_human_diff_hunks.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    functional = [row for row in rows if row["category_code"] != "NONFUNCTIONAL"]
    functional_no_scope = [row for row in functional if not row["scope_only"]]

    def aggregate(items: list[dict[str, object]]) -> dict[str, object]:
        grouped: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "hunks": 0,
                "additions": 0,
                "deletions": 0,
                "changed_lines": 0,
                "code_additions": 0,
                "code_deletions": 0,
                "code_changed_lines": 0,
            }
        )
        for row in items:
            bucket = grouped[str(row["category_code"])]
            bucket["hunks"] += 1
            bucket["additions"] += int(row["additions"])
            bucket["deletions"] += int(row["deletions"])
            bucket["changed_lines"] += int(row["changed_lines"])
            bucket["code_additions"] += int(row["code_additions"])
            bucket["code_deletions"] += int(row["code_deletions"])
            bucket["code_changed_lines"] += int(row["code_changed_lines"])
        totals = {
            "hunks": len(items),
            "additions": sum(int(row["additions"]) for row in items),
            "deletions": sum(int(row["deletions"]) for row in items),
            "changed_lines": sum(int(row["changed_lines"]) for row in items),
            "code_additions": sum(int(row["code_additions"]) for row in items),
            "code_deletions": sum(int(row["code_deletions"]) for row in items),
            "code_changed_lines": sum(int(row["code_changed_lines"]) for row in items),
        }
        for code, bucket in grouped.items():
            bucket["hunk_percent"] = pct(bucket["hunks"], totals["hunks"])
            bucket["addition_percent"] = pct(bucket["additions"], totals["additions"])
            bucket["changed_line_percent"] = pct(bucket["changed_lines"], totals["changed_lines"])
            bucket["code_addition_percent"] = pct(bucket["code_additions"], totals["code_additions"])
            bucket["code_changed_line_percent"] = pct(bucket["code_changed_lines"], totals["code_changed_lines"])
            bucket["name"] = CATEGORIES[code]["name"]
        return {"totals": totals, "categories": dict(grouped)}

    summary = {
        "method": {
            "unit": "three-context unified-diff hunk",
            "classification": "manual, mutually exclusive, dominant purpose of each hunk",
            "source": str(args.source),
            "target": str(args.target),
            "nonfunctional_policy": "excluded from headline functional proportions",
            "scope_sensitivity": "four pure scope-reduction hunks are excluded in the sensitivity view",
        },
        "all_hunks": aggregate(rows),
        "functional_hunks": aggregate(functional),
        "functional_hunks_excluding_scope_only": aggregate(functional_no_scope),
    }
    json_path = args.output_dir / "apt_human_diff_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    main_view = summary["functional_hunks"]
    sensitivity = summary["functional_hunks_excluding_scope_only"]
    main_total = main_view["totals"]["changed_lines"]
    api_lines = main_view["categories"]["API"]["changed_lines"]
    api_add = main_view["categories"]["API"]["additions"]
    total_add = main_view["totals"]["additions"]
    code_total = main_view["totals"]["code_changed_lines"]
    api_code = main_view["categories"]["API"]["code_changed_lines"]
    code_add_total = main_view["totals"]["code_additions"]
    api_code_add = main_view["categories"]["API"]["code_additions"]

    md = [
        "# APT 人工迁移 diff 分类审计",
        "",
        "## 结论",
        "",
        f"在排除 9 个纯注释/格式 hunk 后，共有 **{main_view['totals']['hunks']} 个功能性 hunk、{main_total} 行增删**。",
        f"其中直接 API 映射为 **{api_lines} 行（{pct(api_lines, main_total)}%）**；其余 **{main_total-api_lines} 行（{pct(main_total-api_lines, main_total)}%）**涉及重实现、框架语义或工程适配。",
        f"只看新增行，直接 API 映射为 **{api_add}/{total_add} 行（{pct(api_add, total_add)}%）**。",
        f"进一步排除空白、纯注释和 docstring 后，直接 API 映射占 **{api_code}/{code_total} 个代码物理行（{pct(api_code, code_total)}%）**；只看代码新增行为 **{api_code_add}/{code_add_total}（{pct(api_code_add, code_add_total)}%）**。",
        "",
        "按 hunk 个数看，直接 API 映射接近一半；按行数看却不到六分之一。这说明项目里有许多短小的名称替换，但工作量主要集中在少数较大的重实现和适配块。",
        "",
        "## 主统计",
        "",
        "| 类别 | Hunk 数 | Hunk 占比 | 新增 | 删除 | 增删行 | 行数占比 | 代码行占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["API", "REIMPLEMENT", "SEMANTIC", "ENGINEERING"]
    for code in order:
        item = main_view["categories"][code]
        md.append(
            f"| {item['name']} | {item['hunks']} | {item['hunk_percent']}% | "
            f"{item['additions']} | {item['deletions']} | {item['changed_lines']} | {item['changed_line_percent']}% | {item['code_changed_line_percent']}% |"
        )
    md += [
        f"| **合计** | **{main_view['totals']['hunks']}** | **100%** | **{main_view['totals']['additions']}** | **{main_view['totals']['deletions']}** | **{main_total}** | **100%** | **100%** |",
        "",
        "## 敏感性检查：排除纯功能范围删减",
        "",
        "四个 hunk 只是把未迁移的 ImageNet-R、DomainNet、CUB 等数据集分支从目标目录中删去。它们会放大“工程适配”的删除行数，因此单独排除后再计算：",
        "",
        f"- 剩余 {sensitivity['totals']['hunks']} 个功能性 hunk、{sensitivity['totals']['changed_lines']} 行增删；",
        f"- 直接 API 映射仍只有 {sensitivity['categories']['API']['changed_lines']} 行，占 {sensitivity['categories']['API']['changed_line_percent']}%；",
        f"- 非 API 修改仍占 {round(100-sensitivity['categories']['API']['changed_line_percent'],1)}%。",
        "",
        "因此“非 API 修改占主体”并不是由大段删除未支持数据集这一项单独造成的。",
        "",
        "## 口径",
        "",
        "- 比较范围：13 个同名已修改 Python/Shell 文件，加上新增 `models/jimm.py`。",
        "- hunk：Python `difflib.unified_diff(..., n=3)` 产生的三行上下文连续差异块。",
        "- 每个 hunk 按主导目的人工标一个互斥类别；混合 hunk 不重复计数。",
        "- 纯注释、空白和输出字符串格式共 9 个 hunk、33 行，排除在主比例之外。",
        "- 未复制到目标目录的 17 个配置/数据集文件不计入主统计；共同文件内部的四个纯范围删减 hunk保留在主统计，并另做排除敏感性检查。",
        "- 行数是新增行 + 删除行，衡量 diff 体量，不等于人工分钟数或认知难度。",
        "- `代码行` 进一步排除了空白、纯注释和 AST 可识别的模块/类/函数 docstring；它是防止迁移说明文字放大比例的稳健性口径。",
        "",
        "## 关键大块证据",
        "",
        "- `models/jimm.py`：新增 161 行，用 Jittor 重实现 timm 的 PatchEmbed、DropPath、trunc_normal、adapt_input_conv 等。",
        "- `models/zoo.py#H04`：+168/-15 行，新增位置编码 resize、checkpoint shape 匹配、权重装载和 ViT 创建逻辑。",
        "- `models/vit.py#H03`：Attention 原地切片写入改为 concat 重建张量。",
        "- `learners/prompt.py#H02`：旧类 logits 原地屏蔽改为 `jt.where`，并重写 optimizer 更新流程。",
        "- `utils/schedulers.py#H02`：Jittor optimizer 参数组状态结构不同，不能机械复用 PyTorch 读取方式。",
        "",
        "逐 hunk 的分类、起始行、增删数、理由与代码摘录见 `apt_human_diff_hunks.csv`；机器可读汇总见 `apt_human_diff_summary.json`。",
        "",
    ]
    md_path = args.output_dir / "apt_human_diff_audit.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "report": str(md_path), **summary["functional_hunks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
