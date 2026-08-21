from __future__ import annotations

import ast
import re
from typing import Any

from .edit import CodeMap, Edit, dotted, finding
from .rules import PORT_OPTIMS


class BaseEdit(ast.NodeVisitor):
    def __init__(
        self,
        source: str,
        path: str,
        local_modules: set[str] | None = None,
    ) -> None:
        self.source = source
        self.path = path
        self.local_modules = local_modules or set()
        self.code = CodeMap(source)
        self.edits: list[Edit] = []
        self.findings: list[dict[str, Any]] = []
        self.torch_aliases: set[str] = set()
        self.torch_renames: dict[str, str] = {}
        self.nn_aliases: set[str] = set()
        self.module_aliases: set[str] = set()
        self.optim_aliases: set[str] = set()
        self.numpy_aliases: set[str] = set()
        self.var_aliases: set[str] = set()
        self.port_optims: set[str] = set()
        self.port_lib = False
        self.training_lines: set[int] = set()
        self.seen_blocks: set[tuple[int, str]] = set()

    def transform(self) -> tuple[str, tuple[str, ...], list[dict[str, Any]]]:
        tree = ast.parse(self.source, filename=self.path)
        self._scan_aliases(tree)
        self._train_edits(tree)
        self.visit(tree)
        rendered, rules = self._apply_edits()
        ast.parse(rendered, filename=self.path)
        return rendered, rules, self.findings

    def _add_edit(self, node: ast.AST, text: str, rule: str) -> None:
        start, end = self.code.span(node)
        self.edits.append(Edit(start, end, text, rule))

    def _attr_edit(self, node: ast.Attribute, text: str, rule: str) -> None:
        start, end = self.code.attr_span(node)
        self.edits.append(Edit(start, end, text, rule))

    def _drop_call(self, node: ast.Call, rule: str) -> None:
        func = node.func
        assert isinstance(func, ast.Attribute)
        _, value_end = self.code.span(func.value)
        _, call_end = self.code.span(node)
        self.edits.append(Edit(value_end, call_end, "", rule))

    def _unwrap_call(self, node: ast.Call, rule: str) -> None:
        call_start, call_end = self.code.span(node)
        arg_start, arg_end = self.code.span(node.args[0])
        self.edits.extend(
            [
                Edit(call_start, arg_start, "", rule),
                Edit(arg_end, call_end, "", rule),
            ]
        )

    def _drop_keyword(self, node: ast.Call, name: str, rule: str) -> None:
        keyword = next((item for item in node.keywords if item.arg == name), None)
        if keyword is None:
            return
        start, end = self.code.span(keyword)
        call_start, call_end = self.code.span(node)
        text = self.source
        right = end
        while right < call_end and text[right].isspace():
            right += 1
        if right < call_end and text[right] == ",":
            self.edits.append(Edit(start, right + 1, "", rule))
            return
        left = start
        while left > call_start and text[left - 1].isspace():
            left -= 1
        if left > call_start and text[left - 1] == ",":
            self.edits.append(Edit(left - 1, end, "", rule))
            return
        self.edits.append(Edit(start, end, "", rule))

    def _add_finding(
        self,
        node: ast.AST,
        *,
        code: str,
        level: str,
        symbol: str | None,
        message: str,
        fix: str,
    ) -> None:
        self.findings.append(
            finding(
                code=code,
                level=level,
                file=self.path,
                line=getattr(node, "lineno", None),
                symbol=symbol,
                message=message,
                fix=fix,
            )
        )

    def _local_import(self, module: str) -> bool:
        return module in self.local_modules

    def _scan_aliases(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch":
                        local = alias.asname or "torch"
                        self.torch_aliases.add(local)
                        if alias.asname is None:
                            self.torch_renames[local] = "jt"
                    elif alias.name in {"torch.nn", "torch.nn.functional"}:
                        self.nn_aliases.add(alias.asname or alias.name)
                    elif alias.name == "torch.optim":
                        self.optim_aliases.add(alias.asname or alias.name)
                    elif alias.name == "numpy":
                        self.numpy_aliases.add(alias.asname or "numpy")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "torch":
                    for alias in node.names:
                        if alias.name == "nn":
                            self.nn_aliases.add(alias.asname or "nn")
                        elif alias.name == "optim":
                            self.optim_aliases.add(alias.asname or "optim")
                elif node.module in {"torch.nn", "torch.nn.functional"}:
                    for alias in node.names:
                        if node.module == "torch.nn" and alias.name == "functional":
                            self.nn_aliases.add(alias.asname or "functional")
                        if node.module == "torch.nn" and alias.name == "Module":
                            self.module_aliases.add(alias.asname or alias.name)
                elif node.module == "torch.optim":
                    for alias in node.names:
                        self.optim_aliases.add(alias.asname or alias.name)
                elif node.module == "torch.autograd":
                    for alias in node.names:
                        if alias.name == "Variable":
                            self.var_aliases.add(alias.asname or alias.name)
                elif node.module in {"einops", "einops.layers.torch"}:
                    self.port_lib = True
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = dotted(node.func)
                root = name.split(".", maxsplit=1)[0] if name else None
                suffix = name[len(root) + 1 :] if root and name and "." in name else ""
                if root in self.nn_aliases and suffix == "avg_pool2d" and self._global_pool(node):
                    self.port_lib = True
                if root in self.nn_aliases and suffix == "LayerNorm":
                    self.port_lib = True
                if root in self.torch_aliases and suffix == "argsort":
                    self.port_lib = True
                if root in self.torch_aliases and suffix == "jit.ignore":
                    self.port_lib = True
            if not isinstance(node, ast.Attribute):
                continue
            if dotted(node.value) in self.optim_aliases and node.attr in PORT_OPTIMS:
                self.port_optims.add(node.attr)
            name = dotted(node)
            if name and name.endswith(".utils.data.DataLoader"):
                root = name.split(".", maxsplit=1)[0]
                self.port_lib |= root in self.torch_aliases
            if name and any(
                name == f"{alias}.init" or name.startswith(f"{alias}.init.")
                for alias in self.nn_aliases
            ):
                self.port_lib = True
            if name and any(name == f"{alias}.LayerNorm" for alias in self.nn_aliases):
                self.port_lib = True
            if name and any(name == f"{alias}.jit.ignore" for alias in self.torch_aliases):
                self.port_lib = True

    def _train_edits(self, tree: ast.AST) -> None:
        for owner in ast.walk(tree):
            body = getattr(owner, "body", None)
            if not isinstance(body, list):
                continue
            for index in range(1, len(body)):
                backward = self._method_stmt(body[index - 1], "backward")
                step = self._method_stmt(body[index], "step")
                if not backward or not step:
                    continue
                optim = step[0]
                zero_index = None
                for check in range(index - 2, -1, -1):
                    zero = self._method_stmt(body[check], "zero_grad")
                    if zero and zero[0] == optim:
                        zero_index = check
                        break
                    if self._method_stmt(body[check], "step"):
                        break
                if zero_index is None:
                    continue
                loss = backward[1]
                zero_start, zero_end = self.code.span(body[zero_index])
                back_start, _ = self.code.span(body[index - 1])
                _, step_end = self.code.span(body[index])
                self.edits.extend(
                    [
                        Edit(zero_start, zero_end, "", "training.remove-zero-grad"),
                        Edit(
                            back_start,
                            step_end,
                            f"{optim}.step({self.code.segment(loss)})",
                            "training.optimizer-step",
                        ),
                    ]
                )
                self.training_lines.add(body[index - 1].lineno)

    @staticmethod
    def _method_stmt(node: ast.AST, method: str) -> tuple[str, ast.AST] | None:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return None
        func = node.value.func
        if not isinstance(func, ast.Attribute) or func.attr != method:
            return None
        if method == "backward":
            return "", func.value
        if isinstance(func.value, ast.Name):
            return func.value.id, func.value
        return None

    @staticmethod
    def _device_move(node: ast.Call) -> bool:
        if len(node.args) != 1 or node.keywords:
            return False
        arg = node.args[0]
        if isinstance(arg, ast.Name):
            return "device" in arg.id.lower()
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value.startswith(("cpu", "cuda", "mps", "npu"))
        if isinstance(arg, ast.Call):
            name = dotted(arg.func)
            return bool(name and name.endswith(".device"))
        return False

    @classmethod
    def _global_pool(cls, node: ast.Call) -> bool:
        if len(node.args) < 2:
            return False
        base = node.args[0]
        kernel = node.args[1]
        if not isinstance(kernel, ast.Subscript):
            return False
        value = kernel.value
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            return cls._same(base, value.value)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            return (
                value.func.attr == "size"
                and not value.args
                and cls._same(base, value.func.value)
            )
        return False

    @staticmethod
    def _same(left: ast.AST, right: ast.AST) -> bool:
        return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)

    def _rename_forward(self, node: ast.AST) -> None:
        line = self.code.lines[node.lineno - 1]
        start = self.code.starts[node.lineno - 1]
        match = re.search(r"\bdef\s+forward\b", line)
        if match:
            name_start = start + match.start() + match.group(0).rfind("forward")
            self.edits.append(
                Edit(
                    name_start,
                    name_start + len("forward"),
                    "execute",
                    "module.forward-to-execute",
                )
            )

    def _apply_edits(self) -> tuple[str, tuple[str, ...]]:
        unique = {(item.start, item.end, item.text, item.rule): item for item in self.edits}
        ordered = sorted(unique.values(), key=lambda item: (item.start, -(item.end - item.start)))
        accepted: list[Edit] = []
        for item in ordered:
            if accepted and item.start < accepted[-1].end:
                if item.end <= accepted[-1].end:
                    continue
                raise ValueError(f"迁移规则交叉：{accepted[-1].rule} 与 {item.rule}")
            accepted.append(item)
        rendered = self.source
        for item in reversed(accepted):
            rendered = rendered[: item.start] + item.text + rendered[item.end :]
        return rendered, tuple(sorted({item.rule for item in accepted}))
