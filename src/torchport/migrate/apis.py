from __future__ import annotations

import ast
import re

from .edit import Edit, dotted
from .rules import JT_OPTIMS, NN_ATTRS, PORT_OPTIMS, TORCH_ATTRS

BLOCKED_APIS = (
    "autograd",
    "distributed",
    "fx",
    "jit",
    "library",
    "onnx",
    "ops",
    "profiler",
    "utils.data",
    "utils.cpp_extension",
)

FACTORY_APIS = {
    "arange",
    "empty",
    "eye",
    "full",
    "linspace",
    "ones",
    "rand",
    "randn",
    "zeros",
}


class ApiRules:
    def visit_Name(self, node: ast.Name) -> None:
        text = self.torch_renames.get(node.id)
        if text:
            self._add_edit(node, text, "symbols.torch-alias")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = dotted(node)
        root = name.split(".", maxsplit=1)[0] if name else None
        if root in self.torch_aliases:
            suffix = name[len(root) + 1 :] if name and "." in name else ""
            if suffix == "jit.ignore":
                self._add_edit(node, "portlib.identity", "compat.jit-ignore")
                return
            for blocked in BLOCKED_APIS:
                if suffix in {"utils", "utils.data", "utils.data.DataLoader"}:
                    break
                if suffix == blocked or suffix.startswith(f"{blocked}."):
                    key = (node.lineno, blocked)
                    if key not in self.seen_blocks:
                        self.seen_blocks.add(key)
                        self._add_finding(
                            node,
                            code="native.unsupported-torch-api",
                            level="blocker",
                            symbol=name,
                            message="该 PyTorch 运行能力不能通过通用规则迁移。",
                            fix="使用 JTorch 路线，或增加专用后端实现。",
                        )
                    break
            known_accel = {
                "accelerator.is_available",
                "accelerator.current_accelerator",
            }
            if suffix.startswith("accelerator.") and suffix not in known_accel:
                key = (node.lineno, "accelerator")
                if key not in self.seen_blocks:
                    self.seen_blocks.add(key)
                    self._add_finding(
                        node,
                        code="native.unsupported-torch-api",
                        level="blocker",
                        symbol=name,
                        message="该 accelerator API 尚未映射到 Jittor 全局后端。",
                        fix="将设备选择统一到 TORCHPORT_DEVICE。",
                    )
            text = TORCH_ATTRS.get(node.attr)
            if text:
                self._attr_edit(node, text, f"api.torch.{node.attr}")

        parent = dotted(node.value)
        if (
            node.attr == "data"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr in {"bias", "weight"}
        ):
            self._add_edit(node, self.code.segment(node.value), "tensor.param-data-alias")
            return
        if parent in self.numpy_aliases and node.attr == "float":
            self._add_edit(node, "float", "compat.numpy-float")
            return
        if parent in self.nn_aliases or (parent and parent.endswith(".nn")):
            if node.attr == "init":
                self._add_edit(node, "portlib.init", "compat.nn-init")
                self.generic_visit(node)
                return
            if node.attr == "LayerNorm":
                self._add_edit(node, "portlib.LayerNorm", "compat.layer-norm")
                return
            if node.attr == "AdaptiveAvgPool1d":
                self._add_edit(
                    node,
                    "portlib.AdaptiveAvgPool1d",
                    "compat.adaptive-pool",
                )
                return
            text = NN_ATTRS.get(node.attr)
            if text and text != node.attr:
                self._attr_edit(node, text, f"api.nn.{node.attr}")
        if parent in self.optim_aliases and node.attr[:1].isupper():
            if node.attr in PORT_OPTIMS:
                self._add_edit(node, node.attr, f"compat.optim-{node.attr.lower()}")
            elif node.attr not in JT_OPTIMS:
                self._add_finding(
                    node,
                    code="native.unsupported-optimizer",
                    level="blocker",
                    symbol=name,
                    message="当前 Jittor 没有该优化器的等价实现。",
                    fix="实现 Jittor Optimizer，或改用已支持的优化器。",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted(node.func)
        root = name.split(".", maxsplit=1)[0] if name else None
        suffix = name[len(root) + 1 :] if root and name and "." in name else ""

        if root in self.torch_aliases and suffix == "utils.data.DataLoader":
            start, end = self.code.span(node.func)
            self.edits.append(Edit(start, end, "portlib.DataLoader", "compat.data-loader"))
        elif root in self.nn_aliases and suffix == "ReLU":
            self._drop_keyword(node, "inplace", "api.nn.relu-inplace")
        elif root in self.nn_aliases and suffix == "avg_pool2d" and self._global_pool(node):
            value = self.code.segment(node.args[0])
            self._add_edit(node, f"portlib.global_avg_pool2d({value})", "compat.global-avg-pool")
        elif isinstance(node.func, ast.Name) and node.func.id in self.var_aliases and node.args:
            self._unwrap_call(node, "compat.autograd-variable")
        elif root in self.torch_aliases and suffix in {
            "cuda.is_available",
            "accelerator.is_available",
        }:
            self._add_edit(node, "bool(jt.flags.use_cuda)", "device.global-backend-query")
        elif root in self.torch_aliases and suffix == "accelerator.current_accelerator":
            self._add_edit(
                node,
                '("cuda" if bool(jt.flags.use_cuda) else "cpu")',
                "device.global-backend-current",
            )
        elif (
            root in self.torch_aliases
            and suffix == "device"
            and len(node.args) == 1
            and not node.keywords
        ):
            self._unwrap_call(node, "device.descriptor-to-string")
        elif root in self.torch_aliases and suffix == "cuda.manual_seed_all" and node.args:
            start, end = self.code.span(node.func)
            self.edits.append(Edit(start, end, "jt.set_global_seed", "random.cuda-global-seed"))
        elif root in self.torch_aliases and suffix == "argsort":
            start, end = self.code.span(node.func)
            self.edits.append(Edit(start, end, "portlib.argsort", "compat.argsort"))
        elif root in self.torch_aliases and suffix in FACTORY_APIS:
            self._drop_keyword(node, "device", "device.factory-keyword")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in {"cuda", "cpu"}:
            if not node.args and not node.keywords:
                self._drop_call(node, "device.global-backend-transfer")
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            if self._device_move(node):
                self._drop_call(node, "device.global-backend-transfer")
            else:
                self._add_finding(
                    node,
                    code="native.tensor-to-overload",
                    level="blocker",
                    symbol="Tensor.to",
                    message="无法静态判断 .to() 是设备迁移还是类型转换。",
                    fix="设备使用全局后端，类型使用显式转换方法。",
                )
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "view_as"
            and len(node.args) == 1
            and not node.keywords
        ):
            self._attr_edit(node.func, "reshape", "tensor.view-as-to-reshape")
            _, arg_end = self.code.span(node.args[0])
            self.edits.append(Edit(arg_end, arg_end, ".shape", "tensor.view-as-to-reshape"))
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "view"
            and node.args
            and not node.keywords
        ):
            self._attr_edit(node.func, "reshape", "tensor.view-to-reshape")
        elif self._reshape_kw(node):
            func = self.code.segment(node.func)
            value = self.code.segment(node.keywords[0].value)
            self._add_edit(node, f"{func}({value})", "tensor.reshape-shape-keyword")
        elif self._data_copy(node):
            self._data_update(node)

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "backward"
            and node.lineno not in self.training_lines
        ):
            self._add_finding(
                node,
                code="native.noncanonical-backward",
                level="blocker",
                symbol="Tensor.backward",
                message="未识别出可转换的标准优化器训练步骤。",
                fix="整理为 zero_grad/backward/step 模式后重试。",
            )

        if name and name.endswith("nn.Parameter") and node.args:
            if len(node.args) == 1 and not node.keywords:
                self._unwrap_call(node, "parameters.native-var")
            elif len(node.args) == 1 and self._requires_grad(node) is False:
                value = self._expr(self.code.segment(node.args[0]))
                self._add_edit(node, f"{value}.stop_grad()", "parameters.stop-grad")
            else:
                self._add_finding(
                    node,
                    code="native.parameter-constructor",
                    level="blocker",
                    symbol=name,
                    message="nn.Parameter 带有额外参数，不能直接展开为 Jittor Var。",
                    fix="显式创建 Jittor Var，并验证梯度语义。",
                )

        if name and name.endswith("register_buffer"):
            if self._register_buffer(node):
                owner = self.code.segment(node.func.value)
                key = self.code.segment(node.args[0])
                value = self._expr(self.code.segment(node.args[1]))
                text = f"setattr({owner}, {key}, {value} if {value} is None else {value}.stop_grad())"
                self._add_edit(node, text, "module.register-buffer")
            else:
                self._add_finding(
                    node,
                    code="native.module-registration",
                    level="blocker",
                    symbol=name,
                    message="buffer 注册语义需要结合 state_dict 验证。",
                    fix="生成显式 Jittor Var 属性并验证权重名称。",
                )

        if name and name.endswith("register_parameter"):
            self._add_finding(
                node,
                code="native.module-registration",
                level="blocker",
                symbol=name,
                message="参数注册语义需要结合 state_dict 验证。",
                fix="生成显式 Jittor Var 属性并验证权重名称。",
            )

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and dotted(node.args[0]) in self.torch_aliases
        ):
            self._add_finding(
                node,
                code="native.dynamic-torch-lookup",
                level="blocker",
                symbol="getattr(torch, ...)",
                message="动态 PyTorch API 无法静态确定迁移规则。",
                fix="把动态 API 改为显式映射表。",
            )
        self.generic_visit(node)

    @staticmethod
    def _requires_grad(node: ast.Call) -> bool | None:
        values = [item.value for item in node.keywords if item.arg == "requires_grad"]
        if len(values) != 1 or len(values) != len(node.keywords):
            return None
        value = values[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            return value.value
        return None

    def _data_update(self, node: ast.Call) -> None:
        func = node.func
        assert isinstance(func, ast.Attribute)
        data = func.value
        assert isinstance(data, ast.Attribute)
        start, _ = self.code.span(data)
        _, end = self.code.span(func)
        base = self.code.segment(data.value)
        self.edits.append(Edit(start, end, f"{base}.update", "tensor.data-copy-update"))

    @staticmethod
    def _data_copy(node: ast.Call) -> bool:
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "copy_"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "data"
        )

    @staticmethod
    def _reshape_kw(node: ast.Call) -> bool:
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "reshape"
            and not node.args
            and len(node.keywords) == 1
            and node.keywords[0].arg == "shape"
        )

    @staticmethod
    def _register_buffer(node: ast.Call) -> bool:
        return (
            isinstance(node.func, ast.Attribute)
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and all(keyword.arg == "persistent" for keyword in node.keywords)
        )

    def _expr(self, text: str) -> str:
        for old, new in self.torch_renames.items():
            if text == old or text.startswith(f"{old}."):
                return f"{new}{text[len(old):]}"
        return text

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        is_module = any(
            (name := dotted(base)) is not None
            and (
                name in self.module_aliases
                or name == "nn.Module"
                or name.endswith((".nn.Module", ".Module"))
            )
            for base in node.bases
        )
        if is_module:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "forward"
                ):
                    self._rename_forward(item)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and re.search(r"\btorch(?:\.[A-Za-z_]\w*)+", node.value):
            self._add_finding(
                node,
                code="native.string-encoded-torch-symbol",
                level="warning",
                symbol=node.value[:120],
                message="字符串包含 PyTorch 符号，可能用于动态导入。",
                fix="确认该字符串是否参与 importlib、注册表或配置解析。",
            )
