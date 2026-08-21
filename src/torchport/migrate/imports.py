from __future__ import annotations

import ast

from .rules import BLOCKED_MODS, JT_OPTIMS, PORT_OPTIMS, RISK_DEPS, TORCH_NAMES

TIMM_VIT = {"Attention", "Block", "Mlp", "PatchEmbed"}
TIMM_LAYERS = {"DropPath", "to_2tuple", "trunc_normal_"}
EINOPS = {"rearrange", "repeat"}
EINOPS_LAYERS = {"Rearrange"}


class ImportRules:
    def visit_Import(self, node: ast.Import) -> None:
        changed = False
        aliases: list[ast.alias] = []
        lines: list[str] = []
        for alias in node.names:
            if alias.name == "torch":
                aliases.append(ast.alias(name="jittor", asname=alias.asname or "jt"))
                changed = True
            elif alias.name in {"torch.nn", "torch.nn.functional"}:
                aliases.append(ast.alias(name="jittor.nn", asname=alias.asname))
                changed = True
            elif alias.name == "torch.nn.init":
                change = ast.ImportFrom(
                    module="portlib",
                    names=[ast.alias(name="init", asname=alias.asname)],
                    level=0,
                )
                lines.append(ast.unparse(change))
                changed = True
            elif alias.name == "torch.utils.checkpoint":
                aliases.append(ast.alias(name="portlib", asname=alias.asname or "checkpoint"))
                changed = True
            elif alias.name == "torch.optim":
                aliases.append(ast.alias(name="jittor.optim", asname=alias.asname))
                changed = True
            else:
                aliases.append(alias)
                if alias.name in RISK_DEPS and not self._local_import(alias.name):
                    self._add_finding(
                        node,
                        code="native.third-party-torch-extension",
                        level="blocker",
                        symbol=alias.name,
                        message="依赖 PyTorch 扩展生态，无法安全映射到原生 Jittor。",
                        fix="使用 JTorch 路线，或为该扩展实现 Jittor 适配器。",
                    )
        if changed:
            if aliases:
                lines.insert(0, ast.unparse(ast.Import(names=aliases)))
            if any(alias.name == "torch.optim" for alias in node.names) and self.port_optims:
                names = ", ".join(sorted(self.port_optims))
                lines.append(f"from portlib import {names}")
            if self.port_lib:
                lines.append("import portlib")
            rule = "compat.nn-init" if any(
                alias.name == "torch.nn.init" for alias in node.names
            ) else "compat.checkpoint" if any(
                alias.name == "torch.utils.checkpoint" for alias in node.names
            ) else "imports.torch-to-jittor"
            self._add_edit(node, "\n".join(lines), rule)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if self._local_import(module):
            return
        if module == "torch.nn.init":
            known = {
                "calculate_gain",
                "constant_",
                "kaiming_normal_",
                "kaiming_uniform_",
                "normal_",
                "orthogonal_",
                "xavier_uniform_",
            }
            names = [alias for alias in node.names if alias.name in known]
            unknown = [alias for alias in node.names if alias.name not in known]
            for alias in unknown:
                self._add_finding(
                    node,
                    code="native.unsupported-init",
                    level="blocker",
                    symbol=f"torch.nn.init.{alias.name}",
                    message="该初始化函数尚未有 Jittor 兼容实现。",
                    fix="在 portlib 中补充等价初始化函数并重新验证。",
                )
            if names:
                change = ast.ImportFrom(module="portlib", names=names, level=0)
                self._add_edit(node, ast.unparse(change), "compat.nn-init")
            return
        if module == "timm.models.vision_transformer":
            names = {alias.name for alias in node.names}
            if names <= TIMM_VIT:
                change = ast.ImportFrom(module="portlib", names=node.names, level=node.level)
                self._add_edit(node, ast.unparse(change), "compat.timm-vit")
                return
            for alias in node.names:
                if alias.name not in TIMM_VIT:
                    self._add_finding(
                        node,
                        code="native.unsupported-timm-vit",
                        level="blocker",
                        symbol=f"{module}.{alias.name}",
                        message="该 timm ViT 组件尚未有 Jittor 兼容实现。",
                        fix="由 DSH 判断是否新增通用组件映射，或保留源组件并使用 JTorch 路线。",
                    )
            return
        if module == "timm.models.layers":
            names = {alias.name for alias in node.names}
            if names <= TIMM_LAYERS:
                change = ast.ImportFrom(module="portlib", names=node.names, level=node.level)
                self._add_edit(node, ast.unparse(change), "compat.timm-layers")
                return
            for alias in node.names:
                if alias.name not in TIMM_LAYERS:
                    self._add_finding(
                        node,
                        code="native.unsupported-timm-layer",
                        level="blocker",
                        symbol=f"{module}.{alias.name}",
                        message="该 timm layers 组件尚未有 Jittor 兼容实现。",
                        fix="由 DSH 判断是否新增通用组件映射，或保留源组件并使用 JTorch 路线。",
                    )
            return
        if module == "einops":
            names = {alias.name for alias in node.names}
            if names <= EINOPS:
                change = ast.ImportFrom(module="portlib", names=node.names, level=node.level)
                self._add_edit(node, ast.unparse(change), "compat.einops")
                return
            for alias in node.names:
                if alias.name not in EINOPS:
                    self._add_finding(
                        node,
                        code="native.unsupported-einops",
                        level="blocker",
                        symbol=f"{module}.{alias.name}",
                        message="该 einops API 尚未有 Jittor 兼容实现。",
                        fix="由 DSH 根据实际 pattern 补充 portlib 兼容函数并重新验证。",
                    )
            return
        if module == "einops.layers.torch":
            names = {alias.name for alias in node.names}
            if names <= EINOPS_LAYERS:
                change = ast.ImportFrom(module="portlib", names=node.names, level=node.level)
                self._add_edit(node, ast.unparse(change), "compat.einops")
                return
            for alias in node.names:
                if alias.name not in EINOPS_LAYERS:
                    self._add_finding(
                        node,
                        code="native.unsupported-einops",
                        level="blocker",
                        symbol=f"{module}.{alias.name}",
                        message="该 einops torch layer 尚未有 Jittor 兼容实现。",
                        fix="由 DSH 根据实际 pattern 补充 portlib layer 并重新验证。",
                    )
            return
        if module == "torch.autograd":
            names = {alias.name for alias in node.names}
            if names <= {"Variable"}:
                self._add_edit(node, "", "compat.autograd-variable")
                return
        if module == "torch.utils.data":
            names = {alias.name for alias in node.names}
            if names <= {"DataLoader"}:
                change = ast.ImportFrom(module="portlib", names=node.names, level=node.level)
                self._add_edit(node, ast.unparse(change), "compat.data-loader")
                return
        if module == "torch.optim.lr_scheduler":
            names = {alias.name for alias in node.names}
            if names <= {"StepLR"}:
                change = ast.ImportFrom(module="portlib", names=node.names, level=node.level)
                self._add_edit(node, ast.unparse(change), "compat.step-lr")
                return
        if module == "torchvision":
            names = {alias.name for alias in node.names}
            if names <= {"datasets", "transforms"}:
                change = ast.ImportFrom(module="portlib", names=node.names, level=node.level)
                self._add_edit(node, ast.unparse(change), "compat.vision-data")
                return
        if any(module == item or module.startswith(f"{item}.") for item in BLOCKED_MODS):
            self._add_finding(
                node,
                code="native.unsupported-torch-submodule",
                level="blocker",
                symbol=module,
                message="该 PyTorch 子模块没有可证明等价的原生 Jittor 自动映射。",
                fix="改用 JTorch 路线，或为该子模块增加专用迁移规则。",
            )
            return
        if module.startswith("torch.utils.data"):
            self._add_finding(
                node,
                code="native.dataset-contract",
                level="blocker",
                symbol=module,
                message="PyTorch DataLoader 与 Jittor Dataset 的批处理契约不同。",
                fix="生成 Jittor Dataset 适配器并验证采样顺序、批大小和随机种子。",
            )
            return
        if module in RISK_DEPS or any(module.startswith(f"{item}.") for item in RISK_DEPS):
            self._add_finding(
                node,
                code="native.third-party-torch-extension",
                level="blocker",
                symbol=module,
                message="依赖 PyTorch 扩展生态，无法安全映射到原生 Jittor。",
                fix="使用 JTorch 路线，或为该扩展实现 Jittor 适配器。",
            )
            return

        target = None
        aliases: list[ast.alias] = []
        if module == "torch":
            target = "jittor"
            for alias in node.names:
                name = TORCH_NAMES.get(alias.name, alias.name)
                local = alias.asname
                if name != alias.name and local is None:
                    local = alias.name
                aliases.append(ast.alias(name=name, asname=local))
        elif module == "torch.nn.functional":
            target = "jittor.nn"
            aliases = list(node.names)
        elif module == "torch.nn":
            port = [alias for alias in node.names if alias.name == "init"]
            layers = {
                "AdaptiveAvgPool1d": "compat.adaptive-pool",
                "LayerNorm": "compat.layer-norm",
            }
            layer = [alias for alias in node.names if alias.name in layers]
            native = [
                alias
                for alias in node.names
                if alias.name not in {"init", *layers}
            ]
            lines = []
            if native:
                if len(native) == 1 and native[0].name == "functional":
                    alias = native[0]
                    change = ast.ImportFrom(
                        module="jittor",
                        names=[ast.alias(name="nn", asname=alias.asname or "functional")],
                        level=node.level,
                    )
                else:
                    change = ast.ImportFrom(module="jittor.nn", names=native, level=node.level)
                lines.append(ast.unparse(change))
            if port:
                lines.append(
                    ast.unparse(ast.ImportFrom(module="portlib", names=port, level=0))
                )
            if layer:
                lines.append(
                    ast.unparse(ast.ImportFrom(module="portlib", names=layer, level=0))
                )
            if lines:
                rule = "imports.torch-to-jittor"
                if port:
                    rule = "compat.nn-init"
                if layer:
                    rule = layers[layer[0].name]
                self._add_edit(node, "\n".join(lines), rule)
                return
            target = "jittor.nn"
            aliases = list(node.names)
        elif module == "torch.optim":
            native = [alias for alias in node.names if alias.name in JT_OPTIMS]
            port = [alias for alias in node.names if alias.name in PORT_OPTIMS]
            unknown = [alias for alias in node.names if alias.name not in JT_OPTIMS | PORT_OPTIMS]
            lines = []
            if native:
                lines.append(ast.unparse(ast.ImportFrom(module="jittor.optim", names=native)))
            if port:
                lines.append(ast.unparse(ast.ImportFrom(module="portlib", names=port)))
            if unknown:
                for alias in unknown:
                    self._add_finding(
                        node,
                        code="native.unsupported-optimizer",
                        level="blocker",
                        symbol=f"torch.optim.{alias.name}",
                        message="当前 Jittor 没有该优化器的等价实现。",
                        fix="实现 Jittor Optimizer，或选择已支持的优化器并重新验证。",
                    )
            if lines:
                self._add_edit(node, "\n".join(lines), "imports.torch-to-jittor")
            return
        elif module.startswith("torch"):
            self._add_finding(
                node,
                code="native.unknown-torch-submodule",
                level="blocker",
                symbol=module,
                message="尚未收录该 PyTorch 子模块的原生 Jittor 映射。",
                fix="新增专用规则，或切换为 JTorch 兼容路线。",
            )
            return

        if target:
            change = ast.ImportFrom(module=target, names=aliases, level=node.level)
            lines = [ast.unparse(change)]
            if module == "torch" and self.port_lib:
                lines.append("import portlib")
            self._add_edit(node, "\n".join(lines), "imports.torch-to-jittor")
