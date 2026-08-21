from __future__ import annotations

from .models import MigrationPlan, MigrationTarget, ProjectInspection, SourceFramework


class MigrationPlanner:
    def create(
        self,
        inspection: ProjectInspection,
        target: MigrationTarget,
    ) -> MigrationPlan:
        source = inspection.primary_framework
        blockers = list(inspection.risks)

        if source is SourceFramework.PYTORCH and target is MigrationTarget.JTORCH:
            route = "PyTorch → JTorch 兼容运行 → 数值验证"
        elif source is SourceFramework.PYTORCH:
            route = "PyTorch → 原生 Jittor → 数值验证"
        elif source in {SourceFramework.TENSORFLOW, SourceFramework.CAFFE, SourceFramework.ONNX}:
            if target is MigrationTarget.JTORCH:
                blockers.append("JTorch 只适用于 PyTorch 兼容路线。")
            route = f"{source.value} → 统一迁移表示 → 原生 Jittor → 数值验证"
        elif source is SourceFramework.JITTOR:
            route = "现有 Jittor 项目 → 运行验证与硬件认证"
        else:
            route = "未知源框架 → 人工确认 → 统一迁移表示 → Jittor"

        if target is MigrationTarget.JITTOR_ASCEND:
            route += " → 昇腾后端验证"

        steps = (
            "固定源仓库版本、依赖和运行入口。",
            "在源框架采集相同输入下的参考输出和中间结果。",
            "建立模块、算子、权重和源代码位置映射。",
            "执行确定性迁移规则并生成可审查修改。",
            "运行目标项目并定位首次数值分歧点。",
            "应用经过验证的修复规则，重新执行数值对齐。",
            "生成环境、数值、性能和依赖残留报告。",
        )
        evidence = (
            "源仓库提交号和依赖锁定信息",
            "输入、权重和随机种子哈希",
            "运行环境与硬件信息",
            "最终输出和分层数值对齐结果",
            "全部自动修改及其验证记录",
        )

        return MigrationPlan(
            source_framework=source,
            target=target,
            route=route,
            steps=steps,
            blockers=tuple(blockers),
            required_evidence=evidence,
        )

