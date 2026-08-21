import { dirname, resolve } from "node:path";

import type { Context } from "@deepseek-ai/cordis";
import { defineTool } from "@deepseek-ai/dsh-tools";

import { nextStep, type JsonMap } from "./loop.js";
import { makeReport, readJson, saveReport } from "./report.js";
import { runTool, toolCwd, type RunResult } from "./trace.js";
import type { Config } from "./plugin.js";

const outputSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    ok: { type: "boolean", required: true },
    code: { type: "integer", required: true },
    command: { type: "array", items: { type: "string" }, required: true },
    data: { type: "json", required: true },
    stderr: { type: "string", required: true },
    trace: { type: "string", required: true },
  },
} as const;

const runParams = {
  repository: { type: "string", required: true, description: "源 PyTorch 仓库路径" },
  output: { type: "string", required: true, description: "新的隔离输出目录" },
  target: { type: "string", enum: ["jtorch", "jittor-native"] },
  entrypoint: { type: "string" },
  model: { type: "string" },
  shape: { type: "string", description: "例如 1,3,224,224" },
  dtype: { type: "string", enum: ["float32", "int64"] },
  input: { type: "array", items: { type: "json" } },
  modelArgs: { type: "array", items: { type: "json" } },
  modelKw: { type: "object", properties: {}, additionalProperties: true },
  scope: { type: "string", enum: ["repo", "model"] },
  env: { type: "array", items: { type: "string" } },
  runRoot: { type: "string" },
  weights: { type: "string" },
  seed: { type: "integer" },
  sourcePython: { type: "string" },
  targetPython: { type: "string" },
  sourcePath: { type: "string" },
  targetPath: { type: "string" },
  checkRun: { type: "boolean" },
  tryHints: { type: "boolean" },
  noAlign: { type: "boolean" },
} as const;

function addOption(args: string[], name: string, value: unknown): void {
  if ((typeof value === "string" && value.length > 0) || typeof value === "number") {
    args.push(name, String(value));
  }
}

function addList(args: string[], name: string, value: unknown): void {
  if (!Array.isArray(value)) return;
  for (const item of value) addOption(args, name, item);
}

function addJson(args: string[], name: string, value: unknown): void {
  if (value !== undefined) addOption(args, name, JSON.stringify(value));
}

function addJsonList(args: string[], name: string, value: unknown): void {
  if (!Array.isArray(value)) return;
  for (const item of value) addJson(args, name, item);
}

function autoArgs(values: Record<string, unknown>, agent: JsonMap): string[] {
  const args = ["auto", String(values.repository), "--output", String(values.output)];
  addOption(args, "--target", values.target);
  addOption(args, "--entrypoint", values.entrypoint);
  addOption(args, "--model", values.model);
  addOption(args, "--shape", values.shape);
  addOption(args, "--dtype", values.dtype);
  addJsonList(args, "--input", values.input);
  addJson(args, "--model-args", values.modelArgs);
  addJson(args, "--model-kw", values.modelKw);
  addOption(args, "--scope", values.scope);
  addList(args, "--env", values.env);
  addOption(args, "--run-root", values.runRoot);
  addOption(args, "--weights", values.weights);
  addOption(args, "--seed", values.seed);
  addOption(args, "--source-python", values.sourcePython);
  addOption(args, "--target-python", values.targetPython);
  addOption(args, "--source-path", values.sourcePath);
  addOption(args, "--target-path", values.targetPath);
  if (values.checkRun === true) args.push("--check-run");
  if (values.tryHints === true) args.push("--try-hints");
  if (values.noAlign === true) args.push("--no-align");
  args.push("--agent", JSON.stringify(agent));
  return args;
}

function alignArgs(values: Record<string, unknown>): string[] {
  const args = [
    "align",
    String(values.source),
    String(values.target),
    "--run-root",
    String(values.runRoot),
  ];
  addOption(args, "--entry", values.entry);
  addOption(args, "--model", values.model);
  addOption(args, "--shape", values.shape);
  addOption(args, "--dtype", values.dtype);
  addJsonList(args, "--input", values.input);
  addJson(args, "--model-args", values.modelArgs);
  addJson(args, "--model-kw", values.modelKw);
  addOption(args, "--weights", values.weights);
  addOption(args, "--seed", values.seed);
  addOption(args, "--source-python", values.sourcePython);
  addOption(args, "--target-python", values.targetPython);
  addOption(args, "--source-path", values.sourcePath);
  addOption(args, "--target-path", values.targetPath);
  return args;
}

function render(_args: unknown, value: RunResult) {
  return [{ type: "text" as const, text: JSON.stringify(value, null, 2) }];
}

export function registerTools(ctx: Context, config: Config): void {
  ctx.tools.register(defineTool({
    name: "scan_repository",
    timeoutMs: config.timeoutMs,
    description: "扫描未知 PyTorch 仓库，返回入口、框架、权重、依赖和风险事实。",
    parameters: { repository: { type: "string", required: true } },
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(config, ["scan", args.repository], exec),
  }));
  ctx.tools.register(defineTool({
    name: "create_migration_plan",
    timeoutMs: config.timeoutMs,
    description: "生成 PyTorch 到 Jittor 的确定性迁移计划，不修改仓库。",
    parameters: {
      repository: { type: "string", required: true },
      target: { type: "string", enum: ["jtorch", "jittor-native", "jittor-ascend"] },
    },
    output: { schema: outputSchema, render },
    execute: (args, exec) => {
      const values = ["plan", args.repository];
      addOption(values, "--target", args.target);
      return runTool(config, values, exec);
    },
  }));
  ctx.tools.register(defineTool({
    name: "start_migration",
    timeoutMs: config.timeoutMs,
    description: "执行一次隔离迁移、目标端检查和可选数值对齐，并记录 DSH 决策。",
    parameters: {
      ...runParams,
      round: { type: "integer" },
      decision: { type: "string" },
      candidate: { type: "string" },
    },
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(config, autoArgs(args, {
      mode: "dsh",
      round: args.round ?? 1,
      decision: args.decision ?? "initial migration",
      candidate: args.candidate ?? "torchport auto",
    }), exec, args.output),
  }));
  ctx.tools.register(defineTool({
    name: "run_verification",
    timeoutMs: config.timeoutMs,
    description: "对源 PyTorch 仓库和迁移后 Jittor 仓库执行双端张量采集与数值验证。",
    parameters: {
      source: { type: "string", required: true },
      target: { type: "string", required: true },
      runRoot: { type: "string", required: true },
      entry: { type: "string" }, model: { type: "string" }, shape: { type: "string" },
      dtype: { type: "string", enum: ["float32", "int64"] },
      input: { type: "array", items: { type: "json" } },
      modelArgs: { type: "array", items: { type: "json" } },
      modelKw: { type: "object", properties: {}, additionalProperties: true },
      seed: { type: "integer" },
      weights: { type: "string" }, sourcePython: { type: "string" }, targetPython: { type: "string" },
      sourcePath: { type: "string" }, targetPath: { type: "string" },
    },
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(config, alignArgs(args), exec, args.target),
  }));
  ctx.tools.register(defineTool({
    name: "locate_first_divergence",
    timeoutMs: config.timeoutMs,
    description: "读取 auto.json，返回首次分歧位置和当前闭环状态。",
    parameters: { report: { type: "string", required: true } },
    output: { schema: outputSchema, render },
    execute: async (args, exec) => {
      const data = nextStep(await readJson(resolve(toolCwd(exec), args.report)));
      return { ok: true, code: 0, command: [], data, stderr: "", trace: "" };
    },
  }));
  ctx.tools.register(defineTool({
    name: "propose_repairs",
    timeoutMs: config.timeoutMs,
    description: "读取 TorchPort 的结构化失败与候选动作，供 DSH 选择下一轮修复。",
    parameters: { report: { type: "string", required: true } },
    output: { schema: outputSchema, render },
    execute: async (args, exec) => {
      const data = nextStep(await readJson(resolve(toolCwd(exec), args.report)));
      return { ok: data.state !== "failed", code: 0, command: [], data, stderr: "", trace: "" };
    },
  }));
  ctx.tools.register(defineTool({
    name: "apply_verified_repair",
    timeoutMs: config.timeoutMs,
    description: "在 DSH 选定代码或运行环境修复后，重新迁移并验证，保存修复依据。",
    parameters: {
      ...runParams,
      round: { type: "integer", required: true },
      decision: { type: "string", required: true },
      candidate: { type: "string", required: true },
    },
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(config, autoArgs(args, {
      mode: "dsh", round: args.round, decision: args.decision, candidate: args.candidate,
    }), exec, args.output),
  }));
  ctx.tools.register(defineTool({
    name: "generate_certification_report",
    timeoutMs: config.timeoutMs,
    description: "把 TorchPort auto.json 归一成可发布、可审计的 DSH 迁移报告。",
    parameters: {
      report: { type: "string", required: true },
      output: { type: "string", required: true },
    },
    output: { schema: outputSchema, render },
    execute: async (args, exec) => {
      const cwd = toolCwd(exec);
      const report = makeReport(await readJson(resolve(cwd, args.report)));
      const target = resolve(cwd, args.output);
      await saveReport(target, report);
      return { ok: true, code: 0, command: [], data: report, stderr: "", trace: dirname(target) };
    },
  }));
}
