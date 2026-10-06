import type { Context } from "@deepseek-ai/cordis";
import { defineTool } from "@deepseek-ai/dsh-tools";

import { runTool, type RunResult } from "./trace.js";
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

const taskParams = {
  task: { type: "string", required: true, description: "冻结的 task.yaml / task.json 路径" },
  root: { type: "string", description: "运行证据根目录" },
  trial: { type: "string", description: "本次候选尝试的稳定标识" },
} as const;

function addOption(args: string[], name: string, value: unknown): void {
  if ((typeof value === "string" && value.length > 0) || typeof value === "number") {
    args.push(name, String(value));
  }
}

function harnessArgs(command: string, values: Record<string, unknown>): string[] {
  const args = [command, String(values.task)];
  addOption(args, "--root", values.root);
  addOption(args, "--trial", values.trial);
  return args;
}

function render(_args: unknown, value: RunResult) {
  return [{ type: "text" as const, text: JSON.stringify(value, null, 2) }];
}

export function registerTools(ctx: Context, config: Config): void {
  ctx.tools.register(defineTool({
    name: "torchport_inspect",
    timeoutMs: config.timeoutMs,
    description: "查看公开任务契约，不读取 oracle。对应 torchport inspect。",
    parameters: { task: taskParams.task },
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(config, ["inspect", String(args.task)], exec),
  }));

  ctx.tools.register(defineTool({
    name: "torchport_run",
    timeoutMs: config.timeoutMs,
    description: "按冻结任务执行 source 或 target，输出 RunManifest，不判定迁移成功。对应 torchport run。",
    parameters: {
      ...taskParams,
      side: { type: "string", required: true, enum: ["source", "target"] },
      mode: { type: "string", enum: ["normal", "trace"] },
    },
    output: { schema: outputSchema, render },
    execute: (args, exec) => {
      const command = [...harnessArgs("run", args), String(args.side)];
      addOption(command, "--mode", args.mode);
      return runTool(config, command, exec);
    },
  }));

  ctx.tools.register(defineTool({
    name: "torchport_evaluate",
    timeoutMs: config.timeoutMs,
    description: "读取既有 oracle 与 target RunManifest 并判定，不启动模型。对应 torchport evaluate。",
    parameters: {
      ...taskParams,
      manifest: { type: "string", required: true, description: "target 端 RunManifest 路径" },
    },
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(
      config,
      [...harnessArgs("evaluate", args), String(args.manifest)],
      exec,
    ),
  }));

  ctx.tools.register(defineTool({
    name: "torchport_trace",
    timeoutMs: config.timeoutMs,
    description: "以同一输入和状态采集双端 trace。对应 torchport trace。",
    parameters: taskParams,
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(config, harnessArgs("trace", args), exec),
  }));

  ctx.tools.register(defineTool({
    name: "torchport_localize",
    timeoutMs: config.timeoutMs,
    description: "从已有 trace 报告提取首次分歧。对应 torchport localize。",
    parameters: { report: { type: "string", required: true, description: "trace 报告路径" } },
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(config, ["localize", String(args.report)], exec),
  }));

  ctx.tools.register(defineTool({
    name: "torchport_verify",
    timeoutMs: config.timeoutMs,
    description: "冻结 oracle 并重复运行 target，给出唯一验收结果。对应 torchport verify。",
    parameters: taskParams,
    output: { schema: outputSchema, render },
    execute: (args, exec) => runTool(config, harnessArgs("verify", args), exec),
  }));
}
