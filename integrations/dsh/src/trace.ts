import { spawn } from "node:child_process";
import { appendFile, mkdir } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import type { JsonValue, ToolRunContext } from "@deepseek-ai/dsh-tools";

import { asRecord } from "./loop.js";
import type { Config } from "./plugin.js";

export interface RunResult {
  ok: boolean;
  code: number;
  command: string[];
  data: JsonValue;
  stderr: string;
  trace: string;
}

export interface TraceEvent {
  tool: string;
  command: string[];
  ok: boolean;
  code: number;
  output?: string;
  detail?: unknown;
}

export function tracePath(cwd: string, root: string, output?: string): string {
  if (output) {
    const base = isAbsolute(output) ? output : resolve(cwd, output);
    return resolve(base, ".torchport", "dsh-trace.jsonl");
  }
  const base = isAbsolute(root) ? root : resolve(cwd, root);
  return resolve(base, "trace.jsonl");
}

export async function appendTrace(path: string, event: TraceEvent): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const record = { schema_version: 1, time: new Date().toISOString(), ...event };
  await appendFile(path, `${JSON.stringify(record)}\n`, "utf8");
}

export function toolCwd(exec: ToolRunContext): string {
  return exec.agent?.session.header.cwd ?? process.cwd();
}

function parseJson(text: string): JsonValue {
  const value = text.trim();
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function compactData(tool: string, data: JsonValue): JsonValue {
  if (tool !== "auto") return data;
  const source = asRecord(data);
  const keys = [
    "kind", "status", "passed", "source", "output", "target", "failure",
    "entry_cert", "certificate", "advice", "agent", "artifacts", "attempts",
  ];
  return Object.fromEntries(keys.filter((key) => key in source).map((key) => [key, source[key] ?? null]));
}

export async function runTool(
  config: Config,
  args: string[],
  exec: ToolRunContext,
  output?: string,
): Promise<RunResult> {
  const cwd = toolCwd(exec);
  const command = [config.python, "-m", "torchport", ...args];
  const path = tracePath(cwd, config.traceDir, output);
  const env = { ...process.env };
  if (config.pythonPath) env.PYTHONPATH = [config.pythonPath, env.PYTHONPATH].filter(Boolean).join(":");
  const result = await new Promise<{ code: number; stdout: string; stderr: string }>((done) => {
    const child = spawn(config.python, command.slice(1), { cwd, env });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += String(chunk)));
    child.stderr.on("data", (chunk) => (stderr += String(chunk)));
    const abort = () => child.kill("SIGTERM");
    exec.signal.addEventListener("abort", abort, { once: true });
    child.on("close", (code) => {
      exec.signal.removeEventListener("abort", abort);
      done({ code: code ?? 1, stdout, stderr });
    });
    child.on("error", (error) => done({ code: 1, stdout, stderr: `${stderr}${error.message}` }));
  });
  const value: RunResult = {
    ok: result.code === 0,
    code: result.code,
    command,
    data: compactData(args[0] ?? "", parseJson(result.stdout)),
    stderr: result.stderr.trim(),
    trace: path,
  };
  await appendTrace(path, {
    tool: args[0] ?? "torchport", command, ok: value.ok, code: value.code,
    output, detail: value.data,
  });
  return value;
}
