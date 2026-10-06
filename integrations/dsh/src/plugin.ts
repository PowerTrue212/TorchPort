import type { Context } from "@deepseek-ai/cordis";
import z from "@deepseek-ai/schemastery";

import { registerTools } from "./tools.js";

export const name = "torchport";
export const inject = ["tools"];

export interface Config {
  python: string;
  pythonPath?: string;
  runner?: string;
  traceDir: string;
  timeoutMs: number;
}

export const Config: z<Config> = z.object({
  python: z.string().default("python"),
  pythonPath: z.string().default(""),
  runner: z.string().default(""),
  traceDir: z.string().default(".torchport/dsh"),
  timeoutMs: z.number().default(2100000),
});

export function apply(ctx: Context, config: Config): void {
  registerTools(ctx, config);
}
