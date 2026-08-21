import { readFile, writeFile } from "node:fs/promises";
import type { JsonValue } from "@deepseek-ai/dsh-tools";

import { asRecord, getActions, nextStep, type JsonMap } from "./loop.js";

export interface DshReport extends JsonMap {
  schema_version: number;
  kind: "torchport.dsh-report";
  status: string;
  passed: boolean;
  fp32: boolean;
  strict: boolean;
  first: string | null;
  actions: JsonValue[];
  agent: JsonValue;
  artifacts: JsonValue;
}

export async function readJson(path: string): Promise<JsonMap> {
  return asRecord(JSON.parse(await readFile(path, "utf8")));
}

export function makeReport(source: JsonMap): DshReport {
  const certificate = asRecord(source.certificate);
  const fp32 = asRecord(certificate.fp32);
  const strict = asRecord(certificate.strict);
  const step = nextStep(source);
  return {
    schema_version: 1,
    kind: "torchport.dsh-report",
    status: typeof source.status === "string" ? source.status : "unknown",
    passed: certificate.passed === true || source.passed === true,
    fp32: fp32.passed === true || source.fp32 === true,
    strict: strict.passed === true || source.strict === true,
    first: step.first,
    actions: getActions(source),
    agent: source.agent ?? null,
    artifacts: source.artifacts ?? null,
  };
}

export async function saveReport(path: string, report: DshReport): Promise<void> {
  await writeFile(path, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}
