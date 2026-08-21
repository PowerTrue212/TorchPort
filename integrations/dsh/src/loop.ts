import type { JsonValue } from "@deepseek-ai/dsh-tools";

export type JsonMap = Record<string, JsonValue>;

export interface LoopStep {
  [key: string]: JsonValue;
  state: "done" | "repair" | "input" | "failed";
  status: string;
  summary: string;
  actions: JsonValue[];
  first: string | null;
}

export function asRecord(value: unknown): JsonMap {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as JsonMap)
    : {};
}

export function getActions(report: JsonMap): JsonValue[] {
  const advice = asRecord(report.advice);
  return Array.isArray(advice.actions) ? advice.actions as JsonValue[] : [];
}

function firstPoint(report: JsonMap): string | null {
  const align = asRecord(report.align);
  const strict = asRecord(align.strict);
  const certificate = asRecord(report.certificate);
  const certStrict = asRecord(certificate.strict);
  const value = strict.first ?? certStrict.first ?? certificate.first ?? report.first;
  if (typeof value === "string") return value;
  const point = asRecord(value);
  return typeof point.name === "string" ? point.name : null;
}

export function nextStep(report: JsonMap): LoopStep {
  const status = typeof report.status === "string" ? report.status : "unknown";
  const actions = getActions(report);
  const advice = asRecord(report.advice);
  const summary = typeof advice.summary === "string" ? advice.summary : "TorchPort 未提供摘要。";
  const certificate = asRecord(report.certificate);
  const passed = certificate.passed === true || report.passed === true;
  if (status === "verified" && passed) {
    return { state: "done", status, summary, actions: [], first: firstPoint(report) };
  }
  if (actions.length > 0) {
    const needsInput = actions.some((item) => {
      const action = asRecord(item);
      const id = typeof action.id === "string" ? action.id : "";
      return id.startsWith("set-");
    });
    return {
      state: needsInput ? "input" : "repair",
      status,
      summary,
      actions,
      first: firstPoint(report),
    };
  }
  return { state: "failed", status, summary, actions, first: firstPoint(report) };
}
