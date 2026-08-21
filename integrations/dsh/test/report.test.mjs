import assert from "node:assert/strict";
import test from "node:test";

import { makeReport } from "../lib/report.js";

test("certificate report keeps public evidence", () => {
  const report = makeReport({
    status: "verified",
    certificate: {
      passed: true,
      fp32: { passed: true },
      strict: { passed: false },
      first: "layer.2",
    },
    advice: { actions: [] },
    agent: { mode: "dsh", round: 2 },
    artifacts: { report: "auto.json" },
  });
  assert.equal(report.kind, "torchport.dsh-report");
  assert.equal(report.passed, true);
  assert.equal(report.fp32, true);
  assert.equal(report.strict, false);
  assert.equal(report.first, "layer.2");
});

test("structured first divergence keeps module name", () => {
  const report = makeReport({
    status: "verified",
    certificate: {
      passed: true,
      strict: { passed: false, first: { name: "conv3", position: 5 } },
    },
    advice: { actions: [] },
  });
  assert.equal(report.first, "conv3");
});
