import assert from "node:assert/strict";
import test from "node:test";

import { nextStep } from "../lib/loop.js";

test("verified report ends the loop", () => {
  const step = nextStep({
    status: "verified",
    certificate: { passed: true, first: null },
    advice: { summary: "done", actions: [] },
  });
  assert.equal(step.state, "done");
  assert.equal(step.status, "verified");
});

test("set action requests input", () => {
  const step = nextStep({
    status: "partial",
    advice: { summary: "need model", actions: [{ id: "set-model" }] },
  });
  assert.equal(step.state, "input");
  assert.deepEqual(step.actions, [{ id: "set-model" }]);
});

test("runtime action requests repair", () => {
  const step = nextStep({
    status: "partial",
    advice: { summary: "runtime failed", actions: [{ id: "fix-runtime" }] },
  });
  assert.equal(step.state, "repair");
});
