import assert from "node:assert/strict";
import test from "node:test";

import { registerTools } from "../lib/tools.js";

function registered(feedback) {
  const tools = [];
  const ctx = { tools: { register(tool) { tools.push(tool); } } };
  registerTools(ctx, {
    feedback,
    python: "python",
    pythonPath: "src",
    runner: "scripts/remote/dsh_torchport.py",
    traceDir: ".torchport/dsh",
    timeoutMs: 2100000,
  });
  return tools;
}

test("localized feedback registers all six TorchPort commands", () => {
  const tools = registered("localized");

  assert.deepEqual(
    tools.map((tool) => tool.name),
    [
      "torchport_inspect",
      "torchport_run",
      "torchport_evaluate",
      "torchport_trace",
      "torchport_localize",
      "torchport_verify",
    ],
  );
});

test("diagnostic feedback hides localization tools and trace mode", () => {
  const tools = registered("diagnostic");

  assert.deepEqual(
    tools.map((tool) => tool.name),
    [
      "torchport_inspect",
      "torchport_run",
      "torchport_evaluate",
      "torchport_verify",
    ],
  );
  const run = tools.find((tool) => tool.name === "torchport_run");
  assert.deepEqual(run.parameters.properties.mode.enum, ["normal"]);
});
