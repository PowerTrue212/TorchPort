import assert from "node:assert/strict";
import test from "node:test";

import { registerTools } from "../lib/tools.js";

test("registers the six TorchPort commands without extra policy tools", () => {
  const tools = [];
  const ctx = { tools: { register(tool) { tools.push(tool); } } };
  registerTools(ctx, {
    python: "python",
    pythonPath: "src",
    runner: "scripts/remote/dsh_torchport.py",
    traceDir: ".torchport/dsh",
    timeoutMs: 2100000,
  });

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
