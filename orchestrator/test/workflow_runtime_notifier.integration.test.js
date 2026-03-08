import test from "node:test";
import assert from "node:assert/strict";

import { buildWorkflowRuntimeNotification } from "../src/vnext/workflow_runtime_notifier.js";

test("workflow runtime notifier builds transition notification from workflow state", () => {
  const result = buildWorkflowRuntimeNotification({
    workflowState: {
      steps: [
        { step_index: 0, step_id: "pm_spec", role_name: "pm", status: "succeeded" },
        { step_index: 1, step_id: "arch_design", role_name: "architect", status: "running" },
      ],
    },
    workflowTerminal: { step_index: 0, step_id: "pm_spec" },
    status: "succeeded",
    normalizedErrorCode: "",
    streamError: "",
    output: {},
  });

  assert.equal(result.kind, "transition");
  assert.match(result.message, /pm_spec/);
  assert.match(result.message, /arch_design/);
});

test("workflow runtime notifier builds redacted failure notification from workflow state", () => {
  const result = buildWorkflowRuntimeNotification({
    workflowState: {
      steps: [
        { step_index: 0, step_id: "impl_be", role_name: "backend", status: "failed" },
      ],
    },
    workflowTerminal: { step_index: 0, step_id: "impl_be" },
    status: "failed",
    normalizedErrorCode: "STEP_FAILED",
    streamError: "backend compile failed",
    output: {
      stderr: "secret_token=abcdefghijklmnopqrstuvwx123456\nerror TS1005",
    },
  });

  assert.equal(result.kind, "failure");
  assert.match(result.message, /STEP_FAILED/);
  assert.match(result.message, /\*\*\*REDACTED\*\*\*/);
  assert.doesNotMatch(result.message, /abcdefghijklmnopqrstuvwx123456/);
});
