import assert from "node:assert/strict";

import { createExecuteVNextDispatch } from "../src/vnext/runtime_dispatch.js";

function createHarnessWithInvalidSingleAgentEnvelope() {
  const calls = {
    updateRunStatus: [],
    enqueueTask: [],
  };

  const dispatcher = createExecuteVNextDispatch({
    ensureRun: async () => {},
    parseIntent: async () => null,
    registry: { workflows: {} },
    generateBrainDirectReply: async () => "noop",
    pool: { query: async () => ({ rows: [] }) },
    updateRunStatus: async (...args) => {
      calls.updateRunStatus.push(args);
    },
    enqueueTask: async (args) => {
      calls.enqueueTask.push(args);
      return { task_id: "task-should-not-run", waiting_approval: false, advisory: null };
    },
    workflowEngine: {
      startWorkflowRun: async () => {
        throw new Error("should not start workflow");
      },
    },
    routeTaskRequest: () => ({
      decision: "single_agent",
      task_envelope: {
        task_id: "env-invalid",
        source: "api",
        raw_input: "fix bug",
        normalized_input: { text: "fix bug" },
        intent: "coding",
        requires_orchestration: false,
        target_team: "coding_team",
        expected_outputs: ["repo_changes"],
        constraints: { approval_mode: "manual" },
        context: {},
        decision: "single_agent",
        execution_plan: { tool_name: "coding.delegate" },
        evidence_id: "",
        replay_tag: "",
      },
    }),
  });

  return { dispatcher, calls };
}

async function main() {
  const { dispatcher, calls } = createHarnessWithInvalidSingleAgentEnvelope();
  const result = await dispatcher({
    requestBody: { source: "api", raw_input: "fix bug" },
    run_id: "run-invalid-single-agent",
  });

  assert.equal(result.ok, false);
  assert.equal(result.error_code, "SINGLE_AGENT_GUARDRAILS_INVALID");
  assert.equal(calls.enqueueTask.length, 0);
  assert.equal(calls.updateRunStatus.at(-1)?.[2], "failed");
  console.log("single_agent_guardrails.integration.test.js: all tests passed");
}

main().catch((err) => {
  console.error("single_agent_guardrails.integration.test.js: failed");
  console.error(err);
  process.exit(1);
});
