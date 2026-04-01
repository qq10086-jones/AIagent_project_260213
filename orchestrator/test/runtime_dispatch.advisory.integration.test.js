import assert from "node:assert/strict";

import { createExecuteVNextDispatch } from "../src/vnext/runtime_dispatch.js";

async function main() {
  const dispatcher = createExecuteVNextDispatch({
    ensureRun: async () => {},
    parseIntent: async () => null,
    registry: { workflows: {} },
    generateBrainDirectReply: async () => "noop",
    pool: { query: async () => ({ rows: [] }) },
    updateRunStatus: async () => {},
    enqueueTask: async () => ({
      task_id: "task-adv-1",
      waiting_approval: true,
      advisory: {
        council_advice: "review",
        advisory_summary: "Human approval recommended: destructive_command",
        risk_score: 0.75,
      },
    }),
    workflowEngine: { startWorkflowRun: async () => ({ workflow_run_id: "wf", workflow_id: "wf", first_step: null }) },
    routeTaskRequest: () => ({
      decision: "single_agent",
      task_envelope: {
        task_id: "env-adv-1",
        source: "api",
        raw_input: "rm -rf tmp",
        normalized_input: { text: "rm -rf tmp" },
        intent: "coding",
        requires_orchestration: false,
        target_team: "coding_team",
        expected_outputs: ["repo_changes"],
        constraints: { approval_mode: "manual" },
        context: {},
        decision: "single_agent",
        execution_plan: { tool_name: "coding.execute" },
        evidence_id: "e1",
        replay_tag: "r1",
      },
    }),
  });

  const result = await dispatcher({
    requestBody: { source: "api", raw_input: "rm -rf tmp" },
    run_id: "run-adv-1",
  });

  assert.equal(result.ok, true);
  assert.equal(result.execution.waiting_approval, true);
  assert.equal(result.execution.advisory?.council_advice, "review");
  console.log("runtime_dispatch.advisory.integration.test.js: all tests passed");
}

main().catch((err) => {
  console.error("runtime_dispatch.advisory.integration.test.js: failed");
  console.error(err);
  process.exit(1);
});
