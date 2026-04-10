import test from "node:test";
import assert from "node:assert/strict";

import { createConfirmProjectPlan } from "../src/vnext/runtime_dispatch.js";

function makeConfirmHandler(overrides = {}) {
  const statusLog = [];
  const runStore = overrides.runStore || {};
  return {
    handler: createConfirmProjectPlan({
      pool: { query: async () => ({ rows: [] }) },
      updateRunStatus: async (_pool, run_id, status) => { statusLog.push({ run_id, status }); },
      getRunById: async (_pool, rid) => runStore[rid] || null,
      workflowEngine: {
        startWorkflowRun: async (opts) => ({
          workflow_run_id: `wf-${opts.run_id}`,
          run_id: opts.run_id,
          workflow_id: opts.workflow_id,
          project_type: opts.project_type || "webapp_crm",
          artifact_dir: `/workspace/runtime/artifacts/release/${opts.run_id}`,
          first_step: null,
          first_steps: [],
        }),
        ...overrides.workflowEngine,
      },
      runtimeConfig: { worker_coder: { project_planner_max_runs: 12 } },
      waterfallTraceService: null,
      coderProviderDefault: "opencode",
      coderModelDefault: "minimax-coding-plan/MiniMax-M2.7",
    }),
    statusLog,
  };
}

function makeValidPlan(runCount = 2) {
  const runs = [];
  for (let i = 0; i < runCount; i++) {
    const idx = String(i + 1).padStart(2, "0");
    runs.push({
      run_key: `R-${idx}`,
      title: `Run ${idx}`,
      task_class: "be_create",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      goal: `Goal ${idx}`,
      prompt: "Implement a comprehensive backend module with full CRUD operations and validation logic for this part of the project.",
      target_paths: [`workspace/sandbox/crm_site/part${idx}/`],
      depends_on: i > 0 ? [`R-${String(i).padStart(2, "0")}`] : [],
      acceptance_criteria: [`AC-R${idx}-1: Module ${idx} works correctly`],
      shared_context: { from_runs: i > 0 ? [`R-${String(i).padStart(2, "0")}`] : [], artifacts: [] },
    });
  }
  return {
    project_id: "proj-test",
    project_title: "Test Project",
    runs,
    dependency_graph: Object.fromEntries(runs.map((r) => [r.run_key, r.depends_on])),
    execution_strategy: { max_parallel_runs: 1, failure_policy: "stop_dependents" },
  };
}

test("confirmProjectPlan: valid plan executes successfully", async () => {
  const { handler, statusLog } = makeConfirmHandler({
    runStore: { "run-confirm-1": { run_id: "run-confirm-1", status: "pending_confirmation" } },
  });
  const result = await handler({
    run_id: "run-confirm-1",
    project_plan: makeValidPlan(2),
  });
  assert.equal(result.ok, true);
  assert.ok(result.project_summary);
  assert.equal(result.project_summary.total_runs, 2);
  assert.ok(statusLog.some((s) => s.status === "running"));
  assert.ok(statusLog.some((s) => s.status === "completed"));
});

test("confirmProjectPlan: rejects run not in pending_confirmation state", async () => {
  const { handler } = makeConfirmHandler({
    runStore: { "run-wrong-state": { run_id: "run-wrong-state", status: "completed" } },
  });
  const result = await handler({
    run_id: "run-wrong-state",
    project_plan: makeValidPlan(2),
  });
  assert.equal(result.ok, false);
  assert.equal(result.error_code, "INVALID_RUN_STATE");
  assert.match(result.error, /completed/);
});

test("confirmProjectPlan: rejects unknown run_id", async () => {
  const { handler } = makeConfirmHandler({ runStore: {} });
  const result = await handler({
    run_id: "run-not-exist",
    project_plan: makeValidPlan(2),
  });
  assert.equal(result.ok, false);
  assert.equal(result.error_code, "RUN_NOT_FOUND");
});

test("confirmProjectPlan: invalid plan returns error", async () => {
  const { handler } = makeConfirmHandler({
    runStore: { "run-confirm-bad": { run_id: "run-confirm-bad", status: "pending_confirmation" } },
  });
  const result = await handler({
    run_id: "run-confirm-bad",
    project_plan: { runs: [] },
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /validation failed/i);
});

test("confirmProjectPlan: plan with too many runs rejected", async () => {
  const { handler } = makeConfirmHandler({
    runStore: { "run-confirm-big": { run_id: "run-confirm-big", status: "pending_confirmation" } },
  });
  const bigPlan = makeValidPlan(15);
  const result = await handler({
    run_id: "run-confirm-big",
    project_plan: bigPlan,
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /validation failed/i);
});
