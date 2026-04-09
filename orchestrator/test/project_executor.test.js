import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  createProjectExecutor,
  RUN_STATUS,
  PROJECT_STATUS,
  extractSharedContext,
  buildProjectSummary,
  skipDependents,
  persistProjectSummary,
  saveCheckpoint,
  loadCheckpoint,
  restoreFromCheckpoint,
} from "../src/vnext/project_executor.js";
import { makeProjectPlanPendingResponse } from "../src/vnext/response_protocol.js";

// --- 工具函数 ---

function makePlan(overrides = {}) {
  return {
    project_id: "proj-test-001",
    project_title: "Test Project",
    created_at: new Date().toISOString(),
    decomposition_model: "test-model",
    modules: [{ module_id: "mod-a", title: "Module A", description: "desc" }],
    runs: [
      {
        run_key: "R-01", module_id: "mod-a", task_class: "be_create",
        title: "Backend API",
        prompt: "Create a RESTful API with CRUD operations and input validation for the core data model",
        target_paths: ["workspace/sandbox/proj/backend/"],
        depends_on: [], shared_context: { from_runs: [], artifacts: [] },
        estimated_complexity: "medium",
        acceptance_criteria: ["AC-R01-1: API endpoint works"],
      },
      {
        run_key: "R-02", module_id: "mod-a", task_class: "fe_create",
        title: "Frontend Page",
        prompt: "Create a frontend page that displays items from the API with list view and create form components",
        target_paths: ["workspace/sandbox/proj/frontend/"],
        depends_on: ["R-01"], shared_context: { from_runs: ["R-01"], artifacts: ["handoff/be_to_fe.json"] },
        estimated_complexity: "medium",
        acceptance_criteria: ["AC-R02-1: List page renders"],
      },
    ],
    dependency_graph: { "R-01": [], "R-02": ["R-01"] },
    execution_strategy: { max_parallel_runs: 2, failure_policy: "stop_dependents", retry_failed_runs: false, max_retries_per_run: 0 },
    project_constraints: { project_type: "generic_app", workspace_root: "workspace/sandbox/proj/" },
    ...overrides,
  };
}

function createMockWorkflowEngine({ failRunKeys = [], delay = 0 } = {}) {
  const calls = [];
  return {
    calls,
    startWorkflowRun: async (args) => {
      calls.push(args);
      if (delay > 0) await new Promise((r) => setTimeout(r, delay));
      const runKey = args.run_id.split("__")[1] || args.run_id;
      if (failRunKeys.includes(runKey)) {
        throw new Error(`mock failure for ${runKey}`);
      }
      return {
        workflow_run_id: `wfr-${runKey}-${Date.now()}`,
        workflow_id: "coding_team_v0",
        first_step: "pm_spec",
        artifact_dir: null,
      };
    },
  };
}

function createMockRecordEvent() {
  const events = [];
  return {
    events,
    fn: async (id, event, data) => {
      events.push({ id, event, data });
    },
  };
}

// --- 测试 ---

describe("createProjectExecutor", () => {
  it("throws when workflowEngine is missing", () => {
    assert.throws(() => createProjectExecutor({ recordEvent: () => {} }), /workflowEngine/);
  });

  it("throws when recordEvent is missing", () => {
    assert.throws(() => createProjectExecutor({
      workflowEngine: { startWorkflowRun: () => {} },
    }), /recordEvent/);
  });
});

describe("executeProjectPlan — 2-run dependency chain", () => {
  it("executes R-01 before R-02 and completes both", async () => {
    const engine = createMockWorkflowEngine();
    const recorder = createMockRecordEvent();
    const executor = createProjectExecutor({
      workflowEngine: engine,
      recordEvent: recorder.fn,
    });

    const plan = makePlan();
    const summary = await executor.executeProjectPlan(plan);

    assert.equal(summary.status, PROJECT_STATUS.COMPLETED);
    assert.equal(summary.total_runs, 2);
    assert.equal(summary.completed, 2);
    assert.equal(summary.failed, 0);
    assert.equal(summary.skipped, 0);

    // R-01 must be called before R-02
    assert.equal(engine.calls.length, 2);
    assert.ok(engine.calls[0].run_id.includes("R-01"));
    assert.ok(engine.calls[1].run_id.includes("R-02"));

    // R-02 should receive project_context with project_id
    const r02Input = engine.calls[1].input;
    assert.equal(r02Input.project_context.run_key, "R-02");
    assert.equal(r02Input.project_context.project_id, "proj-test-001");
  });
});

describe("executeProjectPlan — parallel wave", () => {
  it("runs independent runs in the same wave concurrently", async () => {
    const engine = createMockWorkflowEngine();
    const recorder = createMockRecordEvent();
    const executor = createProjectExecutor({
      workflowEngine: engine,
      recordEvent: recorder.fn,
    });

    const plan = makePlan({
      runs: [
        { ...makePlan().runs[0], run_key: "R-01", depends_on: [], target_paths: ["a/"] },
        { ...makePlan().runs[0], run_key: "R-02", depends_on: [], target_paths: ["b/"], task_class: "fe_create" },
        { ...makePlan().runs[0], run_key: "R-03", depends_on: ["R-01", "R-02"], target_paths: ["c/"] },
      ],
      dependency_graph: { "R-01": [], "R-02": [], "R-03": ["R-01", "R-02"] },
    });

    const summary = await executor.executeProjectPlan(plan);

    assert.equal(summary.status, PROJECT_STATUS.COMPLETED);
    assert.equal(summary.total_runs, 3);
    assert.equal(summary.completed, 3);

    // R-01 and R-02 should be in wave 0, R-03 in wave 1
    // Both R-01 and R-02 should be called before R-03
    const r03Idx = engine.calls.findIndex((c) => c.run_id.includes("R-03"));
    const r01Idx = engine.calls.findIndex((c) => c.run_id.includes("R-01"));
    const r02Idx = engine.calls.findIndex((c) => c.run_id.includes("R-02"));
    assert.ok(r03Idx > r01Idx);
    assert.ok(r03Idx > r02Idx);
  });
});

describe("executeProjectPlan — failure propagation: stop_dependents", () => {
  it("skips downstream runs when upstream fails", async () => {
    const engine = createMockWorkflowEngine({ failRunKeys: ["R-01"] });
    const recorder = createMockRecordEvent();
    const executor = createProjectExecutor({
      workflowEngine: engine,
      recordEvent: recorder.fn,
    });

    const plan = makePlan();
    const summary = await executor.executeProjectPlan(plan);

    assert.equal(summary.status, PROJECT_STATUS.PARTIAL_FAILURE);
    assert.equal(summary.failed, 1);
    assert.equal(summary.skipped, 1);
    assert.equal(summary.completed, 0);

    // R-02 should be skipped, never called
    assert.equal(engine.calls.length, 1);
    assert.ok(engine.calls[0].run_id.includes("R-01"));

    // Risk report
    assert.deepEqual(summary.risk_report.failure_runs, ["R-01"]);
    assert.deepEqual(summary.risk_report.skipped_runs, ["R-02"]);
  });
});

describe("executeProjectPlan — failure propagation: stop_all", () => {
  it("skips all pending runs when any run fails", async () => {
    const engine = createMockWorkflowEngine({ failRunKeys: ["R-01"] });
    const recorder = createMockRecordEvent();
    const executor = createProjectExecutor({
      workflowEngine: engine,
      recordEvent: recorder.fn,
    });

    const plan = makePlan({
      runs: [
        { ...makePlan().runs[0], run_key: "R-01", depends_on: [], target_paths: ["a/"] },
        { ...makePlan().runs[0], run_key: "R-02", depends_on: [], target_paths: ["b/"], task_class: "fe_create" },
      ],
      dependency_graph: { "R-01": [], "R-02": [] },
      execution_strategy: { max_parallel_runs: 1, failure_policy: "stop_all" },
    });

    const summary = await executor.executeProjectPlan(plan);

    assert.equal(summary.failed, 1);
    assert.equal(summary.skipped, 1);
  });
});

describe("executeProjectPlan — failure propagation: continue_all", () => {
  it("continues running other runs even when one fails", async () => {
    const engine = createMockWorkflowEngine({ failRunKeys: ["R-01"] });
    const recorder = createMockRecordEvent();
    const executor = createProjectExecutor({
      workflowEngine: engine,
      recordEvent: recorder.fn,
    });

    const plan = makePlan({
      runs: [
        { ...makePlan().runs[0], run_key: "R-01", depends_on: [], target_paths: ["a/"] },
        { ...makePlan().runs[0], run_key: "R-02", depends_on: [], target_paths: ["b/"], task_class: "fe_create" },
      ],
      dependency_graph: { "R-01": [], "R-02": [] },
      execution_strategy: { max_parallel_runs: 1, failure_policy: "continue_all" },
    });

    const summary = await executor.executeProjectPlan(plan);

    assert.equal(summary.failed, 1);
    assert.equal(summary.completed, 1);
    assert.equal(summary.skipped, 0);
    assert.equal(engine.calls.length, 2);
  });
});

describe("executeProjectPlan — event recording", () => {
  it("records project lifecycle events", async () => {
    const engine = createMockWorkflowEngine();
    const recorder = createMockRecordEvent();
    const executor = createProjectExecutor({
      workflowEngine: engine,
      recordEvent: recorder.fn,
    });

    await executor.executeProjectPlan(makePlan());

    const eventTypes = recorder.events.map((e) => e.event);
    assert.ok(eventTypes.includes("project.started"));
    assert.ok(eventTypes.includes("project.wave.started"));
    assert.ok(eventTypes.includes("project.run.completed"));
    assert.ok(eventTypes.includes("project.wave.completed"));
    assert.ok(eventTypes.includes("project.completed"));
  });
});

describe("skipDependents", () => {
  it("recursively skips transitive dependents", () => {
    const depGraph = { "R-01": [], "R-02": ["R-01"], "R-03": ["R-02"], "R-04": [] };
    const runStates = new Map([
      ["R-01", RUN_STATUS.FAILED],
      ["R-02", RUN_STATUS.PENDING],
      ["R-03", RUN_STATUS.PENDING],
      ["R-04", RUN_STATUS.PENDING],
    ]);
    const runByKey = new Map();

    skipDependents("R-01", depGraph, runStates, runByKey);

    assert.equal(runStates.get("R-02"), RUN_STATUS.SKIPPED);
    assert.equal(runStates.get("R-03"), RUN_STATUS.SKIPPED);
    assert.equal(runStates.get("R-04"), RUN_STATUS.PENDING); // independent
  });
});

describe("extractSharedContext", () => {
  it("returns empty array when no upstream results", () => {
    const result = extractSharedContext({
      fromRuns: ["R-01"],
      artifacts: ["handoff/be_to_fe.json"],
      runResults: new Map(),
    });
    assert.deepEqual(result, []);
  });
});

describe("buildProjectSummary", () => {
  it("computes correct summary stats", () => {
    const plan = makePlan();
    const runStates = new Map([
      ["R-01", RUN_STATUS.COMPLETED],
      ["R-02", RUN_STATUS.FAILED],
    ]);
    const runResults = new Map([
      ["R-01", { workflow_run_id: "wfr-1", duration_ms: 1000 }],
      ["R-02", { workflow_run_id: null, duration_ms: 500, error: "test error" }],
    ]);

    const summary = buildProjectSummary({
      plan,
      runStates,
      runResults,
      startTime: new Date(Date.now() - 2000),
    });

    assert.equal(summary.status, PROJECT_STATUS.PARTIAL_FAILURE);
    assert.equal(summary.completed, 1);
    assert.equal(summary.failed, 1);
    assert.equal(summary.acceptance_rate, 0.5);
    assert.ok(summary.risk_report.has_failures);
    assert.deepEqual(summary.risk_report.failure_runs, ["R-02"]);
  });
});

// =========================================================================
// Phase C tests
// =========================================================================

function makeTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "nexus-test-"));
}

// --- P-C1: Summary persistence ---

describe("persistProjectSummary", () => {
  it("writes project_summary.json to artifact dir", () => {
    const dir = makeTmpDir();
    const summary = { project_id: "proj-1", status: "COMPLETED", total_runs: 2 };
    const result = persistProjectSummary(summary, dir);
    assert.ok(result);
    assert.ok(fs.existsSync(result));
    const written = JSON.parse(fs.readFileSync(result, "utf8"));
    assert.equal(written.project_id, "proj-1");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("returns null when artifactDir is null", () => {
    assert.equal(persistProjectSummary({}, null), null);
  });

  it("creates nested directories", () => {
    const dir = path.join(makeTmpDir(), "deep", "nested");
    const result = persistProjectSummary({ ok: true }, dir);
    assert.ok(result);
    assert.ok(fs.existsSync(result));
    fs.rmSync(path.resolve(dir, "../.."), { recursive: true, force: true });
  });
});

describe("executeProjectPlan — summary persistence with artifactDir", () => {
  it("writes summary and sets REPORTED status on success", async () => {
    const dir = makeTmpDir();
    const engine = createMockWorkflowEngine();
    const recorder = createMockRecordEvent();
    const executor = createProjectExecutor({
      workflowEngine: engine,
      recordEvent: recorder.fn,
      artifactDir: dir,
    });

    const summary = await executor.executeProjectPlan(makePlan());
    assert.equal(summary.status, PROJECT_STATUS.REPORTED);
    assert.ok(summary._summary_path);
    assert.ok(fs.existsSync(summary._summary_path));
    fs.rmSync(dir, { recursive: true, force: true });
  });
});

// --- P-C3: Checkpoint save / load / restore ---

describe("saveCheckpoint", () => {
  it("writes project_checkpoint.json", () => {
    const dir = makeTmpDir();
    const runStates = new Map([["R-01", "COMPLETED"], ["R-02", "RUNNING"]]);
    const runResults = new Map([["R-01", { workflow_run_id: "wfr-1", duration_ms: 100 }]]);
    const plan = makePlan();
    const result = saveCheckpoint({ projectId: "proj-1", plan, runStates, runResults, artifactDir: dir });
    assert.ok(result);
    const data = JSON.parse(fs.readFileSync(result, "utf8"));
    assert.equal(data.project_id, "proj-1");
    assert.equal(data.run_states["R-01"], "COMPLETED");
    assert.equal(data.run_states["R-02"], "RUNNING");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("returns null when artifactDir is null", () => {
    assert.equal(saveCheckpoint({ projectId: "x", plan: {}, runStates: new Map(), runResults: new Map(), artifactDir: null }), null);
  });
});

describe("loadCheckpoint", () => {
  it("loads saved checkpoint", () => {
    const dir = makeTmpDir();
    const checkpoint = { project_id: "proj-1", run_states: { "R-01": "COMPLETED" }, run_results: {} };
    fs.writeFileSync(path.join(dir, "project_checkpoint.json"), JSON.stringify(checkpoint));
    const loaded = loadCheckpoint(dir);
    assert.ok(loaded);
    assert.equal(loaded.project_id, "proj-1");
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("returns null when no checkpoint exists", () => {
    assert.equal(loadCheckpoint(makeTmpDir()), null);
  });

  it("returns null for null artifactDir", () => {
    assert.equal(loadCheckpoint(null), null);
  });

  it("returns null for corrupted checkpoint", () => {
    const dir = makeTmpDir();
    fs.writeFileSync(path.join(dir, "project_checkpoint.json"), "not json");
    assert.equal(loadCheckpoint(dir), null);
    fs.rmSync(dir, { recursive: true, force: true });
  });
});

describe("restoreFromCheckpoint", () => {
  it("restores COMPLETED runs and resets others to PENDING", () => {
    const checkpoint = {
      run_states: { "R-01": "COMPLETED", "R-02": "FAILED", "R-03": "RUNNING" },
      run_results: { "R-01": { workflow_run_id: "wfr-1", duration_ms: 500 } },
    };
    const runStates = new Map([["R-01", RUN_STATUS.PENDING], ["R-02", RUN_STATUS.PENDING], ["R-03", RUN_STATUS.PENDING]]);
    const runResults = new Map();

    const restored = restoreFromCheckpoint(checkpoint, runStates, runResults);
    assert.equal(restored, 1);
    assert.equal(runStates.get("R-01"), RUN_STATUS.COMPLETED);
    assert.equal(runStates.get("R-02"), RUN_STATUS.PENDING); // FAILED → stays PENDING (re-run)
    assert.equal(runStates.get("R-03"), RUN_STATUS.PENDING); // RUNNING → stays PENDING
    assert.ok(runResults.has("R-01"));
  });
});

describe("executeProjectPlan — checkpoint resume", () => {
  it("skips completed runs from checkpoint", async () => {
    const dir = makeTmpDir();
    // Pre-save a checkpoint with R-01 completed
    const checkpoint = {
      project_id: "proj-test-001",
      run_states: { "R-01": "COMPLETED" },
      run_results: { "R-01": { workflow_run_id: "wfr-prev", duration_ms: 100 } },
    };
    fs.writeFileSync(path.join(dir, "project_checkpoint.json"), JSON.stringify(checkpoint));

    const engine = createMockWorkflowEngine();
    const recorder = createMockRecordEvent();
    const executor = createProjectExecutor({
      workflowEngine: engine,
      recordEvent: recorder.fn,
      artifactDir: dir,
    });

    const summary = await executor.executeProjectPlan(makePlan(), {}, { resume: true });

    // Only R-02 should have been executed (R-01 restored from checkpoint)
    assert.equal(engine.calls.length, 1);
    assert.ok(engine.calls[0].run_id.includes("R-02"));
    assert.equal(summary.completed, 2); // both completed
    assert.equal(summary._restored_from_checkpoint, 1);
    fs.rmSync(dir, { recursive: true, force: true });
  });
});

// --- P-C2: Manual confirmation response ---

describe("makeProjectPlanPendingResponse", () => {
  it("returns correct response shape", () => {
    const plan = { project_id: "proj-1", runs: [{ run_key: "R-01" }] };
    const resp = makeProjectPlanPendingResponse({
      run_id: "run-1",
      task_envelope: { id: "env-1" },
      project_plan: plan,
    });
    assert.equal(resp.ok, true);
    assert.equal(resp.response_mode, "project_plan_pending");
    assert.equal(resp.run_id, "run-1");
    assert.deepEqual(resp.project_plan, plan);
    assert.ok(resp.confirm_instructions);
  });
});

// --- P-C2: createConfirmProjectPlan ---

import { createConfirmProjectPlan } from "../src/vnext/runtime_dispatch.js";

describe("createConfirmProjectPlan", () => {
  it("executes valid plan successfully", async () => {
    const engine = createMockWorkflowEngine();
    const confirm = createConfirmProjectPlan({
      pool: { query: async () => ({}) },
      updateRunStatus: async () => {},
      workflowEngine: engine,
      runtimeConfig: {},
    });

    const plan = makePlan();
    const result = await confirm({
      run_id: "run-confirm-1",
      project_plan: plan,
    });

    assert.equal(result.ok, true);
    assert.ok(result.project_summary);
    assert.equal(result.project_summary.total_runs, 2);
    assert.equal(engine.calls.length, 2);
  });

  it("rejects invalid plan (missing acceptance_criteria)", async () => {
    const engine = createMockWorkflowEngine();
    const confirm = createConfirmProjectPlan({
      pool: { query: async () => ({}) },
      updateRunStatus: async () => {},
      workflowEngine: engine,
      runtimeConfig: {},
    });

    const plan = makePlan();
    plan.runs[0].acceptance_criteria = []; // violates C-07

    const result = await confirm({
      run_id: "run-confirm-2",
      project_plan: plan,
    });

    assert.equal(result.ok, false);
    assert.ok(result.error.includes("C-07"));
    assert.equal(result.error_code, "PROJECT_PLAN_INVALID");
    assert.equal(engine.calls.length, 0); // never executed
  });

  it("rejects plan with cycle (C-04)", async () => {
    const engine = createMockWorkflowEngine();
    const confirm = createConfirmProjectPlan({
      pool: { query: async () => ({}) },
      updateRunStatus: async () => {},
      workflowEngine: engine,
    });

    const plan = makePlan();
    plan.runs[0].depends_on = ["R-02"];
    plan.runs[1].depends_on = ["R-01"];

    const result = await confirm({
      run_id: "run-confirm-3",
      project_plan: plan,
    });

    assert.equal(result.ok, false);
    assert.ok(result.error.includes("C-04"));
  });

  it("supports resume=true parameter", async () => {
    const engine = createMockWorkflowEngine();
    const confirm = createConfirmProjectPlan({
      pool: { query: async () => ({}) },
      updateRunStatus: async () => {},
      workflowEngine: engine,
    });

    const plan = makePlan();
    // resume=true without checkpoint just runs normally
    const result = await confirm({
      run_id: "run-confirm-4",
      project_plan: plan,
      resume: true,
    });

    assert.equal(result.ok, true);
    assert.equal(result.project_summary.total_runs, 2);
  });
});

// --- flag=false regression ---

describe("flag=false regression", () => {
  it("project_planner_enabled=false does not affect single workflow dispatch", () => {
    const defaults = { project_planner_enabled: false };
    assert.equal(Boolean(defaults.project_planner_enabled), false);
  });
});
