/**
 * canary_coding_team_e2e.js
 *
 * WS-17-05: Full PM → Arch → BE → FE → QA → Release pipeline E2E canary.
 * Uses stub artifacts — no real LLM required.
 *
 * Test cases:
 *   1. happy_path  — all 6 steps complete, workflow reaches succeeded
 *   2. be_failure  — BE step produces invalid artifacts → workflow fails,
 *                    steps 4-6 (FE/QA/release) never execute,
 *                    steps 1-2 (PM/arch) artifacts preserved
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";
import { createWorkflowEngine } from "../src/workflow_engine.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow, getDefaultPromptScriptRegistryPath } from "../src/prompt_script_registry.js";
import { loadHandoffContractsOrThrow, getDefaultHandoffContractPath } from "../src/handoff_contract_registry.js";
import { analyzeTaskRisk } from "../src/policy.js";
import { resolveOrchestratorArtifactPath } from "./_paths.js";

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

function assert(condition, label) {
  if (!condition) throw new Error(`ASSERTION FAILED: ${label}`);
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected='${expected}' actual='${actual}'`);
  }
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }

function writeText(p, text) { ensureDir(path.dirname(p)); fs.writeFileSync(p, text, "utf8"); }

function writeJson(p, value) { ensureDir(path.dirname(p)); fs.writeFileSync(p, JSON.stringify(value, null, 2), "utf8"); }

// ---------------------------------------------------------------------------
// In-memory DB pool (same pattern as canary_coding_team_workflow_integration)
// ---------------------------------------------------------------------------

function createMemoryPool() {
  const state = {
    workflow_runs: [],
    workflow_steps: [],
    tasks: [],
    runs: [],
    workflow_checkpoints: [],
  };

  function findRun(id) {
    return state.workflow_runs.find((r) => r.workflow_run_id === id) || null;
  }
  function findStep(wrid, idx) {
    return state.workflow_steps.find(
      (s) => s.workflow_run_id === wrid && Number(s.step_index) === Number(idx)
    ) || null;
  }

  return {
    state,
    async query(sql, params = []) {
      const text = String(sql).replace(/\s+/g, " ").trim();

      if (text.startsWith("INSERT INTO workflow_runs")) {
        const [workflow_run_id, run_id, workflow_id, project_type, input_json] = params;
        state.workflow_runs.push({ workflow_run_id, run_id, workflow_id, project_type,
          status: "running", current_step_index: 0, input_json,
          error_code: null, error_message: null, last_checkpoint_id: null });
        return { rows: [] };
      }
      if (text.startsWith("INSERT INTO workflow_steps")) {
        const [workflow_run_id, step_index, step_id, role_name, tool_name, gate_name] = params;
        state.workflow_steps.push({ workflow_run_id, step_index, step_id, role_name, tool_name, gate_name,
          status: "pending", task_id: null, risk_level: null, approval_required: false,
          approval_reasons_json: "[]", result_json: null, error_code: null, checkpoint_id: null });
        return { rows: [] };
      }
      if (text === "SELECT * FROM workflow_runs WHERE workflow_run_id=$1") {
        const r = findRun(params[0]);
        return { rows: r ? [r] : [] };
      }
      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 ORDER BY step_index ASC") {
        return { rows: state.workflow_steps.filter((s) => s.workflow_run_id === params[0])
          .sort((a, b) => Number(a.step_index) - Number(b.step_index)) };
      }
      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2") {
        const s = findStep(params[0], params[1]);
        return { rows: s ? [s] : [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status=$3, task_id=$4")) {
        const s = findStep(params[0], params[1]);
        if (s) { s.status = params[2]; s.task_id = params[3]; s.risk_level = params[4];
          s.approval_required = params[5]; s.approval_reasons_json = params[6]; }
        return { rows: [] };
      }
      if (text === "UPDATE workflow_runs SET current_step_index=$2, status='running', updated_at=NOW() WHERE workflow_run_id=$1") {
        const r = findRun(params[0]);
        if (r) { r.current_step_index = params[1]; r.status = "running"; }
        return { rows: [] };
      }
      if (text === "SELECT payload_json FROM tasks WHERE task_id=$1") {
        const t = state.tasks.find((t) => t.task_id === params[0]) || null;
        return { rows: t ? [{ payload_json: t.payload_json }] : [] };
      }
      if (text === "SELECT task_id, run_id, payload_json FROM tasks WHERE task_id=$1") {
        const t = state.tasks.find((t) => t.task_id === params[0]) || null;
        return { rows: t ? [t] : [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='running'")) {
        const s = findStep(params[0], params[1]);
        if (s) s.status = "running";
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='queued'")) {
        const s = findStep(params[0], params[1]);
        if (s) s.status = "queued";
        return { rows: [] };
      }
      if (text === "SELECT gate_name FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2 LIMIT 1") {
        const s = findStep(params[0], params[1]);
        return { rows: [{ gate_name: s?.gate_name || "" }] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='failed', result_json=$3, error_code=$4")) {
        const s = findStep(params[0], params[1]);
        if (s) { s.status = "failed"; s.result_json = params[2]; s.error_code = params[3]; }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='succeeded', result_json=$3, error_code=NULL")) {
        const s = findStep(params[0], params[1]);
        if (s) { s.status = "succeeded"; s.result_json = params[2]; s.error_code = null; s.checkpoint_id = params[3]; }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_runs SET status='failed', error_code=$2, error_message=$3")) {
        const r = findRun(params[0]);
        if (r) { r.status = "failed"; r.error_code = params[1]; r.error_message = params[2]; }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='failed', error_code=$3")) {
        const s = findStep(params[0], params[1]);
        if (s && s.status !== "succeeded") { s.status = "failed"; s.error_code = params[2]; }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_runs SET status='succeeded'")) {
        const r = findRun(params[0]);
        if (r) r.status = "succeeded";
        return { rows: [] };
      }
      if (text === "UPDATE runs SET status='failed' WHERE run_id=$1") {
        return { rows: [] };
      }
      if (text === "UPDATE runs SET status='succeeded' WHERE run_id=$1") {
        return { rows: [] };
      }
      if (text.startsWith("INSERT INTO workflow_checkpoints")) {
        const [checkpoint_id, workflow_run_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, checkpoint_json] = params;
        state.workflow_checkpoints.push({ checkpoint_id, workflow_run_id, step_index, step_id,
          task_id, workspace_hash, artifact_refs_json, checkpoint_json });
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET checkpoint_id=$3")) {
        const s = findStep(params[0], params[1]);
        if (s) s.checkpoint_id = params[2];
        return { rows: [] };
      }
      if (text === "UPDATE workflow_runs SET last_checkpoint_id=$2, updated_at=NOW() WHERE workflow_run_id=$1") {
        const r = findRun(params[0]);
        if (r) r.last_checkpoint_id = params[1];
        return { rows: [] };
      }
      if (text.startsWith("SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash")) {
        return { rows: state.workflow_checkpoints.filter((c) => c.workflow_run_id === params[0])
          .sort((a, b) => Number(a.step_index) - Number(b.step_index)) };
      }
      // Asset/run_status writes are best-effort — swallow
      if (text.startsWith("INSERT INTO assets") || text.startsWith("UPDATE runs SET")) {
        return { rows: [] };
      }
      throw new Error(`Unhandled SQL in memory pool: ${text}`);
    },
  };
}

// ---------------------------------------------------------------------------
// Engine harness
// ---------------------------------------------------------------------------

function createEngineHarness(workspaceRoot) {
  const pool = createMemoryPool();
  const registry = loadRegistryOrThrow(getDefaultRegistryPath());
  const promptScriptRegistry = loadPromptScriptRegistryOrThrow(getDefaultPromptScriptRegistryPath());
  const handoffContracts = loadHandoffContractsOrThrow(getDefaultHandoffContractPath());
  const events = [];
  let nextTaskId = 0;

  const engine = createWorkflowEngine({
    pool,
    registry,
    promptScriptRegistry,
    handoffContracts,
    workspaceRoot,
    strictStepArtifacts: false,
    auditStepArtifacts: true,
    enqueueTask: async ({ payload, run_id, tool_name }) => {
      const task_id = `task-${++nextTaskId}`;
      const risk = analyzeTaskRisk(tool_name, payload || {});
      pool.state.tasks.push({ task_id, run_id, tool_name, payload_json: JSON.stringify(payload) });
      return { task_id, waiting_approval: Boolean(risk.requires_approval) };
    },
    recordEvent: async (stream_id, event_name, payload) => {
      events.push({ stream_id, event_name, payload });
    },
    makeIdempotencyKey: (run_id, tool_name, payload) =>
      crypto.createHash("sha256").update(JSON.stringify({ run_id, tool_name, payload })).digest("hex"),
  });

  return { engine, pool, events };
}

// ---------------------------------------------------------------------------
// Stub artifact writers — each writes exactly what the step validator requires
// ---------------------------------------------------------------------------

function writePmArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "spec.md"),
    "# Scope\n\n## User Stories\n\nAs a user, I want to log in.\n\n## Acceptance Criteria\n\nLogin works.\n\n## Non-Goals\n\nNo email 2FA.\n\n## Artifact List\n\n- plan/spec.md\n");
  writeText(path.join(rootAbs, "plan", "milestones.md"),
    "scope user_stories acceptance_criteria non_goals artifact_list");
  writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
    criteria: ["AC-001: User can log in"],
    artifacts: ["plan/spec.md", "plan/milestones.md"],
    owner: "pm_agent",
    version: "v1",
  });
  writeJson(path.join(rootAbs, "handoff", "pm_to_architect.json"), {
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "CRM MVP — auth only",
    artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    acceptance: { criteria: ["User can log in"] },
  });
}

function writeArchArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "arch.md"),
    "# Module Breakdown\n\nAuth service.\n\n## Interfaces\n\nSee interfaces.md.\n\n## Dependency Choices\n\nPostgres.\n\n## Risk Notes\n\nAuth migration risk.\n");
  writeText(path.join(rootAbs, "plan", "interfaces.md"),
    "# POST /api/login\n\nRequest: { email, password }\nResponse: { token }\nAuth: none\n");
  writeText(path.join(rootAbs, "plan", "workplan.md"),
    "module breakdown interfaces dependency choices risk notes");
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ level: "high", title: "auth", mitigation: "staged rollout" }],
    decision_log: ["Use postgres for auth storage"],
  });
  writeJson(path.join(rootAbs, "handoff", "architect_to_impl.json"), {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe"],
    modules: ["auth-service"],
    interfaces: ["POST /api/login"],
    decisions: [{ adr_id: "ADR-001", title: "Use Postgres for auth storage", status: "accepted" }],
    risks: ["auth migration"],
  });
}

function writeBeArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "impl", "be_changes", "server.js"),
    "// stub backend server\nconst express = require('express');\n");
  writeText(path.join(rootAbs, "impl", "be_notes.md"),
    "# Backend Notes\n\n## API Contracts\n\nPOST /api/login implemented.\n\n## Shared Types\n\n- User: { id, email }\n\n## Scope Constraints\n\n- No email verification in this sprint.\n\nRun: node server.js\n");
  writeJson(path.join(rootAbs, "handoff", "be_to_fe.json"), {
    from_step: "impl_be",
    to_step: "impl_fe",
    be_changes_path: "impl/be_changes",
    api_contracts: [{ name: "login", method: "POST", path: "/api/login", response_shape: "{ token }" }],
    shared_types: [{ name: "User", description: "{ id, email }" }],
    scope_constraints: ["no email verification"],
  });
}

function writeFeArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "impl", "fe_changes", "app.js"),
    "// stub frontend app\nfetch('/api/login', { method: 'POST', body: JSON.stringify({ email, password }) });\n");
  writeText(path.join(rootAbs, "impl", "fe_notes.md"),
    "# Frontend Notes\n\nConsumed POST /api/login from be_to_fe.json.\nRun: npm start\n");
  writeJson(path.join(rootAbs, "handoff", "impl_to_qa.json"), {
    from_steps: ["impl_be", "impl_fe"],
    to_step: "qa_verify",
    be_changes_path: "impl/be_changes",
    fe_changes_path: "impl/fe_changes",
    run_instructions: "Start backend, start frontend, verify login flow.",
    known_limitations: ["no email verification"],
    api_contracts_path: "handoff/be_to_fe.json",
  });
}

function writeQaArtifacts(rootAbs) {
  writeJson(path.join(rootAbs, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [
      { check_id: "D-01", layer: "deterministic", description: "BE artifacts present", status: "pass", detail: "impl/be_changes/ non-empty" },
      { check_id: "D-02", layer: "deterministic", description: "FE artifacts present", status: "pass", detail: "impl/fe_changes/ non-empty" },
      { check_id: "S-01", layer: "semantic", description: "API contracts consistent", status: "pass", detail: "POST /api/login matches FE consumption" },
    ],
    verified_artifacts: ["A1"],
    generated_at: new Date().toISOString(),
    step_id: "qa_verify",
  });
  writeJson(path.join(rootAbs, "handoff", "qa_to_release.json"), {
    from_step: "qa_verify",
    to_step: "release_pack",
    qa_report_path: "verify/qa_report.json",
    overall_status: "pass",
    verified_artifacts: ["A1"],
    run_id: path.basename(rootAbs),
    workflow_run_id: path.basename(rootAbs),
  });
}

function writeReleaseArtifacts(rootAbs, runId) {
  writeText(path.join(rootAbs, "release", "release_notes.md"),
    `# Release Notes\n\nCRM MVP — auth endpoint delivered and verified.\n\n## What was built\n\n- POST /api/login backend + frontend\n\n## QA Status\n\nAll acceptance criteria (AC-001) verified.\n\nRun ID: ${runId}\n`);
  writeJson(path.join(rootAbs, "release", "artifact_manifest.json"), {
    run_id: runId,
    workflow_id: "coding_team_v0",
    completed_at: new Date().toISOString(),
    artifacts: [
      { path: "plan/spec.md", type: "markdown", size_bytes: 120 },
      { path: "plan/acceptance.json", type: "json", size_bytes: 80 },
      { path: "risk/risk_report.json", type: "json", size_bytes: 90 },
      { path: "verify/qa_report.json", type: "json", size_bytes: 200 },
      { path: "release/release_notes.md", type: "markdown", size_bytes: 180 },
    ],
  });
}

// ---------------------------------------------------------------------------
// Helper: advance one step through the workflow
// ---------------------------------------------------------------------------

async function advanceStep({ engine, pool, taskIdx, writeArtifacts, workspaceRoot }) {
  const task = pool.state.tasks[taskIdx];
  if (!task) throw new Error(`No task at index ${taskIdx}`);
  const payload = JSON.parse(task.payload_json);
  const artifactRootAbs = path.resolve(workspaceRoot, payload.artifact_root);
  ensureDir(artifactRootAbs);
  writeArtifacts(artifactRootAbs);
  await engine.handleTaskClaimed(task.task_id);
  return engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output: { artifacts: [] } });
}

// ---------------------------------------------------------------------------
// Test Case 1: Happy Path — all 6 steps succeed
// ---------------------------------------------------------------------------

async function runHappyPath({ workspaceRoot, runName }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${runName}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${runName}-run`,
    input: { goal: "Build CRM webapp with auth", provider: "opencode", model: "qwen-max" },
  });

  assert(started.workflow_run_id, "happy_path: workflow_run_id present");

  // Step 0 — PM
  const pmTerminal = await advanceStep({ engine: harness.engine, pool: harness.pool,
    taskIdx: 0, writeArtifacts: writePmArtifacts, workspaceRoot });
  assertEqual(pmTerminal.handled, true, "happy_path: pm terminal handled");

  // Step 1 — Arch
  const archTerminal = await advanceStep({ engine: harness.engine, pool: harness.pool,
    taskIdx: 1, writeArtifacts: writeArchArtifacts, workspaceRoot });
  assertEqual(archTerminal.handled, true, "happy_path: arch terminal handled");

  // Step 2 — BE
  const beTerminal = await advanceStep({ engine: harness.engine, pool: harness.pool,
    taskIdx: 2, writeArtifacts: writeBeArtifacts, workspaceRoot });
  assertEqual(beTerminal.handled, true, "happy_path: be terminal handled");

  // Step 3 — FE
  const feTerminal = await advanceStep({ engine: harness.engine, pool: harness.pool,
    taskIdx: 3, writeArtifacts: writeFeArtifacts, workspaceRoot });
  assertEqual(feTerminal.handled, true, "happy_path: fe terminal handled");

  // Step 4 — QA
  const qaTerminal = await advanceStep({ engine: harness.engine, pool: harness.pool,
    taskIdx: 4, writeArtifacts: writeQaArtifacts, workspaceRoot });
  assertEqual(qaTerminal.handled, true, "happy_path: qa terminal handled");

  // Step 5 — Release
  const runId = `${runName}-run`;
  const releaseTerminal = await advanceStep({ engine: harness.engine, pool: harness.pool,
    taskIdx: 5,
    writeArtifacts: (rootAbs) => writeReleaseArtifacts(rootAbs, runId),
    workspaceRoot });
  assertEqual(releaseTerminal.handled, true, "happy_path: release terminal handled");

  // Assertions on final state
  const workflowRun = harness.pool.state.workflow_runs.find(
    (r) => r.workflow_run_id === started.workflow_run_id);
  assertEqual(workflowRun.status, "succeeded", "happy_path: workflow status");

  const steps = harness.pool.state.workflow_steps
    .filter((s) => s.workflow_run_id === started.workflow_run_id)
    .sort((a, b) => Number(a.step_index) - Number(b.step_index));
  assertEqual(steps.length, 6, "happy_path: step count");
  for (const step of steps) {
    assertEqual(step.status, "succeeded", `happy_path: step ${step.step_id} status`);
  }
  assertEqual(steps[0].step_id, "pm_spec", "happy_path: step[0] id");
  assertEqual(steps[1].step_id, "arch_design", "happy_path: step[1] id");
  assertEqual(steps[2].step_id, "impl_be", "happy_path: step[2] id");
  assertEqual(steps[3].step_id, "impl_fe", "happy_path: step[3] id");
  assertEqual(steps[4].step_id, "qa_verify", "happy_path: step[4] id");
  assertEqual(steps[5].step_id, "release_pack", "happy_path: step[5] id");

  assert(harness.pool.state.workflow_checkpoints.filter(
    (c) => c.workflow_run_id === started.workflow_run_id).length === 6,
    "happy_path: 6 checkpoints recorded");

  // Verify required output files exist on disk
  const releaseRoot = path.resolve(workspaceRoot, "artifacts", "release", runId);
  assert(fs.existsSync(path.join(releaseRoot, "release", "release_notes.md")),
    "happy_path: release_notes.md present");
  assert(fs.existsSync(path.join(releaseRoot, "release", "artifact_manifest.json")),
    "happy_path: artifact_manifest.json present");
  assert(fs.existsSync(path.join(releaseRoot, "plan", "spec.md")),
    "happy_path: pm spec.md preserved");
  assert(fs.existsSync(path.join(releaseRoot, "plan", "arch.md")),
    "happy_path: arch.md preserved");

  const succeededEvent = harness.events.find((e) => e.event_name === "workflow.succeeded");
  assert(succeededEvent, "happy_path: workflow.succeeded event emitted");

  return {
    workflow_run_id: started.workflow_run_id,
    steps_succeeded: steps.filter((s) => s.status === "succeeded").length,
    checkpoints: harness.pool.state.workflow_checkpoints.filter(
      (c) => c.workflow_run_id === started.workflow_run_id).length,
    release_root: releaseRoot,
  };
}

// ---------------------------------------------------------------------------
// Test Case 2: BE failure injection — steps 4-6 must NOT execute
// ---------------------------------------------------------------------------

async function runBeFailureInjection({ workspaceRoot, runName }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${runName}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${runName}-run`,
    input: { goal: "Build CRM webapp with auth", provider: "opencode", model: "qwen-max" },
  });

  // Step 0 — PM succeeds normally
  await advanceStep({ engine: harness.engine, pool: harness.pool,
    taskIdx: 0, writeArtifacts: writePmArtifacts, workspaceRoot });

  // Step 1 — Arch succeeds normally
  await advanceStep({ engine: harness.engine, pool: harness.pool,
    taskIdx: 1, writeArtifacts: writeArchArtifacts, workspaceRoot });

  // Step 2 — BE: provide bad artifacts (missing impl/be_changes/ and be_notes.md)
  const beTask = harness.pool.state.tasks[2];
  if (!beTask) throw new Error("be_failure: BE task not dispatched");
  const bePayload = JSON.parse(beTask.payload_json);
  const beRootAbs = path.resolve(workspaceRoot, bePayload.artifact_root);
  ensureDir(beRootAbs);
  // Deliberately omit impl/be_changes/ → triggers STEP_IMPL_BE_ARTIFACTS_MISSING
  writeText(path.join(beRootAbs, "impl", "be_notes.md"), "# Incomplete notes\n");

  await harness.engine.handleTaskClaimed(beTask.task_id);
  await harness.engine.handleTaskTerminal({
    task_id: beTask.task_id,
    status: "succeeded",
    output: { artifacts: [] },
  });

  // Assertions
  const workflowRun = harness.pool.state.workflow_runs.find(
    (r) => r.workflow_run_id === started.workflow_run_id);
  assertEqual(workflowRun.status, "failed", "be_failure: workflow status");

  const steps = harness.pool.state.workflow_steps
    .filter((s) => s.workflow_run_id === started.workflow_run_id)
    .sort((a, b) => Number(a.step_index) - Number(b.step_index));

  // Steps 0-1 succeeded
  assertEqual(steps[0].status, "succeeded", "be_failure: pm_spec succeeded");
  assertEqual(steps[1].status, "succeeded", "be_failure: arch_design succeeded");
  // Step 2 (BE) failed
  assertEqual(steps[2].status, "failed", "be_failure: impl_be failed");
  assertEqual(steps[2].error_code, "STEP_IMPL_BE_ARTIFACTS_MISSING",
    "be_failure: impl_be error_code");

  // Steps 3-5 never dispatched (task count = 3: pm, arch, be)
  assertEqual(harness.pool.state.tasks.length, 3, "be_failure: FE/QA/release not dispatched");

  // PM and arch artifacts preserved on disk
  const releaseRoot = path.resolve(workspaceRoot, "artifacts", "release", `${runName}-run`);
  assert(fs.existsSync(path.join(releaseRoot, "plan", "spec.md")),
    "be_failure: pm spec.md preserved after BE failure");
  assert(fs.existsSync(path.join(releaseRoot, "plan", "arch.md")),
    "be_failure: arch.md preserved after BE failure");

  const failedEvent = harness.events.find((e) => e.event_name === "workflow.failed");
  assert(failedEvent, "be_failure: workflow.failed event emitted");

  return {
    workflow_run_id: started.workflow_run_id,
    workflow_status: workflowRun.status,
    be_error_code: steps[2].error_code,
    tasks_dispatched: harness.pool.state.tasks.length,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const baseDir = resolveOrchestratorArtifactPath("canary", "coding_team_e2e");
  ensureDir(baseDir);

  console.log("# Coding Team E2E Canary (WS-17-05)");

  // --- Happy path ---
  const happyWorkspace = path.join(baseDir, "happy_path");
  ensureDir(happyWorkspace);
  const happyResult = await runHappyPath({ workspaceRoot: happyWorkspace, runName: "e2e-happy" });
  console.log(`- happy_path: all 6 steps succeeded, workflow=succeeded`);
  console.log(`  checkpoints=${happyResult.checkpoints} release_root=${happyResult.release_root}`);

  // --- BE failure injection ---
  const failWorkspace = path.join(baseDir, "be_failure");
  ensureDir(failWorkspace);
  const failResult = await runBeFailureInjection({ workspaceRoot: failWorkspace, runName: "e2e-be-fail" });
  console.log(`- be_failure: workflow=failed, be_error=${failResult.be_error_code}, tasks_dispatched=${failResult.tasks_dispatched}`);

  // Write canary artifact
  const report = {
    ok: true,
    generated_at: new Date().toISOString(),
    cases: {
      happy_path: {
        verdict: "pass",
        steps_succeeded: happyResult.steps_succeeded,
        checkpoints: happyResult.checkpoints,
      },
      be_failure: {
        verdict: "pass",
        workflow_status: failResult.workflow_status,
        be_error_code: failResult.be_error_code,
        tasks_dispatched: failResult.tasks_dispatched,
      },
    },
  };

  const reportPath = path.join(baseDir, "coding_team_e2e_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`- report: ${reportPath}`);
  console.log("exit: 0");
}

main().catch((err) => {
  console.error("[canary_coding_team_e2e] FAILED:", err.message || err);
  process.exit(1);
});
