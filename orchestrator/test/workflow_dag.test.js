import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import { createWorkflowEngine } from "../src/workflow_engine.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow, getDefaultPromptScriptRegistryPath } from "../src/prompt_script_registry.js";
import { loadHandoffContractsOrThrow, getDefaultHandoffContractPath } from "../src/handoff_contract_registry.js";
import { analyzeTaskRisk } from "../src/policy.js";

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function writeText(p, text) {
  ensureDir(path.dirname(p));
  fs.writeFileSync(p, text, "utf8");
}

function writeJson(p, value) {
  ensureDir(path.dirname(p));
  fs.writeFileSync(p, JSON.stringify(value, null, 2), "utf8");
}

function createMemoryPool() {
  const state = { workflow_runs: [], workflow_steps: [], tasks: [], runs: [], workflow_checkpoints: [] };

  function findRun(id) {
    return state.workflow_runs.find((r) => r.workflow_run_id === id) || null;
  }

  function findStep(wrid, idx) {
    return state.workflow_steps.find((s) => s.workflow_run_id === wrid && Number(s.step_index) === Number(idx)) || null;
  }

  return {
    state,
    async query(sql, params = []) {
      const text = String(sql).replace(/\s+/g, " ").trim();

      if (text.startsWith("INSERT INTO workflow_runs")) {
        const [workflow_run_id, run_id, workflow_id, project_type, input_json] = params;
        state.workflow_runs.push({
          workflow_run_id,
          run_id,
          workflow_id,
          project_type,
          status: "running",
          current_step_index: 0,
          input_json,
          error_code: null,
          error_message: null,
          last_checkpoint_id: null,
        });
        return { rows: [] };
      }
      if (text.startsWith("INSERT INTO workflow_steps")) {
        const [workflow_run_id, step_index, step_id, role_name, tool_name, gate_name] = params;
        state.workflow_steps.push({
          workflow_run_id,
          step_index,
          step_id,
          role_name,
          tool_name,
          gate_name,
          status: "pending",
          task_id: null,
          risk_level: null,
          approval_required: false,
          approval_reasons_json: "[]",
          result_json: null,
          error_code: null,
          checkpoint_id: null,
        });
        return { rows: [] };
      }
      if (text === "SELECT * FROM workflow_runs WHERE workflow_run_id=$1") {
        const row = findRun(params[0]);
        return { rows: row ? [row] : [] };
      }
      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 ORDER BY step_index ASC") {
        return {
          rows: state.workflow_steps
            .filter((s) => s.workflow_run_id === params[0])
            .sort((a, b) => Number(a.step_index) - Number(b.step_index)),
        };
      }
      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2") {
        const row = findStep(params[0], params[1]);
        return { rows: row ? [row] : [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status=$3, task_id=$4")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = params[2];
          step.task_id = params[3];
          step.risk_level = params[4];
          step.approval_required = params[5];
          step.approval_reasons_json = params[6];
        }
        return { rows: [] };
      }
      if (text === "UPDATE workflow_runs SET current_step_index=$2, status='running', updated_at=NOW() WHERE workflow_run_id=$1") {
        const run = findRun(params[0]);
        if (run) {
          run.current_step_index = params[1];
          run.status = "running";
        }
        return { rows: [] };
      }
      if (text === "UPDATE workflow_runs SET status='partial_failure', error_code=$2, error_message=$3, updated_at=NOW() WHERE workflow_run_id=$1") {
        const run = findRun(params[0]);
        if (run) {
          run.status = "partial_failure";
          run.error_code = params[1];
          run.error_message = params[2];
        }
        return { rows: [] };
      }
      if (text === "SELECT payload_json FROM tasks WHERE task_id=$1") {
        const task = state.tasks.find((t) => t.task_id === params[0]) || null;
        return { rows: task ? [{ payload_json: task.payload_json }] : [] };
      }
      if (text === "SELECT task_id, run_id, payload_json FROM tasks WHERE task_id=$1") {
        const task = state.tasks.find((t) => t.task_id === params[0]) || null;
        return { rows: task ? [task] : [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='running'")) {
        const step = findStep(params[0], params[1]);
        if (step) step.status = "running";
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='queued'")) {
        const step = findStep(params[0], params[1]);
        if (step) step.status = "queued";
        return { rows: [] };
      }
      if (text === "SELECT gate_name FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2 LIMIT 1") {
        const step = findStep(params[0], params[1]);
        return { rows: [{ gate_name: step?.gate_name || "" }] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='failed', result_json=$3, error_code=$4")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = "failed";
          step.result_json = params[2];
          step.error_code = params[3];
        }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='succeeded', result_json=$3, error_code=NULL")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = "succeeded";
          step.result_json = params[2];
          step.error_code = null;
          step.checkpoint_id = params[3];
        }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_runs SET status='failed', error_code=$2, error_message=$3")) {
        const run = findRun(params[0]);
        if (run) {
          run.status = "failed";
          run.error_code = params[1];
          run.error_message = params[2];
        }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='failed', error_code=$3")) {
        const step = findStep(params[0], params[1]);
        if (step && step.status !== "succeeded") {
          step.status = "failed";
          step.error_code = params[2];
        }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_runs SET status='succeeded'")) {
        const run = findRun(params[0]);
        if (run) run.status = "succeeded";
        return { rows: [] };
      }
      if (text === "UPDATE runs SET status=$1 WHERE run_id=$2") {
        const run = state.runs.find((item) => item.run_id === params[1]) || null;
        if (run) run.status = params[0];
        return { rows: [] };
      }
      if (text.startsWith("INSERT INTO workflow_checkpoints")) {
        const [checkpoint_id, workflow_run_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, checkpoint_json] = params;
        state.workflow_checkpoints.push({
          checkpoint_id,
          workflow_run_id,
          step_index,
          step_id,
          task_id,
          workspace_hash,
          artifact_refs_json,
          checkpoint_json,
        });
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET checkpoint_id=$3")) {
        const step = findStep(params[0], params[1]);
        if (step) step.checkpoint_id = params[2];
        return { rows: [] };
      }
      if (text === "UPDATE workflow_runs SET last_checkpoint_id=$2, updated_at=NOW() WHERE workflow_run_id=$1") {
        const run = findRun(params[0]);
        if (run) run.last_checkpoint_id = params[1];
        return { rows: [] };
      }
      if (text.startsWith("SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash")) {
        return {
          rows: state.workflow_checkpoints
            .filter((c) => c.workflow_run_id === params[0])
            .sort((a, b) => Number(a.step_index) - Number(b.step_index)),
        };
      }
      if (text.startsWith("INSERT INTO assets") || text.startsWith("UPDATE runs SET")) {
        return { rows: [] };
      }
      if (text.startsWith("INSERT INTO waterfall_stage_log")) {
        return { rows: [] };
      }
      if (text.startsWith("INSERT INTO routing_decision_log")) {
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='claiming'")) {
        const step = findStep(params[0], params[1]);
        if (step && step.status === params[2]) {
          step.status = "claiming";
          return { rowCount: 1, rows: [] };
        }
        return { rowCount: 0, rows: [] };
      }
      throw new Error(`Unhandled SQL in memory pool: ${text}`);
    },
  };
}

function createParallelRegistry() {
  const registry = loadRegistryOrThrow(getDefaultRegistryPath());
  const cloned = JSON.parse(JSON.stringify(registry));
  const steps = cloned.workflows.coding_team_v0.steps.map((step) => ({ ...step }));
  steps[2].depends_on = ["arch_design"];
  steps[3].depends_on = ["arch_design"];
  steps[4].depends_on = ["impl_be", "impl_fe"];
  steps[5].depends_on = ["smoke_test"];
  steps[6].depends_on = ["qa_verify"];
  cloned.workflows.coding_team_v0.steps = steps;
  return cloned;
}

function createSyntheticDagRegistry() {
  const registry = loadRegistryOrThrow(getDefaultRegistryPath());
  const cloned = JSON.parse(JSON.stringify(registry));
  cloned.workflows.dag_matrix_v0 = {
    project_type: "webapp_crm",
    steps: [
      { id: "root_plan", role: "pm", tool: "coding.delegate", gate: "low" },
      { id: "be_parallel", role: "backend", tool: "coding.delegate", gate: "policy", depends_on: ["root_plan"] },
      { id: "fe_parallel", role: "frontend", tool: "coding.delegate", gate: "policy", depends_on: ["root_plan"] },
      { id: "qa_join", role: "qa", tool: "coding.delegate", gate: "acceptance", depends_on: ["be_parallel", "fe_parallel"] },
    ],
  };
  return cloned;
}

function createHarness(workspaceRoot, registryOverride = null) {
  const pool = createMemoryPool();
  const registry = registryOverride || createParallelRegistry();
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

  return { engine, pool, workspaceRoot, events };
}

function writePmArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "spec.md"), "# Scope\n\n## User Stories\n\n## Acceptance Criteria\n\n## Non-Goals\n\n## Artifact List\n");
  writeText(path.join(rootAbs, "plan", "milestones.md"), "scope user_stories acceptance_criteria non_goals artifact_list");
  writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
    criteria: ["AC-001"],
    artifacts: ["plan/spec.md", "plan/milestones.md"],
    owner: "pm_agent",
    version: "v1",
  });
  writeJson(path.join(rootAbs, "handoff", "pm_to_architect.json"), {
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "CRM MVP",
    artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    acceptance: { criteria: ["AC-001"] },
  });
}

function writeArchArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "arch.md"), "# Module Breakdown\n\n## Interfaces\n\n## Dependency Choices\n\n## Risk Notes\n");
  writeText(path.join(rootAbs, "plan", "interfaces.md"), "# POST /api/login\n\nRequest: { email, password }\nResponse: { token }\n");
  writeText(
    path.join(rootAbs, "plan", "workplan.md"),
    "## BE Tasks\n- [ ] T-BE-1: Implement login endpoint in server.js | verify: POST /api/login returns 200 with token\n\n## FE Tasks\n- [ ] T-FE-1: Wire login form to POST /api/login | verify: submitting login calls /api/login and renders success state\n"
  );
  writeJson(path.join(rootAbs, "plan", "workplan.json"), {
    be_tasks: [
      {
        id: "T-BE-1",
        description: "Implement login endpoint in server.js",
        verify: "POST /api/login returns 200 with token",
      },
    ],
    fe_tasks: [
      {
        id: "T-FE-1",
        description: "Wire login form to POST /api/login",
        verify: "Submitting login calls /api/login and renders success state",
      },
    ],
  });
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ level: "medium", title: "auth", mitigation: "staged rollout" }],
    decision_log: ["Use Postgres"],
  });
  writeJson(path.join(rootAbs, "handoff", "architect_to_impl.json"), {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe"],
    modules: ["auth-service"],
    interfaces: ["POST /api/login"],
    decisions: [{ adr_id: "ADR-001", title: "Use Postgres", status: "accepted" }],
    risks: ["auth migration"],
    workplan: {
      be_tasks: [
        {
          id: "T-BE-1",
          description: "Implement login endpoint in server.js",
          verify: "POST /api/login returns 200 with token",
        },
      ],
      fe_tasks: [
        {
          id: "T-FE-1",
          description: "Wire login form to POST /api/login",
          verify: "Submitting login calls /api/login and renders success state",
        },
      ],
    },
  });
}

function writeArchArtifactsFeSafe(rootAbs) {
  writeArchArtifacts(rootAbs);
  const handoffPath = path.join(rootAbs, "handoff", "architect_to_impl.json");
  const handoff = JSON.parse(fs.readFileSync(handoffPath, "utf8"));
  handoff.parallelization = {
    fe_safe_parallel: true,
    requires_be_handoff: false,
    rationale: "Frontend can proceed from architect interfaces without backend typed handoff.",
  };
  writeJson(handoffPath, handoff);
}

function writeBeArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "impl", "be_changes", "server.js"), "// stub backend server\nconst express = require('express');\n");
  writeJson(path.join(rootAbs, "impl", "be_changes", "package.json"), {
    name: "test-app",
    version: "1.0.0",
    main: "server.js",
    dependencies: { express: "^4.19.2" },
  });
  writeText(
    path.join(rootAbs, "impl", "be_notes.md"),
    "# Backend Notes\n\n## API Contracts\n\nPOST /api/login implemented.\n\n## Shared Types\n\n- User: { id, email }\n\n## Scope Constraints\n\n- No email verification in this sprint.\n\nRun: node server.js\n"
  );
  writeJson(path.join(rootAbs, "handoff", "be_to_fe.json"), {
    from_step: "impl_be",
    to_step: "impl_fe",
    be_changes_path: "impl/be_changes",
    api_contracts: [{ name: "login", method: "POST", path: "/api/login", response_shape: "{ token }" }],
    shared_types: [{ name: "User", description: "{ id, email }" }],
    scope_constraints: ["none"],
  });
}

function writeFeArtifacts(rootAbs) {
  writeText(
    path.join(rootAbs, "impl", "fe_changes", "public", "app.js"),
    "// stub frontend app\nfetch('/api/login', { method: 'POST', body: JSON.stringify({ email, password }) });\n"
  );
  writeText(
    path.join(rootAbs, "impl", "fe_changes", "public", "index.html"),
    "<!doctype html><html><body><div id='app'></div><script type='module' src='./app.js'></script></body></html>\n"
  );
  writeText(
    path.join(rootAbs, "impl", "fe_notes.md"),
    "# Frontend Notes\n\nConsumed POST /api/login from be_to_fe.json.\nRun: npm start\n"
  );
  writeJson(path.join(rootAbs, "handoff", "impl_to_qa.json"), {
    from_steps: ["impl_be", "impl_fe"],
    to_step: "qa_verify",
    be_changes_path: "impl/be_changes",
    fe_changes_path: "impl/fe_changes",
    run_instructions: "Start backend, start frontend, verify login flow.",
    known_limitations: ["no email verification"],
    api_contracts_path: "handoff/be_to_fe.json",
    run_id: path.basename(rootAbs),
    workflow_run_id: path.basename(rootAbs),
  });
}

function writeCombinedImplArtifacts(rootAbs) {
  writeBeArtifacts(rootAbs);
  writeFeArtifacts(rootAbs);
}

async function completeTask(harness, taskIndex, writer) {
  const task = harness.pool.state.tasks[taskIndex];
  const payload = JSON.parse(task.payload_json);
  const rootAbs = path.resolve(harness.workspaceRoot, payload.artifact_root);
  ensureDir(rootAbs);
  writer(rootAbs);
  await harness.engine.handleTaskClaimed(task.task_id);
  return harness.engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output: { artifacts: [] } });
}

async function completeGenericTask(harness, taskIndex) {
  const task = harness.pool.state.tasks[taskIndex];
  await harness.engine.handleTaskClaimed(task.task_id);
  return harness.engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output: { artifacts: [] } });
}

test("dag readiness dispatches BE and FE after architect success", async () => {
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_dag/readiness");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: "dag-readiness-run", status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: "dag-readiness-run",
    input: { goal: "Build CRM" },
  });

  assert.equal(started.first_step.step_id, "pm_spec");

  await completeTask(harness, 0, writePmArtifacts);
  await completeTask(harness, 1, writeArchArtifacts);

  assert.equal(harness.pool.state.tasks.length, 4);
  assert.match(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_be").status, /^(queued|waiting_approval)$/);
  assert.match(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_fe").status, /^(queued|waiting_approval)$/);

  await completeTask(harness, 2, writeBeArtifacts);
  assert.equal(harness.pool.state.tasks.length, 4);

  await completeTask(harness, 3, writeCombinedImplArtifacts);
  const stepStatuses = harness.pool.state.workflow_steps.map((s) => ({
    step_id: s.step_id,
    status: s.status,
    error_code: s.error_code,
  }));
  assert.equal(harness.pool.state.tasks.length, 5, JSON.stringify(stepStatuses));
  assert.equal(JSON.parse(harness.pool.state.tasks[4].payload_json).step_id, "smoke_test");

  await completeGenericTask(harness, 4);
  assert.equal(harness.pool.state.tasks.length, 6, JSON.stringify(harness.pool.state.workflow_steps));
  assert.equal(JSON.parse(harness.pool.state.tasks[5].payload_json).step_id, "qa_verify");
});

test("dag mixed result enters partial_failure and blocks QA dispatch", async () => {
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_dag/partial_failure");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: "dag-partial-run", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: "dag-partial-run",
    input: { goal: "Build CRM" },
  });

  await completeTask(harness, 0, writePmArtifacts);
  await completeTask(harness, 1, writeArchArtifacts);
  await completeTask(harness, 2, writeBeArtifacts);

  const feTask = harness.pool.state.tasks[3];
  await harness.engine.handleTaskClaimed(feTask.task_id);
  await harness.engine.handleTaskTerminal({
    task_id: feTask.task_id,
    status: "failed",
    error_code: "STEP_FAILED",
    output: { reason: "synthetic fe failure" },
  });

  const stepStatuses = harness.pool.state.workflow_steps.map((s) => ({
    step_id: s.step_id,
    status: s.status,
    error_code: s.error_code,
  }));
  assert.equal(harness.pool.state.workflow_runs[0].status, "partial_failure", JSON.stringify(stepStatuses));
  assert.equal(harness.pool.state.tasks.length, 4);
  assert.equal(harness.pool.state.workflow_steps.find((s) => s.step_id === "qa_verify").status, "pending");
});

test("dag BE failure plus FE success enters partial_failure and blocks QA dispatch", async () => {
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_dag/be_failure_fe_success");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, createSyntheticDagRegistry());
  harness.pool.state.runs.push({ run_id: "dag-be-fail-run", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "dag_matrix_v0",
    project_type: "webapp_crm",
    run_id: "dag-be-fail-run",
    input: { goal: "Build CRM" },
  });

  await completeGenericTask(harness, 0);

  const beTask = harness.pool.state.tasks[1];
  await harness.engine.handleTaskClaimed(beTask.task_id);
  await harness.engine.handleTaskTerminal({
    task_id: beTask.task_id,
    status: "failed",
    error_code: "STEP_FAILED",
    output: { reason: "synthetic be failure" },
  });

  await completeGenericTask(harness, 2);

  const stepStatuses = harness.pool.state.workflow_steps.map((s) => ({
    step_id: s.step_id,
    status: s.status,
    error_code: s.error_code,
  }));
  assert.equal(harness.pool.state.workflow_runs[0].status, "partial_failure", JSON.stringify(stepStatuses));
  assert.equal(harness.pool.state.tasks.length, 3);
  assert.equal(harness.pool.state.workflow_steps.find((s) => s.step_id === "qa_join").status, "pending");
});

test("dag BE and FE failure ends as failed", async () => {
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_dag/be_fe_failure");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, createSyntheticDagRegistry());
  harness.pool.state.runs.push({ run_id: "dag-dual-fail-run", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "dag_matrix_v0",
    project_type: "webapp_crm",
    run_id: "dag-dual-fail-run",
    input: { goal: "Build CRM" },
  });

  await completeGenericTask(harness, 0);

  const beTask = harness.pool.state.tasks[1];
  await harness.engine.handleTaskClaimed(beTask.task_id);
  await harness.engine.handleTaskTerminal({
    task_id: beTask.task_id,
    status: "failed",
    error_code: "STEP_FAILED",
    output: { reason: "synthetic be failure" },
  });

  const feTask = harness.pool.state.tasks[2];
  await harness.engine.handleTaskClaimed(feTask.task_id);
  await harness.engine.handleTaskTerminal({
    task_id: feTask.task_id,
    status: "failed",
    error_code: "STEP_FAILED",
    output: { reason: "synthetic fe failure" },
  });

  assert.equal(harness.pool.state.workflow_runs[0].status, "failed");
  assert.equal(harness.pool.state.tasks.length, 3);
  assert.equal(harness.pool.state.workflow_steps.find((s) => s.step_id === "qa_join").status, "pending");
});

test("parallelization gate keeps coding_team_v0 sequential when rollout master is disabled", async () => {
  // WS-24.5-01: hardcoded lock replaced by policy-driven gate.
  // No production_parallel_rollout.json in the test workspace → rollout_master_disabled.
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_dag/gate_sequential");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, loadRegistryOrThrow(getDefaultRegistryPath()));
  harness.pool.state.runs.push({ run_id: "gate-seq-run", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: "gate-seq-run",
    input: { goal: "Build CRM" },
  });

  await completeTask(harness, 0, writePmArtifacts);
  await completeTask(harness, 1, writeArchArtifactsFeSafe);

  assert.equal(harness.pool.state.tasks.length, 3);
  assert.match(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_be").status, /^(queued|waiting_approval)$/);
  assert.equal(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_fe").status, "pending");

  const gateEvent = harness.events.find((e) => e.event_name === "workflow.parallelization.gate_decided" && e.payload?.effective_exposure_decision_source === "rollout_master_disabled");
  assert.ok(gateEvent, "gate event with rollout_master_disabled must be emitted");
  assert.equal(gateEvent.payload?.mode, "sequential");
});

test("parallelization gate keeps coding_team_v0 sequential even with registry fe_safe hints when rollout master is disabled", async () => {
  // WS-24.5-01: policy-driven gate ignores legacy registry hints; rollout master is the first layer.
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_dag/gate_parallel");
  ensureDir(workspaceRoot);
  const registry = loadRegistryOrThrow(getDefaultRegistryPath());
  const mutated = JSON.parse(JSON.stringify(registry));
  mutated.project_types.webapp_crm.parallelization = { fe_safe_enabled: true };
  const harness = createHarness(workspaceRoot, mutated);
  harness.pool.state.runs.push({ run_id: "gate-par-run", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: "gate-par-run",
    input: { goal: "Build CRM" },
  });

  await completeTask(harness, 0, writePmArtifacts);
  await completeTask(harness, 1, writeArchArtifactsFeSafe);

  assert.equal(harness.pool.state.tasks.length, 3);
  assert.match(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_be").status, /^(queued|waiting_approval)$/);
  assert.equal(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_fe").status, "pending");

  const gateEvent = harness.events.find((e) => e.event_name === "workflow.parallelization.gate_decided" && e.payload?.effective_exposure_decision_source === "rollout_master_disabled");
  assert.ok(gateEvent, "gate event with rollout_master_disabled must be emitted");
  assert.equal(gateEvent.payload?.mode, "sequential");
});
