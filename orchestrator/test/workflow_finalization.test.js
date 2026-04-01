import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

import { createWorkflowEngine } from "../src/workflow_engine.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow, getDefaultPromptScriptRegistryPath } from "../src/prompt_script_registry.js";
import { loadHandoffContractsOrThrow, getDefaultHandoffContractPath } from "../src/handoff_contract_registry.js";
import { analyzeTaskRisk } from "../src/policy.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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
      if (
        text.startsWith("INSERT INTO assets") ||
        text.startsWith("UPDATE runs SET") ||
        text.startsWith("INSERT INTO waterfall_stage_log") ||
        text.startsWith("INSERT INTO routing_decision_log")
      ) {
        return { rows: [] };
      }
      throw new Error(`Unhandled SQL in memory pool: ${text}`);
    },
  };
}

function createSingleStepRegistry() {
  const registry = loadRegistryOrThrow(getDefaultRegistryPath());
  const cloned = JSON.parse(JSON.stringify(registry));
  cloned.workflows.workflow_finalization_v0 = {
    project_type: "webapp_crm",
    steps: [
      { id: "pm_spec", role: "pm", tool: "coding.delegate", gate: "low" },
    ],
  };
  return cloned;
}

function createHarness(workspaceRoot, engineOverrides = {}) {
  const pool = createMemoryPool();
  const registry = createSingleStepRegistry();
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
      pool.state.tasks.push({ task_id, run_id, tool_name, payload_json: JSON.stringify(payload), risk_level: risk.risk_level });
      return { task_id, waiting_approval: Boolean(risk.requires_approval) };
    },
    recordEvent: async (stream_id, event_name, payload) => {
      events.push({ stream_id, event_name, payload });
    },
    makeIdempotencyKey: (run_id, tool_name, payload) =>
      crypto.createHash("sha256").update(JSON.stringify({ run_id, tool_name, payload })).digest("hex"),
    ...engineOverrides,
  });

  return { engine, pool, events, workspaceRoot };
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
  writeText(path.join(rootAbs, "plan", "workplan.md"), "module breakdown interfaces dependency choices risk notes");
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
  });
}

function writeBeArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "impl", "be_changes", "server.js"), "// stub backend server\nconst express = require('express');\n");
  writeText(path.join(rootAbs, "impl", "be_notes.md"), "# Backend Notes\n\nPOST /api/login implemented.\n");
  writeJson(path.join(rootAbs, "handoff", "be_to_fe.json"), {
    from_step: "impl_be",
    to_step: "impl_fe",
    be_changes_path: "impl/be_changes",
    api_contracts: [{ name: "login", method: "POST", path: "/api/login", response_shape: "{ token }" }],
    shared_types: [{ name: "User", description: "{ id, email }" }],
    scope_constraints: ["none"],
  });
}

function writeCombinedImplArtifacts(rootAbs) {
  writeBeArtifacts(rootAbs);
  writeText(
    path.join(rootAbs, "impl", "fe_changes", "app.js"),
    "// stub frontend app\nfetch('/api/login', { method: 'POST' });\n",
  );
  writeText(path.join(rootAbs, "impl", "fe_notes.md"), "# Frontend Notes\n");
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

async function completeTask(harness, taskIndex, writer) {
  const task = harness.pool.state.tasks[taskIndex];
  const payload = JSON.parse(task.payload_json);
  const rootAbs = path.resolve(harness.workspaceRoot, payload.artifact_root);
  ensureDir(rootAbs);
  writer(rootAbs);
  await harness.engine.handleTaskClaimed(task.task_id);
  return harness.engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output: { artifacts: [] } });
}

async function completeLatestTask(harness) {
  const task = harness.pool.state.tasks[harness.pool.state.tasks.length - 1];
  await harness.engine.handleTaskClaimed(task.task_id);
  return harness.engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output: { artifacts: [] } });
}

test("full workflow success transitions run to succeeded", async () => {
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_finalization/full_success");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, {
    artifactPackService: {
      generateArtifactPack: async () => ({
        ok: true,
        run_manifest_path: path.join(workspaceRoot, "artifacts", "release", "run_manifest.json"),
        summary_path: path.join(workspaceRoot, "artifacts", "release", "run_summary.md"),
        go_no_go_result_path: path.join(workspaceRoot, "artifacts", "release", "go_no_go_result.json"),
        go_no_go_verdict: "GO",
        strict_canary_report_path: path.join(workspaceRoot, "artifacts", "release", "strict_canary_report.md"),
        strict_canary_json_path: path.join(workspaceRoot, "artifacts", "release", "strict_canary_report.json"),
        strict_canary_verdict: "pass",
      }),
    },
  });
  harness.pool.state.runs.push({ run_id: "wf-finalization-success", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "workflow_finalization_v0",
    project_type: "webapp_crm",
    run_id: "wf-finalization-success",
    input: { goal: "Build CRM" },
  });

  await completeTask(harness, 0, writePmArtifacts);

  assert.equal(harness.pool.state.workflow_runs[0].status, "succeeded");
  assert.equal(harness.pool.state.runs[0].status, "completed");
});

test("artifact-pack crash fails workflow instead of leaving it running", async () => {
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_finalization/finalization_failure");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, {
    artifactPackService: {
      generateArtifactPack: async () => {
        throw new Error("synthetic artifact pack crash");
      },
    },
  });
  harness.pool.state.runs.push({ run_id: "wf-finalization-fail", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "workflow_finalization_v0",
    project_type: "webapp_crm",
    run_id: "wf-finalization-fail",
    input: { goal: "Build CRM" },
  });

  await completeTask(harness, 0, writePmArtifacts);

  assert.equal(harness.pool.state.workflow_runs[0].status, "failed");
  assert.equal(harness.pool.state.runs[0].status, "failed");
  assert.ok(harness.events.some((event) => event.event_name === "workflow.finalization.failed"));
});

test("workflow completion notification uses manifest runtime evidence summary when present", async () => {
  const workspaceRoot = path.join(__dirname, "../artifacts/test/workflow_finalization/runtime_summary");
  ensureDir(workspaceRoot);
  const transitions = [];
  const manifestPath = path.join(workspaceRoot, "artifacts", "release", "runtime-summary", "meta", "run_manifest.json");
  const summaryPath = path.join(workspaceRoot, "artifacts", "release", "runtime-summary", "summary", "run_summary.md");
  ensureDir(path.dirname(manifestPath));
  ensureDir(path.dirname(summaryPath));
  writeJson(manifestPath, {
    workflow_run_id: "wf-runtime-summary",
    status: "succeeded",
    runtime_evidence_summary: {
      smoke_verdict: "pass",
      smoke_root_status: 200,
      smoke_api_status: 200,
      superpowers_configured_steps: 2,
      superpowers_available_steps: 2,
      superpowers_steps_used: 2,
    },
  });
  writeText(summaryPath, "# Run Summary\n\n- smoke_verdict: pass\n");

  const harness = createHarness(workspaceRoot, {
    onStepTransition: async (event) => {
      transitions.push(event);
    },
    artifactPackService: {
      generateArtifactPack: async () => ({
        ok: true,
        run_manifest_path: manifestPath,
        summary_path: summaryPath,
        go_no_go_result_path: path.join(workspaceRoot, "artifacts", "release", "runtime-summary", "meta", "go_no_go_result.json"),
        go_no_go_verdict: "GO",
        strict_canary_report_path: path.join(workspaceRoot, "artifacts", "release", "runtime-summary", "summary", "strict_canary_report.md"),
        strict_canary_json_path: path.join(workspaceRoot, "artifacts", "release", "runtime-summary", "meta", "strict_canary_report.json"),
        strict_canary_verdict: "pass",
      }),
    },
  });
  harness.pool.state.runs.push({ run_id: "runtime-summary", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "workflow_finalization_v0",
    project_type: "webapp_crm",
    run_id: "runtime-summary",
    input: { goal: "Build CRM" },
  });

  await completeTask(harness, 0, writePmArtifacts);

  const completed = transitions.find((event) => event.event === "workflow.completed");
  assert.ok(completed);
  assert.match(String(completed.run_summary || ""), /smoke=pass/);
  assert.match(String(completed.run_summary || ""), /root=200/);
  assert.match(String(completed.run_summary || ""), /api=200/);
  assert.match(String(completed.run_summary || ""), /superpowers_configured_steps=2/);
  assert.match(String(completed.run_summary || ""), /superpowers_steps_used=2/);
});
