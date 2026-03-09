import fs from "fs";
import path from "path";
import crypto from "crypto";

import { createWorkflowEngine } from "../src/workflow_engine.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow, getDefaultPromptScriptRegistryPath } from "../src/prompt_script_registry.js";
import { loadHandoffContractsOrThrow, getDefaultHandoffContractPath } from "../src/handoff_contract_registry.js";
import { analyzeTaskRisk } from "../src/policy.js";
import { resolveOrchestratorArtifactPath } from "./_paths.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

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
          workflow_run_id, run_id, workflow_id, project_type,
          status: "running", current_step_index: 0, input_json,
          error_code: null, error_message: null, last_checkpoint_id: null,
        });
        return { rows: [] };
      }
      if (text.startsWith("INSERT INTO workflow_steps")) {
        const [workflow_run_id, step_index, step_id, role_name, tool_name, gate_name] = params;
        state.workflow_steps.push({
          workflow_run_id, step_index, step_id, role_name, tool_name, gate_name,
          status: "pending", task_id: null, risk_level: null, approval_required: false,
          approval_reasons_json: "[]", result_json: null, error_code: null, checkpoint_id: null,
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
          checkpoint_id, workflow_run_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, checkpoint_json,
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
      if (text.startsWith("INSERT INTO assets") || text.startsWith("UPDATE runs SET")) return { rows: [] };
      throw new Error(`Unhandled SQL in memory pool: ${text}`);
    },
  };
}

function createHarness(workspaceRoot, registry) {
  const pool = createMemoryPool();
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

  return { engine, pool, events, workspaceRoot };
}

function createBaseRegistry() {
  return loadRegistryOrThrow(getDefaultRegistryPath());
}

function createSyntheticDagRegistry() {
  const registry = createBaseRegistry();
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

function writeArchArtifacts(rootAbs, { feSafeParallel = false } = {}) {
  writeText(path.join(rootAbs, "plan", "arch.md"), "# Module Breakdown\n\n## Interfaces\n\n## Dependency Choices\n\n## Risk Notes\n");
  writeText(path.join(rootAbs, "plan", "interfaces.md"), "# POST /api/login\n\nRequest: { email, password }\nResponse: { token }\n");
  writeText(path.join(rootAbs, "plan", "workplan.md"), "module breakdown interfaces dependency choices risk notes");
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ level: "medium", title: "auth", mitigation: "staged rollout" }],
    decision_log: ["Use Postgres"],
  });
  const handoff = {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe"],
    modules: ["auth-service"],
    interfaces: ["POST /api/login"],
    decisions: [{ adr_id: "ADR-001", title: "Use Postgres", status: "accepted" }],
    risks: ["auth migration"],
  };
  if (feSafeParallel) {
    handoff.parallelization = {
      fe_safe_parallel: true,
      requires_be_handoff: false,
      rationale: "Frontend can proceed from architect interfaces without backend handoff.",
    };
  }
  writeJson(path.join(rootAbs, "handoff", "architect_to_impl.json"), handoff);
}


async function completeTask(harness, taskIndex, writer = null) {
  const task = harness.pool.state.tasks[taskIndex];
  const payload = JSON.parse(task.payload_json);
  const rootAbs = path.resolve(harness.workspaceRoot, payload.artifact_root);
  ensureDir(rootAbs);
  if (typeof writer === "function") writer(rootAbs);
  await harness.engine.handleTaskClaimed(task.task_id);
  return harness.engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output: { artifacts: [] } });
}

async function failTask(harness, taskIndex, reason) {
  const task = harness.pool.state.tasks[taskIndex];
  await harness.engine.handleTaskClaimed(task.task_id);
  return harness.engine.handleTaskTerminal({
    task_id: task.task_id,
    status: "failed",
    error_code: "STEP_FAILED",
    output: { reason },
  });
}

async function runSequentialGateCase(baseDir) {
  const workspaceRoot = path.join(baseDir, "sequential_gate");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, createBaseRegistry());
  harness.pool.state.runs.push({ run_id: "dag-canary-seq", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: "dag-canary-seq",
    input: { goal: "Build CRM" },
  });
  await completeTask(harness, 0, writePmArtifacts);
  await completeTask(harness, 1, (rootAbs) => writeArchArtifacts(rootAbs, { feSafeParallel: true }));

  assert(harness.pool.state.tasks.length === 3, "sequential gate should dispatch BE only");
  assert(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_be")?.status === "queued", "BE should be queued");
  assert(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_fe")?.status === "pending", "FE should remain pending");
  return {
    verdict: "pass",
    tasks_dispatched: harness.pool.state.tasks.length,
    gate_reason: harness.events.find((event) => event.event_name === "workflow.parallelization.gate_decided" && event.payload?.reason_code === "PRODUCTION_WORKFLOW_SEQUENTIAL_LOCK")?.payload?.reason_code || "",
  };
}

async function runProductionSequentialLockCase(baseDir) {
  const workspaceRoot = path.join(baseDir, "production_sequential_lock");
  ensureDir(workspaceRoot);
  const registry = createBaseRegistry();
  registry.project_types.webapp_crm.parallelization = { fe_safe_enabled: true };
  const harness = createHarness(workspaceRoot, registry);
  harness.pool.state.runs.push({ run_id: "dag-canary-par", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: "dag-canary-par",
    input: { goal: "Build CRM" },
  });
  await completeTask(harness, 0, writePmArtifacts);
  await completeTask(harness, 1, (rootAbs) => writeArchArtifacts(rootAbs, { feSafeParallel: true }));

  assert(harness.pool.state.tasks.length === 3, "production workflow should remain sequential");
  assert(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_be")?.status === "queued", "BE should be queued");
  assert(harness.pool.state.workflow_steps.find((s) => s.step_id === "impl_fe")?.status === "pending", "FE should remain pending");

  return {
    verdict: "pass",
    tasks_dispatched: harness.pool.state.tasks.length,
    gate_reason: harness.events.find((event) => event.event_name === "workflow.parallelization.gate_decided" && event.payload?.reason_code === "PRODUCTION_WORKFLOW_SEQUENTIAL_LOCK")?.payload?.reason_code || "",
  };
}

async function runSyntheticParallelCase(baseDir) {
  const workspaceRoot = path.join(baseDir, "synthetic_parallel");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, createSyntheticDagRegistry());
  harness.pool.state.runs.push({ run_id: "dag-canary-synth", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "dag_matrix_v0",
    project_type: "webapp_crm",
    run_id: "dag-canary-synth",
    input: { goal: "Build CRM" },
  });
  await completeTask(harness, 0);

  assert(harness.pool.state.tasks.length === 3, "synthetic DAG should dispatch both parallel tasks after root");
  assert(harness.pool.state.workflow_steps.find((s) => s.step_id === "be_parallel")?.status === "queued", "BE parallel step should be queued");
  assert(harness.pool.state.workflow_steps.find((s) => s.step_id === "fe_parallel")?.status === "queued", "FE parallel step should be queued");

  return {
    verdict: "pass",
    tasks_dispatched: harness.pool.state.tasks.length,
    parallel_steps: ["be_parallel", "fe_parallel"],
  };
}

async function runPartialFailureCase(baseDir) {
  const workspaceRoot = path.join(baseDir, "partial_failure");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, createSyntheticDagRegistry());
  harness.pool.state.runs.push({ run_id: "dag-canary-partial", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "dag_matrix_v0",
    project_type: "webapp_crm",
    run_id: "dag-canary-partial",
    input: { goal: "Build CRM" },
  });
  await completeTask(harness, 0);
  await failTask(harness, 1, "synthetic backend failure");
  await completeTask(harness, 2);

  assert(harness.pool.state.workflow_runs[0].status === "partial_failure", "mixed result should be partial_failure");
  assert(harness.pool.state.workflow_steps.find((s) => s.step_id === "qa_join")?.status === "pending", "downstream step should be blocked");
  return {
    verdict: "pass",
    workflow_status: harness.pool.state.workflow_runs[0].status,
    blocked_step: "qa_join",
  };
}

async function runDualFailureCase(baseDir) {
  const workspaceRoot = path.join(baseDir, "dual_failure");
  ensureDir(workspaceRoot);
  const harness = createHarness(workspaceRoot, createSyntheticDagRegistry());
  harness.pool.state.runs.push({ run_id: "dag-canary-failed", status: "running" });

  await harness.engine.startWorkflowRun({
    workflow_id: "dag_matrix_v0",
    project_type: "webapp_crm",
    run_id: "dag-canary-failed",
    input: { goal: "Build CRM" },
  });
  await completeTask(harness, 0);
  await failTask(harness, 1, "synthetic backend failure");
  await failTask(harness, 2, "synthetic frontend failure");

  assert(harness.pool.state.workflow_runs[0].status === "failed", "dual failure should end as failed");
  return {
    verdict: "pass",
    workflow_status: harness.pool.state.workflow_runs[0].status,
  };
}

async function main() {
  const baseDir = resolveOrchestratorArtifactPath("canary", "workflow_dag");
  ensureDir(baseDir);

  const sequential = await runSequentialGateCase(baseDir);
  const productionLock = await runProductionSequentialLockCase(baseDir);
  const syntheticParallel = await runSyntheticParallelCase(baseDir);
  const partial = await runPartialFailureCase(baseDir);
  const dualFailure = await runDualFailureCase(baseDir);

  const reportPath = path.join(baseDir, "workflow_dag_canary.json");
  writeJson(reportPath, {
    ok: true,
    generated_at: new Date().toISOString(),
    cases: {
      sequential_gate: sequential,
      production_sequential_lock: productionLock,
      synthetic_parallel: syntheticParallel,
      partial_failure: partial,
      dual_failure: dualFailure,
    },
  });

  console.log("# Workflow DAG Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main().catch((err) => {
  console.error("[canary_workflow_dag] FAILED:", err.message || err);
  process.exit(1);
});
