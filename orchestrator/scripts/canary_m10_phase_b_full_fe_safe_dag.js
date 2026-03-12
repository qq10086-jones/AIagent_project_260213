#!/usr/bin/env node

import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

import { createWorkflowEngine } from "../src/workflow_engine.js";
import { createWaterfallTraceService } from "../src/domain/waterfall_trace_service.js";
import { createRoutingAuditLogService } from "../src/domain/routing_audit_log.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow, getDefaultPromptScriptRegistryPath } from "../src/prompt_script_registry.js";
import { loadHandoffContractsOrThrow, getDefaultHandoffContractPath } from "../src/handoff_contract_registry.js";
import { analyzeTaskRisk } from "../src/policy.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const ARTIFACT_DIR = path.join(ROOT, "artifacts/canary/m10_phase_b_full_fe_safe_dag");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function writeText(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, value, "utf8");
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeContextBudgetReport(rootAbs, stepId) {
  const relPath = `metrics/context_budget_${stepId}.json`;
  writeJson(path.join(rootAbs, relPath), {
    step_id: stepId,
    artifact_count: 1,
    bytes_total: 128,
    largest_artifact_bytes: 128,
    prompt_chars: 64,
    injected_context_bytes: 0,
    status: "ok",
    threshold_source: "default_thresholds",
  });
  return {
    context_budget_report_path: relPath,
    context_budget_report: {
      status: "ok",
      threshold_source: "default_thresholds",
    },
  };
}

function createConfigWorkspace() {
  const dir = fs.mkdtempSync(path.join(ROOT, "artifacts/tmp_canary_m10_phase_b_full_"));
  const configDir = path.join(dir, "configs");
  fs.mkdirSync(configDir, { recursive: true });
  fs.writeFileSync(
    path.join(configDir, "production_parallel_rollout.json"),
    JSON.stringify({
      master_enabled: true,
      force_sequential: false,
      dynamic_routing_enabled: true,
      router_mode: "dynamic_routing_enforced",
      circuit_breaker: { activated: false },
      classifier_circuit_breaker: { activated: false },
      last_policy_change: new Date().toISOString(),
    })
  );
  fs.writeFileSync(
    path.join(configDir, "parallel_exposure_policy.json"),
    JSON.stringify({
      allowed_workflow_types: ["coding_team_v0"],
      allowed_project_types: ["webapp_crm"],
      fe_safe_eligible_input_classes: ["fe_led"],
    })
  );
  fs.writeFileSync(
    path.join(configDir, "m7_exposure_cohorts.json"),
    JSON.stringify({
      allowed_workflow_types: ["coding_team_v0"],
      allowed_project_types: ["webapp_crm"],
      allowed_input_classes: ["fe_led"],
      runtime_controls: {
        cohort_enabled: true,
        environment: "staging",
      },
    })
  );
  return dir;
}

function cleanup(dirPath) {
  fs.rmSync(dirPath, { recursive: true, force: true });
}

function createMemoryPool() {
  const state = {
    workflow_runs: [],
    workflow_steps: [],
    tasks: [],
    runs: [],
    workflow_checkpoints: [],
    routing_decision_log: [],
    waterfall_stage_log: [],
  };

  function findRun(workflowRunId) {
    return state.workflow_runs.find((item) => item.workflow_run_id === workflowRunId) || null;
  }

  function findStep(workflowRunId, stepIndex) {
    return state.workflow_steps.find(
      (item) => item.workflow_run_id === workflowRunId && Number(item.step_index) === Number(stepIndex)
    ) || null;
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
          created_at: new Date().toISOString(),
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
          started_at: null,
          ended_at: null,
        });
        return { rows: [] };
      }

      if (text === "SELECT * FROM workflow_runs WHERE workflow_run_id=$1") {
        const row = findRun(params[0]);
        return { rows: row ? [row] : [] };
      }

      if (text === "SELECT workflow_run_id FROM workflow_runs WHERE run_id=$1 ORDER BY created_at DESC LIMIT 1") {
        const row = [...state.workflow_runs].reverse().find((item) => item.run_id === params[0]) || null;
        return { rows: row ? [{ workflow_run_id: row.workflow_run_id }] : [] };
      }

      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 ORDER BY step_index ASC") {
        return {
          rows: state.workflow_steps
            .filter((item) => item.workflow_run_id === params[0])
            .sort((a, b) => Number(a.step_index) - Number(b.step_index)),
        };
      }

      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2") {
        const row = findStep(params[0], params[1]);
        return { rows: row ? [row] : [] };
      }

      if (text.startsWith("SELECT step_id, started_at, ended_at, status FROM workflow_steps")) {
        return {
          rows: state.workflow_steps
            .filter(
              (item) =>
                item.workflow_run_id === params[0] &&
                ["impl_be", "impl_fe", "qa_verify", "release_pack"].includes(String(item.step_id || ""))
            )
            .sort((a, b) => new Date(a.started_at || 0) - new Date(b.started_at || 0))
            .map((item) => ({
              step_id: item.step_id,
              started_at: item.started_at,
              ended_at: item.ended_at,
              status: item.status,
            })),
        };
      }

      if (text.startsWith("UPDATE workflow_steps SET status=$3, task_id=$4")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = params[2];
          step.task_id = params[3];
          step.risk_level = params[4];
          step.approval_required = params[5];
          step.approval_reasons_json = params[6];
          if (!step.started_at) step.started_at = new Date().toISOString();
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

      if (text === "SELECT payload_json FROM tasks WHERE task_id=$1") {
        const task = state.tasks.find((item) => item.task_id === params[0]) || null;
        return { rows: task ? [{ payload_json: task.payload_json }] : [] };
      }

      if (text === "SELECT task_id, run_id, payload_json FROM tasks WHERE task_id=$1") {
        const task = state.tasks.find((item) => item.task_id === params[0]) || null;
        return { rows: task ? [task] : [] };
      }

      if (text.startsWith("UPDATE workflow_steps SET status='running'")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = "running";
          if (!step.started_at) step.started_at = new Date().toISOString();
        }
        return { rows: [] };
      }

      if (text.startsWith("UPDATE workflow_steps SET status='queued'")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = "queued";
          if (!step.started_at) step.started_at = new Date().toISOString();
        }
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
          step.ended_at = new Date().toISOString();
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
          step.ended_at = new Date().toISOString();
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

      if (text.startsWith("UPDATE workflow_runs SET status='partial_failure', error_code=$2, error_message=$3")) {
        const run = findRun(params[0]);
        if (run) {
          run.status = "partial_failure";
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
          step.ended_at = new Date().toISOString();
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
            .filter((item) => item.workflow_run_id === params[0])
            .sort((a, b) => Number(a.step_index) - Number(b.step_index)),
        };
      }

      if (text.startsWith("INSERT INTO routing_decision_log")) {
        state.routing_decision_log.push({
          log_id: params[0],
          run_id: params[1],
          workflow_run_id: params[2],
          workflow_id: params[3],
          router_mode: params[4],
          dynamic_routing_enabled: params[5],
          classifier_version: params[6],
          classifier_confidence: params[7],
          classifier_confidence_band: params[8],
          classifier_work_shape: params[9],
          classifier_domain_lead: params[10],
          classifier_parallel_candidate: params[11],
          classifier_model_tier: params[12],
          classifier_deny_or_degrade_reason: params[13],
          feature_snapshot_ref: params[14],
          routing_decision_source: params[15],
          final_execution_decision: params[16],
          safety_override_result: params[17],
          decision_json: params[18],
          created_at: new Date().toISOString(),
        });
        return { rows: [] };
      }

      if (text.startsWith("SELECT * FROM routing_decision_log WHERE run_id=$1")) {
        return {
          rows: state.routing_decision_log
            .filter((item) => item.run_id === params[0])
            .sort((a, b) => new Date(a.created_at) - new Date(b.created_at)),
        };
      }

      if (text.startsWith("SELECT * FROM routing_decision_log WHERE workflow_run_id=$1")) {
        const rows = state.routing_decision_log
          .filter((item) => item.workflow_run_id === params[0])
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        return { rows: rows.slice(0, 1) };
      }

      if (text.startsWith("INSERT INTO waterfall_stage_log")) {
        state.waterfall_stage_log.push({
          run_id: params[0],
          workflow_run_id: params[1],
          stage: params[2],
          started_at: params[3],
          ended_at: params[4],
          duration_ms: params[5],
          metadata_json: params[6],
        });
        return { rows: [] };
      }

      if (text.startsWith("SELECT * FROM waterfall_stage_log WHERE run_id=$1")) {
        return {
          rows: state.waterfall_stage_log
            .filter((item) => item.run_id === params[0])
            .sort((a, b) => new Date(a.started_at) - new Date(b.started_at)),
        };
      }

      if (text.startsWith("INSERT INTO assets") || text.startsWith("UPDATE runs SET")) {
        return { rows: [] };
      }

      throw new Error(`Unhandled SQL in memory pool: ${text}`);
    },
  };
}

function createHarness(workspaceRoot) {
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
      pool.state.tasks.push({
        task_id,
        run_id,
        tool_name,
        payload_json: JSON.stringify(payload),
      });
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

function getTaskForStep(harness, workflowRunId, stepId) {
  const step = harness.pool.state.workflow_steps.find(
    (item) => item.workflow_run_id === workflowRunId && String(item.step_id || "") === String(stepId || "")
  );
  assert(step?.task_id, `task missing for step ${stepId}`);
  const task = harness.pool.state.tasks.find((item) => item.task_id === step.task_id);
  assert(task, `task record missing for step ${stepId}`);
  return {
    step,
    task,
    payload: JSON.parse(task.payload_json),
  };
}

function writePmArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "spec.md"), "# Scope\n\n## User Stories\n\n## Acceptance Criteria\n\n## Non-Goals\n\n## Artifact List\n");
  writeText(path.join(rootAbs, "plan", "milestones.md"), "scope user_stories acceptance_criteria non_goals artifact_list");
  writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
    criteria: ["AC-001", "AC-002"],
    artifacts: ["plan/spec.md", "plan/milestones.md"],
    owner: "pm_agent",
    version: "v1",
  });
  writeJson(path.join(rootAbs, "handoff", "pm_to_architect.json"), {
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "CRM customer dashboard MVP",
    artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    acceptance: { criteria: ["AC-001", "AC-002"] },
  });
}

function writeArchArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "arch.md"), "# Module Breakdown\n\n## Interfaces\n\n## Dependency Choices\n\n## Risk Notes\n");
  writeText(path.join(rootAbs, "plan", "interfaces.md"), "# GET /api/customers\n\nResponse: { items: [] }\n");
  writeText(path.join(rootAbs, "plan", "workplan.md"), "module breakdown interfaces dependency choices risk notes");
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ level: "medium", title: "parallel coordination", mitigation: "strict path isolation and QA gate" }],
    decision_log: ["Use fixed FE/BE target paths and merge only via release pack"],
  });
  writeJson(path.join(rootAbs, "handoff", "architect_to_impl.json"), {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe"],
    modules: ["customer-api", "customer-ui"],
    interfaces: ["GET /api/customers"],
    decisions: [{ adr_id: "ADR-M10-T23C", title: "Full fe_safe DAG enforced canary", status: "accepted" }],
    risks: ["frontend/backend contract drift"],
    parallelization: {
      fe_safe_parallel: true,
      requires_be_handoff: false,
      rationale: "Frontend and backend operate on disjoint paths and converge through QA plus release evidence.",
    },
  });
}

function writeBackendArtifacts(rootAbs) {
  writeJson(path.join(rootAbs, "impl", "be_patch_bundle.json"), {
    bundle_id: "m10-t23c-be-bundle",
    step_id: "impl_be",
    mode: "full_file_fallback",
    operations: [
      {
        type: "create_file",
        target_file: "sandbox/crm_site/server.js",
        file_content: "export function listCustomers() { return []; }\n",
      },
    ],
    target_files: ["sandbox/crm_site/server.js"],
    summary: "Mock backend patch bundle for full fe_safe DAG canary.",
  });
  writeText(path.join(rootAbs, "impl", "be_changes", "server.js"), "export function listCustomers() { return []; }\n");
  writeText(
    path.join(rootAbs, "impl", "be_notes.md"),
    "# Backend Notes\n\n## API Contracts\n\n- GET /api/customers\n\n## Shared Types\n\n- Customer\n\n## Scope Constraints\n\n- Only sandbox/crm_site/server.js is modified.\n"
  );
  writeJson(path.join(rootAbs, "handoff", "be_to_fe.json"), {
    from_step: "impl_be",
    to_step: "impl_fe",
    be_changes_path: "impl/be_changes",
    api_contracts: [{ name: "List Customers", method: "GET", path: "/api/customers", response_shape: "{ items: [] }" }],
    shared_types: [{ name: "Customer", description: "CRM customer record." }],
    scope_constraints: ["Read-only stub for enforced canary."],
  });
}

function writeFrontendArtifacts(rootAbs) {
  writeJson(path.join(rootAbs, "impl", "fe_patch_bundle.json"), {
    bundle_id: "m10-t23c-fe-bundle",
    step_id: "impl_fe",
    mode: "full_file_fallback",
    operations: [
      {
        type: "create_file",
        target_file: "sandbox/crm_site/app.js",
        file_content: "export function CustomerList() { return null; }\n",
      },
    ],
    target_files: ["sandbox/crm_site/app.js"],
    summary: "Mock frontend patch bundle for full fe_safe DAG canary.",
  });
  writeText(path.join(rootAbs, "impl", "fe_changes", "app.js"), "export function CustomerList() { return null; }\n");
  writeText(
    path.join(rootAbs, "impl", "fe_notes.md"),
    "# Frontend Notes\n\n## UI Scope\n\n- Customer list placeholder\n\n## API Consumption\n\n- Consume GET /api/customers from handoff/be_to_fe.json\n\n## Run Instructions\n\n1. Render empty customer list.\n"
  );
  writeJson(path.join(rootAbs, "handoff", "be_to_fe.json"), {
    from_step: "impl_be",
    to_step: "impl_fe",
    be_changes_path: "impl/be_changes",
    api_contracts: [{ name: "List Customers", method: "GET", path: "/api/customers", response_shape: "{ items: [] }" }],
    shared_types: [{ name: "Customer", description: "CRM customer record." }],
    scope_constraints: ["Read-only stub for enforced canary."],
  });
  writeJson(path.join(rootAbs, "handoff", "impl_to_qa.json"), {
    from_steps: ["impl_be", "impl_fe"],
    to_step: "qa_verify",
    be_changes_path: "impl/be_changes",
    fe_changes_path: "impl/fe_changes",
    run_instructions: "Start backend, render customer list view, verify empty state.",
    known_limitations: ["Synthetic canary artifacts only"],
    api_contracts_path: "handoff/be_to_fe.json",
  });
}

function writeCombinedImplArtifacts(rootAbs) {
  writeBackendArtifacts(rootAbs);
  writeFrontendArtifacts(rootAbs);
}

function writeQaArtifacts(rootAbs, runId, workflowRunId) {
  writeJson(path.join(rootAbs, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [
      { check_id: "D-01", layer: "deterministic", description: "BE artifacts present", status: "pass", detail: "impl/be_changes/server.js present" },
      { check_id: "D-02", layer: "deterministic", description: "FE artifacts present", status: "pass", detail: "impl/fe_changes/app.js present" },
      { check_id: "S-01", layer: "semantic", description: "API contract aligns with FE placeholder", status: "pass", detail: "GET /api/customers contract matched" },
    ],
    verified_artifacts: ["A1", "A2"],
    generated_at: new Date().toISOString(),
    step_id: "qa_verify",
  });
  writeJson(path.join(rootAbs, "handoff", "qa_to_release.json"), {
    from_step: "qa_verify",
    to_step: "release_pack",
    qa_report_path: "verify/qa_report.json",
    overall_status: "pass",
    verified_artifacts: ["impl/be_changes/server.js", "impl/fe_changes/app.js"],
    run_id: runId,
    workflow_run_id: workflowRunId,
  });
}

function writeReleaseArtifacts(rootAbs, runId) {
  writeText(
    path.join(rootAbs, "release", "release_notes.md"),
    `# Release Notes\n\nCustomer dashboard stub delivered and verified.\n\n## What changed\n\n- backend customer list endpoint stub\n- frontend customer list placeholder\n\n## QA Status\n\nAll checks passed.\n\nRun ID: ${runId}\n`
  );
  writeJson(path.join(rootAbs, "release", "artifact_manifest.json"), {
    run_id: runId,
    workflow_id: "coding_team_v0",
    completed_at: new Date().toISOString(),
    artifacts: [
      { path: "plan/spec.md", type: "markdown", size_bytes: 120 },
      { path: "plan/acceptance.json", type: "json", size_bytes: 100 },
      { path: "risk/risk_report.json", type: "json", size_bytes: 110 },
      { path: "verify/qa_report.json", type: "json", size_bytes: 240 },
      { path: "release/release_notes.md", type: "markdown", size_bytes: 220 },
    ],
  });
}

async function completeTaskByStep(harness, workflowRunId, stepId, writer = null, output = {}) {
  const { task, payload } = getTaskForStep(harness, workflowRunId, stepId);
  const artifactRootAbs = path.resolve(harness.workspaceRoot, payload.artifact_root);
  ensureDir(artifactRootAbs);
  if (typeof writer === "function") writer(artifactRootAbs);
  const contextBudgetOutput = writeContextBudgetReport(artifactRootAbs, stepId);
  await harness.engine.handleTaskClaimed(task.task_id);
  return harness.engine.handleTaskTerminal({
    task_id: task.task_id,
    status: "succeeded",
    output: { ...output, ...contextBudgetOutput },
  });
}

async function runCanary() {
  const workspaceRoot = createConfigWorkspace();
  try {
    const harness = createHarness(workspaceRoot);
    const runId = "m10-t23c-run";
    harness.pool.state.runs.push({ run_id: runId, status: "running" });

    const started = await harness.engine.startWorkflowRun({
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      run_id: runId,
      input: {
        goal: "Execute full fe_safe DAG with QA and release-pack tracing",
        input_class: "fe_led",
        task_envelope: {
          classifier_result: {
            confidence_band: "high",
            final_execution_decision: "gated_parallel_allowed",
            model_tier: "balanced_default",
            domain_lead: "fe_led",
            work_shape: "dual_branch_parallel_candidate",
            parallel_candidate: true,
          },
        },
        step_payloads: {
          impl_be: {
            target_paths: ["sandbox/crm_site/server.js"],
            opencode_command: ["mock-inline-autofix", "sandbox/crm_site/server.js", "{{task_prompt}}"],
          },
          impl_fe: {
            target_paths: ["sandbox/crm_site/app.js"],
            opencode_command: ["mock-inline-autofix", "sandbox/crm_site/app.js", "{{task_prompt}}"],
          },
        },
      },
    });

    const workflowRunId = started.workflow_run_id;
    await completeTaskByStep(harness, workflowRunId, "pm_spec", writePmArtifacts);
    await completeTaskByStep(harness, workflowRunId, "arch_design", writeArchArtifacts);

    const beStep = harness.pool.state.workflow_steps.find(
      (item) => item.workflow_run_id === workflowRunId && item.step_id === "impl_be"
    );
    const feStep = harness.pool.state.workflow_steps.find(
      (item) => item.workflow_run_id === workflowRunId && item.step_id === "impl_fe"
    );
    assert(beStep?.status === "queued", `impl_be should be queued, got ${beStep?.status}`);
    assert(feStep?.status === "queued", `impl_fe should be queued, got ${feStep?.status}`);

    await completeTaskByStep(harness, workflowRunId, "impl_be", writeBackendArtifacts, {
      files_changed: ["sandbox/crm_site/server.js"],
      diff_stats: { files: 1 },
    });
    await completeTaskByStep(harness, workflowRunId, "impl_fe", writeCombinedImplArtifacts, {
      files_changed: ["sandbox/crm_site/app.js"],
      diff_stats: { files: 1 },
    });

    const qaStep = harness.pool.state.workflow_steps.find(
      (item) => item.workflow_run_id === workflowRunId && item.step_id === "qa_verify"
    );
    assert(
      qaStep?.task_id,
      `qa_verify should be dispatched after both impl steps: ${JSON.stringify(
        harness.pool.state.workflow_steps
          .filter((item) => item.workflow_run_id === workflowRunId)
          .map((item) => ({ step_id: item.step_id, status: item.status, error_code: item.error_code }))
      )}`
    );
    await completeTaskByStep(
      harness,
      workflowRunId,
      "qa_verify",
      (rootAbs) => writeQaArtifacts(rootAbs, runId, workflowRunId)
    );
    await completeTaskByStep(
      harness,
      workflowRunId,
      "release_pack",
      (rootAbs) => writeReleaseArtifacts(rootAbs, runId)
    );

    const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === workflowRunId);
    assert(
      workflowRun?.status === "succeeded",
      `workflow should succeed, got ${workflowRun?.status} error=${workflowRun?.error_code || ""}: ${JSON.stringify(
        harness.pool.state.workflow_steps
          .filter((item) => item.workflow_run_id === workflowRunId)
          .map((item) => ({ step_id: item.step_id, status: item.status, error_code: item.error_code }))
      )}`
    );
    const parentRun = harness.pool.state.runs.find((item) => item.run_id === runId);
    assert(parentRun?.status === "completed", `parent run should be completed, got ${parentRun?.status}`);

    const routingAuditLogService = createRoutingAuditLogService({ pool: harness.pool, workspaceRoot });
    const waterfallTraceService = createWaterfallTraceService({ pool: harness.pool });
    const routingRows = await routingAuditLogService.queryByRunId(runId);
    const latestRouting = await routingAuditLogService.getLatestForWorkflowRun(workflowRunId);
    const waterfallReport = await waterfallTraceService.buildWaterfallReport(runId);

    assert(routingRows.length >= 1, "routing_decision_log queryByRunId returned no rows");
    assert(latestRouting?.final_execution_decision === "gated_parallel_allowed", "routing decision should remain gated_parallel_allowed");

    const stageNames = waterfallReport.stages.map((item) => item.stage);
    for (const expected of ["policy_evaluation", "branch_completion_be", "branch_completion_fe", "qa_admission", "release_pack_readiness"]) {
      assert(stageNames.includes(expected), `waterfall report missing ${expected}`);
    }

    const releaseRoot = path.join(workspaceRoot, "artifacts", "release", runId);
    const manifestPath = path.join(releaseRoot, "meta", "run_manifest.json");
    const summaryPath = path.join(releaseRoot, "summary", "run_summary.md");
    const goNoGoPath = path.join(releaseRoot, "qa", "go_no_go_result.json");
    const strictCanaryPath = path.join(releaseRoot, "qa", "strict_canary_report.json");
    assert(fs.existsSync(manifestPath), "run manifest missing");
    assert(fs.existsSync(summaryPath), "run summary missing");
    assert(fs.existsSync(goNoGoPath), "go_no_go_result missing");
    assert(fs.existsSync(strictCanaryPath), "strict_canary_report missing");
    assert(fs.existsSync(path.join(releaseRoot, "verify", "qa_report.json")), "qa_report not preserved in release pack");
    assert(fs.existsSync(path.join(releaseRoot, "release", "release_notes.md")), "release_notes not preserved in release pack");

    const manifest = readJson(manifestPath);
    const goNoGo = readJson(goNoGoPath);
    const strictCanary = readJson(strictCanaryPath);
    assert(String(goNoGo.verdict || "") === "GO", `go/no-go should be GO, got ${goNoGo.verdict}`);
    assert(String(strictCanary.verdict || "") === "pass", `strict canary should pass, got ${strictCanary.verdict}`);
    assert(Array.isArray(manifest.steps) && manifest.steps.length === 6, `manifest should contain 6 steps, got ${manifest.steps?.length}`);
    assert(Number(manifest?.coding_execution_summary?.total_steps || 0) >= 2, "coding execution evidence should include impl steps");

    return {
      run_id: runId,
      workflow_run_id: workflowRunId,
      routing_decision_source: latestRouting.routing_decision_source,
      final_execution_decision: latestRouting.final_execution_decision,
      waterfall_stage_names: stageNames,
      go_no_go_verdict: goNoGo.verdict,
      strict_canary_verdict: strictCanary.verdict,
      manifest_step_count: manifest.steps.length,
      release_artifacts: {
        run_manifest: path.relative(ROOT, manifestPath).replace(/\\/g, "/"),
        run_summary: path.relative(ROOT, summaryPath).replace(/\\/g, "/"),
        go_no_go_result: path.relative(ROOT, goNoGoPath).replace(/\\/g, "/"),
      },
    };
  } finally {
    cleanup(workspaceRoot);
  }
}

async function main() {
  const result = await runCanary();
  const artifact = {
    canary: "canary_m10_phase_b_full_fe_safe_dag",
    milestone: "m10_phase_b",
    timestamp: new Date().toISOString(),
    status: "PASS",
    result,
  };
  writeJson(path.join(ARTIFACT_DIR, "canary_m10_phase_b_full_fe_safe_dag.json"), artifact);
  console.log();
  console.log("M10 Phase B Full fe_safe DAG Canary: PASS");
  console.log("Artifact: orchestrator/artifacts/canary/m10_phase_b_full_fe_safe_dag/canary_m10_phase_b_full_fe_safe_dag.json");
}

main().catch((err) => {
  console.error(`FAIL  ${err?.stack || err?.message || err}`);
  process.exit(1);
});
