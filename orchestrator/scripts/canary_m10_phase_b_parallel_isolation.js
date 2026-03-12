#!/usr/bin/env node

import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

import { createWorkflowEngine } from "../src/workflow_engine.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow, getDefaultPromptScriptRegistryPath } from "../src/prompt_script_registry.js";
import { loadHandoffContractsOrThrow, getDefaultHandoffContractPath } from "../src/handoff_contract_registry.js";
import { analyzeTaskRisk } from "../src/policy.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const ARTIFACT_DIR = path.join(ROOT, "artifacts/canary/m10_phase_b_parallel_isolation");

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

function createConfigWorkspace() {
  const dir = fs.mkdtempSync(path.join(ROOT, "artifacts/tmp_canary_m10_phase_b_parallel_"));
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
          step.started_at = new Date().toISOString();
        }
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

function getTaskPayloadByStep(harness, workflowRunId, stepId) {
  const step = harness.pool.state.workflow_steps.find(
    (item) => item.workflow_run_id === workflowRunId && String(item.step_id || "") === String(stepId || "")
  );
  assert(step?.task_id, `task missing for step ${stepId}`);
  const task = harness.pool.state.tasks.find((item) => item.task_id === step.task_id);
  assert(task, `task record missing for step ${stepId}`);
  return JSON.parse(task.payload_json);
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
  writeText(path.join(rootAbs, "plan", "interfaces.md"), "# GET /api/customers\n\nResponse: { items: [] }\n");
  writeText(path.join(rootAbs, "plan", "workplan.md"), "module breakdown interfaces dependency choices risk notes");
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ level: "medium", title: "coordination", mitigation: "strict path isolation" }],
    decision_log: ["Use isolated FE/BE target paths"],
  });
  writeJson(path.join(rootAbs, "handoff", "architect_to_impl.json"), {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe"],
    modules: ["customer-api", "customer-ui"],
    interfaces: ["GET /api/customers"],
    decisions: [{ adr_id: "ADR-M10-B", title: "Parallel FE/BE with isolated files", status: "accepted" }],
    risks: ["frontend/backend contract drift"],
    parallelization: {
      fe_safe_parallel: true,
      requires_be_handoff: false,
      rationale: "Frontend and backend operate on isolated files with fixed target_paths.",
    },
  });
}

async function completeTask(harness, taskIndex, writer = null) {
  const task = harness.pool.state.tasks[taskIndex];
  const payload = JSON.parse(task.payload_json);
  const artifactRootAbs = path.resolve(harness.workspaceRoot, payload.artifact_root);
  ensureDir(artifactRootAbs);
  if (typeof writer === "function") writer(artifactRootAbs);
  await harness.engine.handleTaskClaimed(task.task_id);
  return harness.engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output: { artifacts: [] } });
}

async function runCanary() {
  const workspaceRoot = createConfigWorkspace();
  try {
    const harness = createHarness(workspaceRoot);
    const runId = "m10-canary-b-run";
    harness.pool.state.runs.push({ run_id: runId, status: "running" });

    const started = await harness.engine.startWorkflowRun({
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      run_id: runId,
      input: {
        goal: "Execute FE + BE parallel task with isolated target paths",
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

    await completeTask(harness, 0, writePmArtifacts);
    await completeTask(harness, 1, writeArchArtifacts);

    const workflowRunId = started.workflow_run_id;
    const beStep = harness.pool.state.workflow_steps.find(
      (item) => item.workflow_run_id === workflowRunId && item.step_id === "impl_be"
    );
    const feStep = harness.pool.state.workflow_steps.find(
      (item) => item.workflow_run_id === workflowRunId && item.step_id === "impl_fe"
    );
    assert(beStep?.status === "queued", `impl_be should be queued, got ${beStep?.status}`);
    assert(feStep?.status === "queued", `impl_fe should be queued, got ${feStep?.status}`);

    const bePayload = getTaskPayloadByStep(harness, workflowRunId, "impl_be");
    const fePayload = getTaskPayloadByStep(harness, workflowRunId, "impl_fe");
    const beTargets = Array.isArray(bePayload.target_paths) ? bePayload.target_paths : [];
    const feTargets = Array.isArray(fePayload.target_paths) ? fePayload.target_paths : [];
    const overlap = beTargets.filter((item) => feTargets.includes(item));
    assert(beTargets.length === 1, `impl_be target_paths expected 1 entry, got ${beTargets.length}`);
    assert(feTargets.length === 1, `impl_fe target_paths expected 1 entry, got ${feTargets.length}`);
    assert(overlap.length === 0, `target_paths overlap detected: ${overlap.join(", ")}`);

    const gateEvent = harness.events.find(
      (item) =>
        item.event_name === "workflow.parallelization.gate_decided" &&
        item.payload?.workflow_run_id === workflowRunId
    );
    assert(gateEvent, "parallelization gate event missing");
    assert(gateEvent.payload?.reason_code === "GATED_PARALLEL_ALLOWED", `unexpected gate reason: ${gateEvent.payload?.reason_code}`);
    assert(gateEvent.payload?.routing_decision_source === "classifier_recommended_parallel", `unexpected routing source: ${gateEvent.payload?.routing_decision_source}`);

    const routingDecision = harness.pool.state.routing_decision_log.find((item) => item.workflow_run_id === workflowRunId);
    assert(routingDecision, "routing decision audit log missing");
    assert(routingDecision.final_execution_decision === "gated_parallel_allowed", `unexpected final execution decision: ${routingDecision.final_execution_decision}`);

    const policyStage = harness.pool.state.waterfall_stage_log.find(
      (item) => item.run_id === runId && item.stage === "policy_evaluation"
    );
    assert(policyStage, "policy_evaluation waterfall stage missing");
    assert(String(policyStage.workflow_run_id || "") === workflowRunId, "policy_evaluation workflow_run_id mismatch");

    return {
      workflow_run_id: workflowRunId,
      run_id: runId,
      queued_steps: ["impl_be", "impl_fe"],
      impl_be_target_paths: beTargets,
      impl_fe_target_paths: feTargets,
      routing_decision_source: gateEvent.payload.routing_decision_source,
      routing_final_execution_decision: routingDecision.final_execution_decision,
      waterfall_stage: {
        stage: policyStage.stage,
        run_id: policyStage.run_id,
        workflow_run_id: policyStage.workflow_run_id,
      },
    };
  } finally {
    cleanup(workspaceRoot);
  }
}

async function main() {
  const result = await runCanary();
  const artifact = {
    canary: "canary_m10_phase_b_parallel_isolation",
    milestone: "m10_phase_b",
    timestamp: new Date().toISOString(),
    status: "PASS",
    result,
  };
  writeJson(path.join(ARTIFACT_DIR, "canary_m10_phase_b_parallel_isolation.json"), artifact);
  console.log();
  console.log("M10 Phase B Parallel Isolation Canary: PASS");
  console.log("Artifact: orchestrator/artifacts/canary/m10_phase_b_parallel_isolation/canary_m10_phase_b_parallel_isolation.json");
}

main().catch((err) => {
  console.error(`FAIL  ${err?.stack || err?.message || err}`);
  process.exit(1);
});
