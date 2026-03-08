import fs from "fs";
import path from "path";
import crypto from "crypto";
import { createWorkflowEngine } from "../src/workflow_engine.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow, getDefaultPromptScriptRegistryPath } from "../src/prompt_script_registry.js";
import { loadHandoffContractsOrThrow, getDefaultHandoffContractPath } from "../src/handoff_contract_registry.js";
import { analyzeTaskRisk } from "../src/policy.js";
import { resolveOrchestratorArtifactPath } from "./_paths.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeJson(absPath, value) {
  ensureDir(path.dirname(absPath));
  fs.writeFileSync(absPath, JSON.stringify(value, null, 2), "utf8");
}

function writeText(absPath, value) {
  ensureDir(path.dirname(absPath));
  fs.writeFileSync(absPath, value, "utf8");
}

function createMemoryPool() {
  const state = {
    workflow_runs: [],
    workflow_steps: [],
    tasks: [],
    runs: [],
    workflow_checkpoints: [],
  };

  function findRun(workflow_run_id) {
    return state.workflow_runs.find((item) => item.workflow_run_id === workflow_run_id) || null;
  }

  function findStep(workflow_run_id, step_index) {
    return state.workflow_steps.find(
      (item) => item.workflow_run_id === workflow_run_id && Number(item.step_index) === Number(step_index)
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
        const run = findRun(params[0]);
        return { rows: run ? [run] : [] };
      }

      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 ORDER BY step_index ASC") {
        const rows = state.workflow_steps
          .filter((item) => item.workflow_run_id === params[0])
          .sort((a, b) => Number(a.step_index) - Number(b.step_index));
        return { rows };
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

      if (text.startsWith("UPDATE workflow_steps SET status='failed', error_code='APPROVAL_REJECTED', result_json=$3")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = "failed";
          step.result_json = params[2];
          step.error_code = "APPROVAL_REJECTED";
        }
        return { rows: [] };
      }

      if (text === "UPDATE runs SET status='failed' WHERE run_id=$1") {
        const run = state.runs.find((item) => item.run_id === params[0]);
        if (run) run.status = "failed";
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
        const rows = state.workflow_checkpoints
          .filter((item) => item.workflow_run_id === params[0])
          .sort((a, b) => Number(a.step_index) - Number(b.step_index));
        return { rows };
      }

      throw new Error(`Unhandled SQL in memory pool: ${text}`);
    },
  };
}

function createEngineHarness(workspaceRoot, registryOverride = null) {
  const pool = createMemoryPool();
  const registry = registryOverride || loadRegistryOrThrow(getDefaultRegistryPath());
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

  return { engine, pool, events };
}

async function runPermissionGateFailureCase({ name, workspaceRoot }) {
  const registry = loadRegistryOrThrow(getDefaultRegistryPath());
  const mutatedRegistry = JSON.parse(JSON.stringify(registry));
  const pmStep = mutatedRegistry.workflows?.coding_team_v0?.steps?.find((item) => String(item.id || "") === "pm_spec");
  if (!pmStep) throw new Error(`${name}: pm_spec step missing`);
  pmStep.tool = "bash.execute";

  const harness = createEngineHarness(workspaceRoot, mutatedRegistry);
  harness.pool.state.runs.push({ run_id: `${name}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${name}-run`,
    input: { goal: "Build CRM MVP", provider: "opencode", model: "qwen-coder-next" },
  });

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  const pmStepState = harness.pool.state.workflow_steps.find(
    (item) => item.workflow_run_id === started.workflow_run_id && item.step_id === "pm_spec"
  );

  assertEqual(workflowRun.status, "failed", `${name}.workflow.status`);
  assertEqual(workflowRun.error_code, "TOOL_PERMISSION_DENIED", `${name}.workflow.error_code`);
  assertEqual(pmStepState.status, "failed", `${name}.pm_step.status`);
  assertEqual(pmStepState.error_code, "TOOL_PERMISSION_DENIED", `${name}.pm_step.error_code`);

  const deniedEvent = harness.events.find((item) => item.event_name === "policy.tool_permission.denied");
  if (!deniedEvent) {
    throw new Error(`${name}.policy.tool_permission.denied missing`);
  }

  return {
    workflow_run_id: started.workflow_run_id,
    workflow_status: workflowRun.status,
    error_code: workflowRun.error_code,
    step_status: pmStepState.status,
    denied_event: deniedEvent,
  };
}

async function runApprovalGateCase({ name, workspaceRoot }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${name}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${name}-run`,
    input: { goal: "Update .env secrets and credentials handling for CRM MVP", provider: "opencode", model: "qwen-coder-next" },
  });

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  const pmStepState = harness.pool.state.workflow_steps.find(
    (item) => item.workflow_run_id === started.workflow_run_id && item.step_id === "pm_spec"
  );

  assertEqual(workflowRun.status, "running", `${name}.workflow.status`);
  assertEqual(pmStepState.status, "waiting_approval", `${name}.pm_step.status`);
  assertEqual(Boolean(pmStepState.approval_required), true, `${name}.pm_step.approval_required`);
  assertEqual(Boolean(started.first_step?.waiting_approval), true, `${name}.first_step.waiting_approval`);

  const approvalEvent = harness.events.find((item) => item.event_name === "policy.gate.checked");
  if (!approvalEvent) {
    throw new Error(`${name}.policy.gate.checked missing`);
  }

  return {
    workflow_run_id: started.workflow_run_id,
    workflow_status: workflowRun.status,
    step_status: pmStepState.status,
    approval_required: pmStepState.approval_required,
    first_step: started.first_step,
    approval_event: approvalEvent,
  };
}

async function runApprovalApprovedCase({ name, workspaceRoot }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${name}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${name}-run`,
    input: { goal: "Update .env secrets and credentials handling for CRM MVP", provider: "opencode", model: "qwen-coder-next" },
  });

  const taskId = started.first_step?.task_id;
  if (!taskId) throw new Error(`${name}: first approval task missing`);

  const approved = await harness.engine.handleTaskApproved(taskId);
  assertEqual(Boolean(approved.handled), true, `${name}.approved.handled`);

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  const pmStepState = harness.pool.state.workflow_steps.find(
    (item) => item.workflow_run_id === started.workflow_run_id && item.step_id === "pm_spec"
  );

  assertEqual(workflowRun.status, "running", `${name}.workflow.status`);
  assertEqual(pmStepState.status, "queued", `${name}.pm_step.status`);

  const approvedEvent = harness.events.find((item) => item.event_name === "workflow.step.approval.approved");
  if (!approvedEvent) {
    throw new Error(`${name}.workflow.step.approval.approved missing`);
  }

  return {
    workflow_run_id: started.workflow_run_id,
    workflow_status: workflowRun.status,
    step_status: pmStepState.status,
    approved_event: approvedEvent,
  };
}

async function runApprovalRejectedCase({ name, workspaceRoot }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${name}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${name}-run`,
    input: { goal: "Update .env secrets and credentials handling for CRM MVP", provider: "opencode", model: "qwen-coder-next" },
  });

  const taskId = started.first_step?.task_id;
  if (!taskId) throw new Error(`${name}: first approval task missing`);

  const rejected = await harness.engine.handleTaskRejected(taskId, "manual rejection for canary");
  assertEqual(Boolean(rejected.handled), true, `${name}.rejected.handled`);

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  const pmStepState = harness.pool.state.workflow_steps.find(
    (item) => item.workflow_run_id === started.workflow_run_id && item.step_id === "pm_spec"
  );

  assertEqual(workflowRun.status, "failed", `${name}.workflow.status`);
  assertEqual(workflowRun.error_code, "APPROVAL_REJECTED", `${name}.workflow.error_code`);
  assertEqual(pmStepState.status, "failed", `${name}.pm_step.status`);
  assertEqual(pmStepState.error_code, "APPROVAL_REJECTED", `${name}.pm_step.error_code`);

  const rejectedEvent = harness.events.find((item) => item.event_name === "workflow.step.approval.rejected");
  if (!rejectedEvent) {
    throw new Error(`${name}.workflow.step.approval.rejected missing`);
  }

  return {
    workflow_run_id: started.workflow_run_id,
    workflow_status: workflowRun.status,
    workflow_error_code: workflowRun.error_code,
    step_status: pmStepState.status,
    step_error_code: pmStepState.error_code,
    rejected_event: rejectedEvent,
  };
}

function writePmSuccessArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "spec.md"), "# Scope\n\n## User Stories\n\n## Acceptance Criteria\n\n## Non-Goals\n\n## Artifact List\n");
  writeText(path.join(rootAbs, "plan", "milestones.md"), "scope user_stories acceptance_criteria non_goals artifact_list");
  writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
    criteria: ["Login works"],
    artifacts: ["plan/spec.md", "plan/milestones.md"],
    owner: "pm_agent",
    version: "v1",
  });
  writeJson(path.join(rootAbs, "handoff", "pm_to_architect.json"), {
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "CRM MVP scope",
    artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    acceptance: { criteria: ["Login works"] },
  });
}

async function runPmWorkflowFailureCase({ name, workspaceRoot, fileWriter, expectedCode }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${name}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${name}-run`,
    input: { goal: "Build CRM MVP", provider: "opencode", model: "qwen-coder-next" },
  });

  const taskId = started.first_step?.task_id;
  const task = harness.pool.state.tasks.find((item) => item.task_id === taskId);
  const payload = JSON.parse(task.payload_json);
  const artifactRootAbs = path.resolve(workspaceRoot, payload.artifact_root);
  ensureDir(artifactRootAbs);
  fileWriter(artifactRootAbs);

  await harness.engine.handleTaskClaimed(taskId);
  const terminal = await harness.engine.handleTaskTerminal({
    task_id: taskId,
    status: "succeeded",
    output: { artifacts: [] },
  });

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  const pmStep = harness.pool.state.workflow_steps.find(
    (item) => item.workflow_run_id === started.workflow_run_id && item.step_id === "pm_spec"
  );

  assertEqual(workflowRun.status, "failed", `${name}.workflow.status`);
  assertEqual(pmStep.status, "failed", `${name}.pm_step.status`);
  assertEqual(pmStep.error_code, expectedCode, `${name}.pm_step.error_code`);
  assertEqual(terminal.handled, true, `${name}.terminal.handled`);

  return {
    workflow_run_id: started.workflow_run_id,
    task_id: taskId,
    terminal,
    workflow_status: workflowRun.status,
    step_status: pmStep.status,
    error_code: pmStep.error_code,
  };
}

async function runArchWorkflowFailureCase({ name, workspaceRoot, archFileWriter, expectedCode }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${name}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${name}-run`,
    input: { goal: "Build CRM MVP", provider: "opencode", model: "qwen-coder-next" },
  });

  const pmTaskId = started.first_step?.task_id;
  const pmTask = harness.pool.state.tasks.find((item) => item.task_id === pmTaskId);
  const pmPayload = JSON.parse(pmTask.payload_json);
  const pmArtifactRootAbs = path.resolve(workspaceRoot, pmPayload.artifact_root);
  ensureDir(pmArtifactRootAbs);
  writePmSuccessArtifacts(pmArtifactRootAbs);

  await harness.engine.handleTaskClaimed(pmTaskId);
  const pmTerminal = await harness.engine.handleTaskTerminal({
    task_id: pmTaskId,
    status: "succeeded",
    output: { artifacts: [] },
  });
  assertEqual(pmTerminal.handled, true, `${name}.pmTerminal.handled`);

  const archTask = harness.pool.state.tasks.find((item) => item.task_id !== pmTaskId);
  if (!archTask) throw new Error(`${name}: arch task not dispatched`);
  const archTaskId = archTask.task_id;
  const archPayload = JSON.parse(archTask.payload_json);
  const archArtifactRootAbs = path.resolve(workspaceRoot, archPayload.artifact_root);
  ensureDir(archArtifactRootAbs);
  archFileWriter(archArtifactRootAbs);

  await harness.engine.handleTaskClaimed(archTaskId);
  const archTerminal = await harness.engine.handleTaskTerminal({
    task_id: archTaskId,
    status: "succeeded",
    output: { artifacts: [] },
  });

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  const archStep = harness.pool.state.workflow_steps.find(
    (item) => item.workflow_run_id === started.workflow_run_id && item.step_id === "arch_design"
  );

  assertEqual(workflowRun.status, "failed", `${name}.workflow.status`);
  assertEqual(archStep.status, "failed", `${name}.arch_step.status`);
  assertEqual(archStep.error_code, expectedCode, `${name}.arch_step.error_code`);
  assertEqual(archTerminal.handled, true, `${name}.archTerminal.handled`);

  return {
    workflow_run_id: started.workflow_run_id,
    pm_task_id: pmTaskId,
    arch_task_id: archTaskId,
    pm_terminal: pmTerminal,
    arch_terminal: archTerminal,
    workflow_status: workflowRun.status,
    step_status: archStep.status,
    error_code: archStep.error_code,
  };
}

async function main() {
  const baseWorkspace = resolveOrchestratorArtifactPath("canary", "workflow_integration_fixture");
  ensureDir(baseWorkspace);

  const pmRoleFailure = await runPmWorkflowFailureCase({
    name: "pm_role_failure",
    workspaceRoot: path.join(baseWorkspace, "pm_role_failure"),
    expectedCode: "PM_REQUIRED_SECTIONS_MISSING",
    fileWriter: (rootAbs) => {
      writeText(path.join(rootAbs, "plan", "spec.md"), "# Scope\n\n## User Stories\n\n## Acceptance Criteria\n\n## Non-Goals\n\n## Artifact List\n");
      writeText(path.join(rootAbs, "plan", "milestones.md"), "scope user stories acceptance criteria non-goals artifact list");
      writeJson(path.join(rootAbs, "plan", "acceptance.json"), { criteria: ["Login works"] });
    },
  });

  const pmHandoffFailure = await runPmWorkflowFailureCase({
    name: "pm_handoff_failure",
    workspaceRoot: path.join(baseWorkspace, "pm_handoff_failure"),
    expectedCode: "HANDOFF_TYPED_FIELDS_MISSING",
    fileWriter: (rootAbs) => {
      writeText(path.join(rootAbs, "plan", "spec.md"), "# Scope\n\n## User Stories\n\n## Acceptance Criteria\n\n## Non-Goals\n\n## Artifact List\n");
      writeText(path.join(rootAbs, "plan", "milestones.md"), "scope user_stories acceptance_criteria non_goals artifact_list");
      writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
        criteria: ["Login works"],
        artifacts: ["plan/spec.md"],
        owner: "pm_agent",
        version: "v1",
      });
      writeJson(path.join(rootAbs, "handoff", "pm_to_architect.json"), {
        from_step: "pm_spec",
        to_steps: ["arch_design"],
        artifacts: ["plan/spec.md"],
        acceptance: { criteria: ["Login works"] },
      });
    },
  });

  const archRoleFailure = await runArchWorkflowFailureCase({
    name: "arch_role_failure",
    workspaceRoot: path.join(baseWorkspace, "arch_role_failure"),
    expectedCode: "ARCH_REQUIRED_SECTIONS_MISSING",
    archFileWriter: (rootAbs) => {
      writeText(path.join(rootAbs, "plan", "arch.md"), "# Module Breakdown\n\n## Interfaces\n\n## Dependency Choices\n\n## Risk Notes\n");
      writeText(path.join(rootAbs, "plan", "workplan.md"), "module breakdown interfaces dependency choices risk notes");
      // interfaces.md exists but has no API endpoint headings → triggers ARCH_INTERFACES_UNSPECIFIED → ARCH_REQUIRED_SECTIONS_MISSING
      writeText(path.join(rootAbs, "plan", "interfaces.md"), "# General Overview\n\nThis is a stub with no endpoints defined.\n");
      writeJson(path.join(rootAbs, "risk", "risk_report.json"), { risks: [{ title: "auth" }] });
    },
  });

  const archHandoffFailure = await runArchWorkflowFailureCase({
    name: "arch_handoff_failure",
    workspaceRoot: path.join(baseWorkspace, "arch_handoff_failure"),
    expectedCode: "HANDOFF_TYPED_SCHEMA_INVALID",
    archFileWriter: (rootAbs) => {
      writeText(path.join(rootAbs, "plan", "arch.md"), "# Module Breakdown\n\n## Interfaces\n\n## Dependency Choices\n\n## Risk Notes\n");
      writeText(path.join(rootAbs, "plan", "workplan.md"), "module_breakdown interfaces dependency_choices risk_notes");
      // interfaces.md with valid endpoint heading so arch validator passes, handoff validation fails instead
      writeText(path.join(rootAbs, "plan", "interfaces.md"), "# POST /api/login\n\nRequest: { email, password }\nResponse: { token }\n");
      writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
        risks: [{ title: "auth", mitigation: "staged rollout" }],
        decision_log: ["Use postgres"],
      });
      writeJson(path.join(rootAbs, "handoff", "architect_to_impl.json"), {
        from_step: "arch_design",
        to_steps: ["impl_be"],
        modules: [""],
        interfaces: ["POST /login"],
        decisions: ["postgres"],
        risks: ["auth migration"],
      });
    },
  });

  const permissionGateFailure = await runPermissionGateFailureCase({
    name: "tool_permission_denied",
    workspaceRoot: path.join(baseWorkspace, "tool_permission_denied"),
  });

  const approvalGate = await runApprovalGateCase({
    name: "approval_gate",
    workspaceRoot: path.join(baseWorkspace, "approval_gate"),
  });

  const approvalApproved = await runApprovalApprovedCase({
    name: "approval_approved",
    workspaceRoot: path.join(baseWorkspace, "approval_approved"),
  });

  const approvalRejected = await runApprovalRejectedCase({
    name: "approval_rejected",
    workspaceRoot: path.join(baseWorkspace, "approval_rejected"),
  });

  const outDir = resolveOrchestratorArtifactPath("canary", "coding_team_workflow_integration");
  ensureDir(outDir);
  const reportPath = path.join(outDir, "coding_team_workflow_integration_canary.json");
  fs.writeFileSync(
    reportPath,
    JSON.stringify(
      {
        ok: true,
        generated_at: new Date().toISOString(),
        pm_role_failure: pmRoleFailure,
        pm_handoff_failure: pmHandoffFailure,
        arch_role_failure: archRoleFailure,
        arch_handoff_failure: archHandoffFailure,
        tool_permission_denied: permissionGateFailure,
        approval_gate: approvalGate,
        approval_approved: approvalApproved,
        approval_rejected: approvalRejected,
      },
      null,
      2
    ),
    "utf8"
  );
  console.log("# Coding Team Workflow Integration Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
