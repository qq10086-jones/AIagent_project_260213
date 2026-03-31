import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";
import { analyzeTaskRisk } from "./policy.js";
import { validateToolPermission } from "./vnext/tool_permission_guard.js";
import { parseJsonSafe } from "./domain/workflow_runner.js";
import { normalizeStepStatus, validatePromptScriptBinding } from "./domain/workflow_state.js";
import { buildReleasePackPaths, classifyArtifactReasons, buildFailurePayload } from "./domain/workflow_artifact_audit.js";
import { createWorkflowReleasePackService } from "./domain/workflow_release_pack.js";
import { createStepBuilder } from "./domain/workflow_step_builder.js";
import { runStepSuccessValidations } from "./domain/workflow_step_validator.js";
import { createArtifactPackService } from "./domain/workflow_artifact_pack.js";
import { createTaskHandlerService } from "./domain/workflow_task_handler.js";
import { createResumeService } from "./domain/workflow_resume.js";
import { persistWorkflowMemory } from "./domain/memory_writer.js";
import { resolveWorkflowResultUrl } from "./domain/workflow_preview_result.js";
import { createPatchBundleService } from "./domain/patch_bundle_service.js";
import { createContextBudgetService } from "./domain/context_budget_service.js";
import { createWorkflowParallelizationRuntime } from "./domain/workflow_parallelization_runtime.js";
import { createRoutingAuditLogService } from "./domain/routing_audit_log.js";
import { createWaterfallTraceService } from "./domain/waterfall_trace_service.js";
import { summarizeDagProgress } from "./domain/dag_scheduler.js";
import { getTaskPayloadRecord } from "./data/task_repository.js";
import { updateRunStatus } from "./data/run_repository.js";
import { createWorkflowStepArtifactHelpers } from "./domain/workflow_step_artifacts.js";
import { createWorkflowCheckpointService } from "./domain/workflow_checkpoint.js";
import {
  failWorkflowStepIfNotSucceeded,
  getWorkflowRunById,
  getWorkflowStepByIndex,
  getWorkflowStepGateByIndex,
  insertWorkflowRun,
  insertWorkflowStep,
  listWorkflowCheckpoints,
  listWorkflowSteps,
  updateWorkflowRunCurrentStep,
  updateWorkflowRunFailed,
  updateWorkflowRunSucceeded,
  updateWorkflowStepDispatchState,
  updateWorkflowStepFailed,
  updateWorkflowStepSucceeded,
} from "./data/workflow_repository.js";

export function createWorkflowEngine({
  pool,
  registry,
  promptScriptRegistry = null,
  handoffContracts = null,
  enqueueTask,
  recordEvent,
  makeIdempotencyKey,
  resumeTokenSecret = "dev-resume-secret",
  resumeTokenTtlSec = 86400,
  workspaceRoot = "/workspace",
  auditStepArtifacts = true,
  strictStepArtifacts = false,
  minio = null,
  onStepTransition = null,
  runtimeConfig = {},
  waterfallTraceService = null,
  artifactPackService = null,
}) {
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function withTimeout(promise, timeoutMs, timeoutMessage) {
    let timer = null;
    try {
      return await Promise.race([
        promise,
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
        }),
      ]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  const { archiveReleasePackToMinio, indexReleasePackToDb, minioBucket } =
    createWorkflowReleasePackService({ pool, recordEvent, workspaceRoot, minio });

  const { buildStepPayload } = createStepBuilder({ registry, promptScriptRegistry, handoffContracts, workspaceRoot, runtimeConfig });
  const patchBundleService = createPatchBundleService({ workspaceRoot });
  const contextBudgetService = createContextBudgetService();
  const routingAuditLogService = createRoutingAuditLogService({ pool, workspaceRoot });
  const _waterfallSvc = waterfallTraceService ?? createWaterfallTraceService({ pool });
  const parallelizationRuntime = createWorkflowParallelizationRuntime({ registry, workspaceRoot, recordEvent, pool, routingAuditLogService, waterfallTraceService: _waterfallSvc });

  const { writeContextBudgetReport, writeContextArtifacts, applyStructuredPatchIfPresent } =
    createWorkflowStepArtifactHelpers({ workspaceRoot, contextBudgetService, patchBundleService });
  const { createCheckpoint } = createWorkflowCheckpointService({ pool });

  async function getRun(workflow_run_id) {
    return getWorkflowRunById(pool, workflow_run_id);
  }

  async function getSteps(workflow_run_id) {
    return listWorkflowSteps(pool, workflow_run_id);
  }

  async function failWorkflowRun({ run, stepDef, stepIndex, error_code, error_message, failure_payload = null }) {
    await updateWorkflowRunFailed(pool, run.workflow_run_id, error_code, error_message);
    if (Number.isInteger(stepIndex)) {
      await failWorkflowStepIfNotSucceeded(pool, run.workflow_run_id, stepIndex, error_code);
    }
    if (run.run_id) {
      await updateRunStatus(pool, run.run_id, "failed").catch(() => {});
    }
    await recordEvent(run.workflow_run_id, "workflow.failed", {
      workflow_run_id: run.workflow_run_id,
      step_id: stepDef?.id || null,
      step_index: Number.isInteger(stepIndex) ? stepIndex : null,
      error_code: String(error_code || "WORKFLOW_FAILED"),
      error: String(error_message || "workflow failed"),
      failure_payload: failure_payload || null,
    });
    
    if (typeof onStepTransition === "function") {
      onStepTransition({
        event: "workflow.failed",
        workflow_run_id: run.workflow_run_id,
        step_id: stepDef?.id || null,
        error_message: String(error_message || "workflow failed")
      }).catch(() => {});
    }
  }

  async function succeedWorkflowRun(run, { failOnPackError = true } = {}) {
    const steps = await getSteps(run.workflow_run_id);
    let pack = null;
    const releasePaths = buildReleasePackPaths(workspaceRoot, run);
    try {
      pack = await withTimeout(
        artifactPack.generateArtifactPack(run),
        15000,
        `artifact pack generation timed out after 15000ms`,
      );
    } catch (err) {
      if (fs.existsSync(releasePaths.manifest_path) && fs.existsSync(releasePaths.summary_path)) {
        await updateWorkflowRunSucceeded(pool, run.workflow_run_id);
        if (run.run_id) {
          await updateRunStatus(pool, run.run_id, "completed").catch(() => {});
        }
        await recordEvent(run.workflow_run_id, "workflow.succeeded", {
          workflow_run_id: run.workflow_run_id,
          adopted_existing_artifact_pack: true,
          partial_finalization: true,
          finalization_error: err?.message || String(err),
        });
        return { ok: true, adopted_existing_artifact_pack: true };
      }
      if (!failOnPackError) {
        return {
          ok: false,
          deferred: true,
          error_code: "WORKFLOW_FINALIZATION_FAILED",
          error: err?.message || "workflow finalization failed",
        };
      }
      const errorCode = "WORKFLOW_FINALIZATION_FAILED";
      const errorMessage = err?.message || "workflow finalization failed";
      await updateWorkflowRunFailed(pool, run.workflow_run_id, errorCode, errorMessage);
      if (run.run_id) await updateRunStatus(pool, run.run_id, "failed").catch(() => {});
      await recordEvent(run.workflow_run_id, "workflow.finalization.failed", {
        workflow_run_id: run.workflow_run_id,
        error_code: errorCode,
        error: String(errorMessage),
      });
      return { ok: false, error_code: errorCode, error: errorMessage };
    }
    if (!pack.ok) {
      if (!failOnPackError) {
        return {
          ok: false,
          deferred: true,
          error_code: "ARTIFACT_INCOMPLETE",
          error: pack.error || "artifact pack incomplete",
          reasons: pack.reasons || [],
        };
      }
      const classified = classifyArtifactReasons(pack.reasons || []);
      const failurePayload = buildFailurePayload({
        errorCode: "ARTIFACT_INCOMPLETE",
        failedStep: "release_pack",
        missing: classified.missing,
        invalid: classified.invalid,
        detail: pack.error || "artifact pack incomplete",
      });
      await updateWorkflowRunFailed(pool, run.workflow_run_id, "ARTIFACT_INCOMPLETE", pack.error || "artifact pack incomplete");
      if (run.run_id) await updateRunStatus(pool, run.run_id, "failed").catch(() => {});
      await recordEvent(run.workflow_run_id, "artifact.pack.failed", {
        workflow_run_id: run.workflow_run_id,
        error_code: "ARTIFACT_INCOMPLETE",
        reasons: pack.reasons || [],
        failure_payload: failurePayload,
      });
      return { ok: false, error_code: "ARTIFACT_INCOMPLETE", error: pack.error || "artifact pack incomplete" };
    }
    await updateWorkflowRunSucceeded(pool, run.workflow_run_id);
    if (run.run_id) {
      await updateRunStatus(pool, run.run_id, "completed").catch(() => {});
    }
    await recordEvent(run.workflow_run_id, "workflow.succeeded", { workflow_run_id: run.workflow_run_id });
    const workflowResult = resolveWorkflowResultUrl({ steps, minio, minioBucket, run });
    
    let runSummary = "";
    try {
      const manifest = pack.run_manifest_path && fs.existsSync(pack.run_manifest_path)
        ? JSON.parse(fs.readFileSync(pack.run_manifest_path, "utf8"))
        : null;
      const runtimeEvidence = manifest?.runtime_evidence_summary && typeof manifest.runtime_evidence_summary === "object"
        ? manifest.runtime_evidence_summary
        : null;
      if (runtimeEvidence) {
        const summaryLines = [
          `status=${String(manifest?.status || run.status || "succeeded")}`,
          `smoke=${String(runtimeEvidence.smoke_verdict || "not_found")}`,
          `root=${String(runtimeEvidence.smoke_root_status ?? "n/a")}`,
          `api=${String(runtimeEvidence.smoke_api_status ?? "n/a")}`,
          `superpowers_configured_steps=${Number(runtimeEvidence.superpowers_configured_steps || 0)}`,
          `superpowers_available_steps=${Number(runtimeEvidence.superpowers_available_steps || 0)}`,
        ];
        runSummary = summaryLines.join("\n").trim();
      }
      if (!runSummary) {
        const releaseRoot = path.dirname(path.dirname(pack.run_manifest_path || ""));
        const readmePath = path.join(releaseRoot, "release", "README.md");
        if (fs.existsSync(readmePath)) {
          runSummary = fs.readFileSync(readmePath, "utf8").split(/\r?\n/).slice(0, 8).join("\n").trim();
        }
      }
    } catch {
      runSummary = "";
    }

    if (typeof onStepTransition === "function") {
      onStepTransition({
        event: "workflow.completed",
        workflow_run_id: run.workflow_run_id,
        result_url: workflowResult.result_url,
        run_summary: runSummary,
      }).catch(() => {});
    }

    await recordEvent(run.workflow_run_id, "artifact.pack.generated", {
      workflow_run_id: run.workflow_run_id,
      run_manifest: pack.run_manifest_path,
      release_summary: pack.summary_path,
      go_no_go_result: pack.go_no_go_result_path || null,
      go_no_go_verdict: pack.go_no_go_verdict || null,
      strict_canary_report: pack.strict_canary_report_path || null,
      strict_canary_json: pack.strict_canary_json_path || null,
      strict_canary_verdict: pack.strict_canary_verdict || null,
      preview_validation_report: pack.preview_validation_report_path || null,
      preview_validation_classification: pack.preview_validation_classification || null,
      preview_validation_warning: pack.preview_validation_warning || null,
      product_fidelity_report: pack.product_fidelity_report_path || null,
      product_fidelity_classification: pack.product_fidelity_classification || null,
      product_fidelity_warning: pack.product_fidelity_warning || null,
      preview_url: workflowResult.preview_url,
      preview_status: workflowResult.preview_status,
      preview_fallback_reason: workflowResult.fallback_reason,
      deployment_result_path: workflowResult.deployment_result_path,
    });
    try {
      const releaseRoot = path.dirname(path.dirname(pack.run_manifest_path || ""));
      const runtimeRoot = path.join(path.dirname(path.dirname(releaseRoot)), "runs", String(run.run_id || ""));
      const memoryWrite = persistWorkflowMemory({ run, releaseRoot, runtimeRoot });
      await recordEvent(run.workflow_run_id, "memory.write.succeeded", {
        workflow_run_id: run.workflow_run_id,
        project_id: memoryWrite.project_id,
        task_history_path: memoryWrite.task_history_path,
        copied_adr_paths: memoryWrite.copied_adr_paths,
        coding_failure_memory: memoryWrite.coding_failure_memory,
      });
    } catch (err) {
      await recordEvent(run.workflow_run_id, "memory.write.failed", {
        workflow_run_id: run.workflow_run_id,
        error: err?.message || String(err),
      });
    }
    return { ok: true };
  }

  async function finalizeWorkflowRunIfReady(workflow_run_id) {
    const run = await getRun(workflow_run_id);
    if (!run) return { ok: false, error_code: "WORKFLOW_RUN_NOT_FOUND" };
    if (["failed", "succeeded", "partial_failure"].includes(String(run.status || ""))) {
      return { ok: true, skipped: true, reason: `run status ${run.status}` };
    }
    const steps = await getSteps(workflow_run_id);
    if (!Array.isArray(steps) || steps.length === 0) {
      return { ok: false, ready: false, reason: "no workflow steps found" };
    }
    if (steps.some((step) => normalizeStepStatus(step.status) !== "succeeded")) {
      return { ok: false, ready: false, reason: "not all workflow steps are succeeded" };
    }
    const checkpoints = await listWorkflowCheckpoints(pool, workflow_run_id);
    if (checkpoints.length < steps.length) {
      return {
        ok: false,
        ready: false,
        reason: `checkpoint count lower than step count (${checkpoints.length}/${steps.length})`,
      };
    }
    if (fs.existsSync(releasePaths.manifest_path) && fs.existsSync(releasePaths.summary_path)) {
      await updateWorkflowRunSucceeded(pool, run.workflow_run_id);
      if (run.run_id) {
        await updateRunStatus(pool, run.run_id, "completed").catch(() => {});
      }
      await recordEvent(run.workflow_run_id, "workflow.succeeded", {
        workflow_run_id: run.workflow_run_id,
        adopted_existing_artifact_pack: true,
      });
      return { ok: true, adopted_existing_artifact_pack: true };
    }
    return succeedWorkflowRun(run, { failOnPackError: false });
  }

  async function dispatchStepByIndex(workflow_run_id, stepIndex, context = null) {
    const run = await getRun(workflow_run_id);
    if (!run) throw new Error(`workflow_run '${workflow_run_id}' not found`);
    if (["failed", "succeeded", "partial_failure"].includes(String(run.status || ""))) {
      return { skipped: true, reason: `run status ${run.status}` };
    }
    const resolved = await parallelizationRuntime.getResolvedWorkflow(run);
    const wf = resolved?.workflow || null;
    if (!wf || !Array.isArray(wf.steps)) {
      await failWorkflowRun({
        run, stepDef: null, stepIndex,
        error_code: "WORKFLOW_DEF_MISSING",
        error_message: `workflow '${run.workflow_id}' not found in registry`,
      });
      return { failed: true, error_code: "WORKFLOW_DEF_MISSING" };
    }
    const stepDef = wf.steps[stepIndex];
    if (!stepDef) {
      await succeedWorkflowRun(run);
      return { completed: true };
    }
    const stepRow = await getWorkflowStepByIndex(pool, workflow_run_id, stepIndex);
    if (!stepRow) {
      await failWorkflowRun({
        run, stepDef, stepIndex,
        error_code: "STEP_STATE_MISSING",
        error_message: `step state missing for ${stepDef.id}`,
      });
      return { failed: true, error_code: "STEP_STATE_MISSING" };
    }
    const stepStatus = normalizeStepStatus(stepRow.status);
    if (!["pending", "failed"].includes(stepStatus)) {
      return { skipped: true, reason: `step status ${stepStatus}` };
    }
    const payload = buildStepPayload({ run, stepDef, stepIndex });
    const toolPermission = validateToolPermission(String(stepDef.role || ""), String(stepDef.tool || ""));
    if (!toolPermission.allowed) {
      await recordEvent(workflow_run_id, "policy.tool_permission.denied", {
        workflow_run_id, step_id: stepDef.id, step_index: stepIndex,
        role_name: String(stepDef.role || ""), tool_name: String(stepDef.tool || ""),
        reason: String(toolPermission.reason || ""),
      });
      await failWorkflowRun({
        run, stepDef, stepIndex,
        error_code: "TOOL_PERMISSION_DENIED",
        error_message: toolPermission.reason || `role '${stepDef.role}' cannot use tool '${stepDef.tool}'`,
      });
      return { failed: true, error_code: "TOOL_PERMISSION_DENIED", error: toolPermission.reason || "tool permission denied" };
    }
    const risk = analyzeTaskRisk(stepDef.tool, payload);
    await recordEvent(workflow_run_id, "policy.gate.checked", {
      workflow_run_id, step_id: stepDef.id, step_index: stepIndex,
      tool_name: stepDef.tool, risk_level: risk.risk_level,
      requires_approval: Boolean(risk.requires_approval), reasons: risk.reasons || [],
    });
    try {
      const enq = await enqueueTask({
        tool_name: stepDef.tool, payload, run_id: run.run_id, risk_level: risk.risk_level,
        idempotency_key: makeIdempotencyKey(run.run_id, stepDef.tool, {
          workflow_run_id, step_id: stepDef.id, step_index: stepIndex, payload,
        }),
        context,
      });
      await updateWorkflowStepDispatchState(pool, workflow_run_id, stepIndex, {
        status: enq.waiting_approval ? "waiting_approval" : "queued",
        task_id: enq.task_id,
        risk_level: risk.risk_level || "low",
        approval_required: Boolean(enq.waiting_approval),
        approval_reasons: risk.reasons || [],
      });
      await updateWorkflowRunCurrentStep(pool, workflow_run_id, stepIndex);
      await recordEvent(enq.task_id, "workflow.step.dispatched", {
        workflow_run_id, step_id: stepDef.id, step_index: stepIndex,
        waiting_approval: Boolean(enq.waiting_approval),
      });

      if (typeof onStepTransition === "function" && !enq.waiting_approval) {
        onStepTransition({
          event: "step.started",
          workflow_run_id,
          step_id: stepDef.id,
          step_index: stepIndex,
          action_summary: `Executing task: ${stepDef.tool}`
        }).catch(() => {});
      }

      return {
        ok: true,
        task_id: enq.task_id,
        waiting_approval: Boolean(enq.waiting_approval),
        step_id: stepDef.id,
        step_index: stepIndex,
      };
    } catch (err) {
      const code = String(err?.code || "");
      await failWorkflowRun({
        run, stepDef, stepIndex,
        error_code: code || "STEP_DISPATCH_FAILED",
        error_message: err?.message || "step dispatch failed",
      });
      return { failed: true, error_code: code || "STEP_DISPATCH_FAILED", error: err?.message || "step dispatch failed" };
    }
  }

  async function dispatchReadySteps(workflow_run_id, context = null) {
    const run = await getRun(workflow_run_id);
    if (!run) throw new Error(`workflow_run '${workflow_run_id}' not found`);
    if (["failed", "succeeded", "partial_failure"].includes(String(run.status || ""))) {
      return [];
    }
    const dagPlan = await parallelizationRuntime.getDagPlan(run);
    if (!dagPlan) return [];
    const stepRows = await getSteps(workflow_run_id);
    const { readyStepIndexes } = summarizeDagProgress({ stepRows, dagPlan });
    if (readyStepIndexes.length === 0) return [];
    return Promise.all(readyStepIndexes.map((stepIndex) => dispatchStepByIndex(workflow_run_id, stepIndex, context)));
  }

  async function reconcileWorkflowState(workflow_run_id, context = null) {
    const run = await getRun(workflow_run_id);
    if (!run) return { handled: false };
    const dagPlan = await parallelizationRuntime.getDagPlan(run);
    if (!dagPlan) return { handled: false };

    const stepRows = await getSteps(workflow_run_id);
    const dagState = summarizeDagProgress({ stepRows, dagPlan });
    if (dagState.readyStepIndexes.length > 0) {
      const dispatched = await dispatchReadySteps(workflow_run_id, context);
      return { state: "dispatched", dispatched };
    }
    if (dagState.hasActiveWork) {
      return { state: "running" };
    }
    if (dagState.allSucceeded) {
      await succeedWorkflowRun(run);
      return { state: "succeeded" };
    }
    if (dagState.mixedFailureGroup) {
      const failedSteps = stepRows.filter(
        (row) =>
          dagState.mixedFailureGroup.some((meta) => meta.step_index === Number(row.step_index)) &&
          normalizeStepStatus(row.status) === "failed"
      );
      await parallelizationRuntime.markWorkflowPartialFailure(run, failedSteps);
      return { state: "partial_failure", failed_steps: failedSteps };
    }
    if (dagState.anyFailed) {
      const failedSteps = stepRows.filter((row) => normalizeStepStatus(row.status) === "failed");
      const firstFailed = failedSteps[0] || null;
      await failWorkflowRun({
        run,
        stepDef: firstFailed ? { id: String(firstFailed.step_id || "") } : null,
        stepIndex: null,
        error_code: String(firstFailed?.error_code || "STEP_FAILED"),
        error_message: firstFailed
          ? `workflow failed at step '${String(firstFailed.step_id || firstFailed.step_index)}'`
          : "workflow failed",
      });
      return { state: "failed", failed_steps: failedSteps };
    }
    return { state: "idle" };
  }

  async function startWorkflowRun({ workflow_id, project_type, run_id, input = {}, context = null }) {
    const wf = registry.workflows?.[workflow_id];
    if (!wf) {
      const err = new Error(`workflow '${workflow_id}' not found`); err.code = "WORKFLOW_NOT_FOUND"; throw err;
    }
    const resolvedProjectType = String(project_type || wf.project_type || "");
    if (!registry.project_types?.[resolvedProjectType]) {
      const err = new Error(`project_type '${resolvedProjectType}' not found`); err.code = "PROJECT_TYPE_NOT_FOUND"; throw err;
    }
    // Allow if: workflow declares no project_type, types match, or the resolved type maps back to this workflow
    const workflowOwnsType = registry.project_types?.[resolvedProjectType]?.default_workflow === workflow_id;
    if (wf.project_type && wf.project_type !== resolvedProjectType && !workflowOwnsType) {
      const err = new Error(`workflow '${workflow_id}' project_type mismatch: expected '${wf.project_type}', got '${resolvedProjectType}'`);
      err.code = "WORKFLOW_PROJECT_TYPE_MISMATCH"; throw err;
    }
    const steps = Array.isArray(wf.steps) ? wf.steps : [];
    if (steps.length === 0) {
      const err = new Error(`workflow '${workflow_id}' has no steps`); err.code = "WORKFLOW_EMPTY"; throw err;
    }
    for (const step of steps) {
      const promptScriptId = String(step?.prompt_script_id || "").trim();
      if (!promptScriptId) continue;
      const promptScript = promptScriptRegistry?.scripts?.[promptScriptId] || null;
      const checked = validatePromptScriptBinding({ stepDef: step, promptScriptRegistry, promptScript });
      if (!checked.ok) {
        const err = new Error(checked.detail || `workflow '${workflow_id}' prompt script binding invalid`);
        err.code = checked.code || "PROMPT_SCRIPT_BINDING_INVALID"; throw err;
      }
    }
    const workflow_run_id = uuidv4();
    await insertWorkflowRun(pool, {
      workflow_run_id,
      run_id,
      workflow_id,
      project_type: resolvedProjectType,
      input,
    });
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      await insertWorkflowStep(pool, {
        workflow_run_id,
        step_index: i,
        step_id: String(step.id || `step_${i}`),
        role_name: String(step.role || ""),
        tool_name: String(step.tool || ""),
        gate_name: String(step.gate || ""),
      });
    }
    await recordEvent(workflow_run_id, "workflow.started", {
      workflow_run_id, workflow_id, project_type: resolvedProjectType, run_id,
      steps: steps.map((s, idx) => ({ step_index: idx, step_id: s.id, role: s.role, tool: s.tool, gate: s.gate })),
    });
    const firstBatch = await dispatchReadySteps(workflow_run_id, context);
    const first = firstBatch[0] || null;
    if (typeof onStepTransition === "function") {
      onStepTransition({
        event: "workflow.started", workflow_run_id, run_id, workflow_id,
        step_count: steps.length,
        first_step_id: String(first?.step_id || steps[0]?.id || ""),
        step_ids: steps.map((step) => String(step?.id || "")),
      }).catch(() => {});
    }
    return {
      workflow_run_id,
      run_id,
      workflow_id,
      project_type: resolvedProjectType,
      first_step: first,
      first_steps: firstBatch,
    };
  }

  async function handleTaskTerminal({ task_id, status, output, error_code }) {
    const task = await getTaskPayloadRecord(pool, task_id);
    if (!task) return { handled: false };
    const payload = parseJsonSafe(task.payload_json, {});
    const workflow_run_id = payload.workflow_run_id;
    const step_index = Number(payload.step_index);
    const step_id = String(payload.step_id || "");
    if (!workflow_run_id || !Number.isInteger(step_index)) return { handled: false };

    const run = await getRun(workflow_run_id);
    if (!run) return { handled: false };
    const stepRow = await getWorkflowStepGateByIndex(pool, workflow_run_id, step_index);
    const gateName = String(stepRow?.gate_name || "");

    // Idempotency guard: if the step is already in a terminal DB state, this
    // message was redelivered after a consumer crash between updateTaskTerminalResult
    // and XACK.  Skip re-processing to prevent duplicate step dispatch.
    const stepStatus = String(stepRow?.status || "");
    if (stepStatus === "succeeded" || stepStatus === "failed") {
      console.warn(`[workflow] handleTaskTerminal: step ${step_id}[${step_index}] already ${stepStatus} - skipping duplicate`);
      return { handled: false, skipped_duplicate: true };
    }

    if (status === "succeeded") {
      const budgetMetrics = writeContextBudgetReport(payload);
      const contextArtifacts = writeContextArtifacts(payload);
      let patchApplication = null;
      if (["impl_be", "impl_fe"].includes(step_id)) {
        try {
          patchApplication = applyStructuredPatchIfPresent(payload);
        } catch (err) {
          const failurePayload = buildFailurePayload({
            errorCode: String(err?.code || "PATCH_APPLICATION_FAILED"),
            failedStep: step_id,
            detail: err?.message || "structured patch application failed",
          });
          await updateWorkflowStepFailed(pool, workflow_run_id, step_index, { ...(output || {}), failure_payload: failurePayload }, String(err?.code || "PATCH_APPLICATION_FAILED"));
          const next = await reconcileWorkflowState(workflow_run_id);
          return { handled: true, workflow_run_id, step_index, failed_due_to_patch_application: true, next };
        }
      }
      const valResult = await runStepSuccessValidations({
        run, step_id, payload, output, workflow_run_id,
        pool, workspaceRoot, handoffContracts,
        auditStepArtifacts, strictStepArtifacts,
      });
      if (valResult.ok) {
        valResult.mergedOutput = {
          ...valResult.mergedOutput,
          execution_mode_used:
            valResult.mergedOutput?.impl_validation?.execution_mode_used ||
            patchApplication?.mode ||
            "full_file_fallback",
          context_budget_report_path: budgetMetrics.report_path,
          context_budget_report: budgetMetrics.report,
          context_packet_path: contextArtifacts.context_packet_path,
          repo_map_path: contextArtifacts.repo_map_path,
          patch_application: patchApplication,
        };
      }

      if (valResult.logMissingArtifacts) {
        await recordEvent(task_id, "workflow.step.artifacts.missing", {
          workflow_run_id, step_index, step_id,
          artifact_root: valResult.mergedOutput?.artifact_check?.artifact_root || "",
          missing: valResult.mergedOutput?.artifact_check?.missing || [],
          found: valResult.mergedOutput?.artifact_check?.found || [],
          strict_mode: Boolean(strictStepArtifacts),
        });
      }

      if (!valResult.ok) {
        await updateWorkflowStepFailed(pool, workflow_run_id, step_index, valResult.mergedOutput, valResult.code);
        const next = await reconcileWorkflowState(workflow_run_id);
        return { handled: true, workflow_run_id, step_index, [`failed_due_to_${valResult.failKey}_validation`]: true, next };
      }

      const checkpoint = await createCheckpoint({
        workflow_run_id, stepIndex: step_index, step_id, task_id, status,
        output: valResult.mergedOutput,
      });
      await updateWorkflowStepSucceeded(pool, workflow_run_id, step_index, valResult.mergedOutput, checkpoint.checkpoint_id);
      const nextResult = await reconcileWorkflowState(workflow_run_id);
      if (typeof onStepTransition === "function" && Array.isArray(nextResult?.dispatched)) {
        for (const dispatched of nextResult.dispatched.filter((item) => item?.ok || item?.waiting_approval)) {
          if (dispatched.waiting_approval) {
            onStepTransition({
              event: "step.approval_required",
              workflow_run_id,
              step_id: String(dispatched.step_id || ""),
              step_index: Number(dispatched.step_index),
            }).catch(() => {});
          } else {
            onStepTransition({
              event: "step.completed",
              workflow_run_id,
              completed_step_id: step_id,
              completed_step_index: step_index,
              next_step_id: String(dispatched.step_id || ""),
              next_step_index: Number(dispatched.step_index),
            }).catch(() => {});
          }
        }
      }
      return { handled: true, workflow_run_id, step_index, checkpoint_id: checkpoint.checkpoint_id, next: nextResult };
    }

    // failure branch
    const genericCode = String(error_code || (gateName === "acceptance" ? "ACCEPTANCE_FAILED" : "STEP_FAILED"));
    const genericMsg = String(error_code || (gateName === "acceptance" ? "acceptance gate failed" : "step failed"));
    const genericFailurePayload = buildFailurePayload({ errorCode: genericCode, failedStep: step_id, detail: genericMsg });
    await updateWorkflowStepFailed(pool, workflow_run_id, step_index, { ...(output || {}), failure_payload: genericFailurePayload }, genericCode);
    
    if (typeof onStepTransition === "function") {
      onStepTransition({
        event: "step.failed",
        workflow_run_id,
        step_id: step_id,
        step_index: step_index,
        error_code: genericCode,
        error_message: genericMsg
      }).catch(() => {});
    }

    const next = await reconcileWorkflowState(workflow_run_id);
    return { handled: true, workflow_run_id, step_index, next };
  }

  async function getWorkflowRunStatus(workflow_run_id) {
    const run = await getRun(workflow_run_id);
    if (!run) return null;
    const steps = await getSteps(workflow_run_id);
    const checkpoints = await listWorkflowCheckpoints(pool, workflow_run_id);
    return { run, steps, checkpoints };
  }

  // Service objects that reference local functions (constructed after function declarations)
  const artifactPack = artifactPackService || createArtifactPackService({
    pool, workspaceRoot, registry,
    archiveReleasePackToMinio, indexReleasePackToDb, minioBucket,
    recordEvent, getSteps, runtimeConfig,
  });

  const { handleTaskClaimed, handleTaskApproved, handleTaskRejected } = createTaskHandlerService({
    pool, recordEvent, getRun, failWorkflowRun,
  });

  const { issueResumeToken, resumeFromToken } = createResumeService({
    pool, resumeTokenSecret, resumeTokenTtlSec,
    getRun, getSteps, dispatchStepByIndex, recordEvent,
  });

  return {
    startWorkflowRun,
    handleTaskClaimed,
    handleTaskApproved,
    handleTaskRejected,
    handleTaskTerminal,
    finalizeWorkflowRunIfReady,
    issueResumeToken,
    resumeFromToken,
    getWorkflowRunStatus,
    validateRunArtifactPack: artifactPack.validateRunArtifactPack,
    archiveRunArtifactPack: artifactPack.archiveRunArtifactPack,
  };
}


