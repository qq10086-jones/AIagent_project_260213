/**
 * workflow_step_builder.js
 *
 * Step payload construction helpers.
 * Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import fs from "fs";
import path from "path";
import { getProjectContext, getPriorADRs, getTaskHistory } from "./memory_reader.js";
import {
  buildStepPrompt,
  STEP_CONTRACTS,
  validatePromptScriptBinding,
} from "./workflow_state.js";
import {
  buildBackendExecutionPacket,
  validateBackendExecutionPacket,
  buildFrontendExecutionPacket,
  validateFrontendExecutionPacket,
} from "../coding_execution_adapters.js";
import { buildCodingExecutorRequest, validateCodingExecutorRequest } from "../coding_executor.js";
import { parseJsonSafe } from "./workflow_runner.js";
import { createContextBudgetService } from "./context_budget_service.js";

export function pathForRunArtifacts(run_id) {
  return `artifacts/release/${run_id || "unknown-run"}`;
}

function listTargetFilesWithContent({ workspaceRoot, targetPaths, maxFiles = 3, maxCharsPerFile = 4000 }) {
  const out = [];
  const workspaceAbs = path.resolve(workspaceRoot);
  for (const targetPath of targetPaths) {
    const normalized = String(targetPath || "").replace(/\\/g, "/").replace(/^\/+/, "");
    if (!normalized) continue;
    const absPath = path.resolve(workspaceAbs, normalized);
    if (!absPath.startsWith(workspaceAbs) || !fs.existsSync(absPath)) continue;
    const stat = fs.statSync(absPath);
    if (stat.isFile()) {
      const content = fs.readFileSync(absPath, "utf8");
      out.push({ path: normalized, content: content.slice(0, maxCharsPerFile) });
    } else if (stat.isDirectory()) {
      const entries = fs.readdirSync(absPath, { withFileTypes: true });
      for (const entry of entries) {
        if (!entry.isFile()) continue;
        const fileAbs = path.join(absPath, entry.name);
        const relPath = path.relative(workspaceAbs, fileAbs).replace(/\\/g, "/");
        const content = fs.readFileSync(fileAbs, "utf8");
        out.push({ path: relPath, content: content.slice(0, maxCharsPerFile) });
        if (out.length >= maxFiles) return out;
      }
    }
    if (out.length >= maxFiles) return out;
  }
  return out;
}

function buildTargetFileContextBlock(targetFileContext = []) {
  if (!Array.isArray(targetFileContext) || targetFileContext.length === 0) return "";
  const lines = ["", "[Target File Context]"];
  for (const item of targetFileContext) {
    lines.push(`File: ${item.path}`);
    lines.push("```");
    lines.push(String(item.content || ""));
    lines.push("```");
  }
  return lines.join("\n");
}

function chooseImplementationMode({
  stepDef,
  input,
  payload,
  runtimeConfig,
  workspaceRoot,
  contextBudgetService,
}) {
  if (!["impl_be", "impl_fe"].includes(String(stepDef?.id || ""))) {
    return {
      executionModeRequested: "full_file_fallback",
      promptScriptId: String(stepDef?.prompt_script_id || ""),
      targetFileContext: [],
      contextBudgetPreview: null,
    };
  }
  const diffFirstEnabled = runtimeConfig?.execution?.diff_first_enabled !== false;
  const targetPaths = Array.isArray(payload?.target_paths) ? payload.target_paths : ["sandbox/crm_site/"];
  const targetFileContext = diffFirstEnabled
    ? listTargetFilesWithContent({ workspaceRoot, targetPaths })
    : [];
  const hasTargetFiles = targetFileContext.length > 0;
  const v2ScriptId = stepDef.id === "impl_be" ? "backend.impl.v2" : "frontend.impl.v2";
  const injectedContext = targetFileContext.map((item) => `${item.path}\n${item.content}`).join("\n");
  const contextBudgetPreview = contextBudgetService.buildReport({
    stepId: stepDef.id,
    role: stepDef.role,
    prompt: String(input.goal || input.task_prompt || input.prompt || ""),
    injectedContext,
    runId: payload.run_id,
    workflowRunId: payload.workflow_run_id,
  });
  const shouldUseDiffFirst = diffFirstEnabled && hasTargetFiles && contextBudgetPreview.status !== "overflow_risk";
  return {
    executionModeRequested: shouldUseDiffFirst ? "structured_patch" : "full_file_fallback",
    promptScriptId: shouldUseDiffFirst ? v2ScriptId : String(stepDef?.prompt_script_id || ""),
    targetFileContext: shouldUseDiffFirst ? targetFileContext : [],
    contextBudgetPreview,
  };
}

function formatMemoryValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

export function buildArchitectMemoryContext({ projectContext = null, priorADRs = [], taskHistory = [] }) {
  const hasProjectContext = projectContext && typeof projectContext === "object";
  const safeADRs = Array.isArray(priorADRs) ? priorADRs.filter(Boolean) : [];
  const safeTaskHistory = Array.isArray(taskHistory) ? taskHistory.filter(Boolean) : [];
  if (!hasProjectContext && safeADRs.length === 0 && safeTaskHistory.length === 0) return "";

  const lines = ["", "[Project Memory Context - Read Only]"];
  if (hasProjectContext) {
    lines.push("Project Context:");
    for (const [key, value] of Object.entries(projectContext)) {
      lines.push(`- ${key}: ${formatMemoryValue(value)}`);
    }
  }
  if (safeADRs.length > 0) {
    lines.push(`Prior ADR Summaries (${safeADRs.length}):`);
    for (const adr of safeADRs) {
      const adrId = String(adr?.adr_id || "ADR-UNKNOWN");
      const title = String(adr?.title || "untitled");
      const status = String(adr?.status || "unknown");
      lines.push(`- ${adrId} | ${status} | ${title}`);
    }
  }
  if (safeTaskHistory.length > 0) {
    lines.push(`Recent Task History (${safeTaskHistory.length}):`);
    for (const entry of safeTaskHistory) {
      const stepId = String(entry?.step_id || entry?.task_id || "unknown");
      const status = String(entry?.status || "unknown");
      const summary = String(entry?.summary || entry?.result || entry?.note || "").trim();
      lines.push(summary ? `- ${stepId} | ${status} | ${summary}` : `- ${stepId} | ${status}`);
    }
  }
  return lines.join("\n");
}

/**
 * Create a step payload builder bound to engine-level config.
 * @param {{ registry, promptScriptRegistry, handoffContracts }} config
 */
export function createStepBuilder({ registry, promptScriptRegistry, handoffContracts, workspaceRoot = ".", runtimeConfig = {} }) {
  const contextBudgetService = createContextBudgetService();
  function buildStepPayload({ run, stepDef, stepIndex }) {
    const input = parseJsonSafe(run.input_json, {});
    const artifactRoot = pathForRunArtifacts(run.run_id);
    const contract = STEP_CONTRACTS[stepDef.id] || null;
    const fastMode = Boolean(input.fast_mode);
    const downstreamHandoff = Object.values(handoffContracts?.handoffs || {}).find(
      (item) => String(item?.from_step || "") === String(stepDef.id || "")
    );
    const upstreamHandoffs = Object.values(handoffContracts?.handoffs || {}).filter(
      (item) => Array.isArray(item?.to_steps) && item.to_steps.includes(String(stepDef.id || ""))
    );
    const expectedArtifacts = Array.from(
      new Set([
        ...(Array.isArray(contract?.required_artifacts) ? contract.required_artifacts : []),
        ...(Array.isArray(downstreamHandoff?.required_artifacts) ? downstreamHandoff.required_artifacts : []),
      ])
    );
    const payload = {
      ...(input.step_payloads?.[stepDef.id] || {}),
      ...(input.default_payload || {}),
      project_type: run.project_type,
      workflow_id: run.workflow_id,
      workflow_run_id: run.workflow_run_id,
      role: stepDef.role,
      step_id: stepDef.id,
      step_index: stepIndex,
      run_id: run.run_id,
      artifact_root: artifactRoot,
      expected_artifacts: expectedArtifacts,
      prompt_script_id: stepDef.prompt_script_id || "",
      prompt_script: null,
      llm_role: String(stepDef.role || ""),
      handoff_contract_out: downstreamHandoff || null,
      handoff_contract_in: upstreamHandoffs,
    };

    const implMode = chooseImplementationMode({
      stepDef,
      input,
      payload,
      runtimeConfig,
      workspaceRoot,
      contextBudgetService,
    });
    payload.execution_mode_requested = implMode.executionModeRequested;
    payload.diff_first_enabled = runtimeConfig?.execution?.diff_first_enabled !== false;
    payload.context_budget_preview = implMode.contextBudgetPreview;
    payload.target_file_context = implMode.targetFileContext;
    payload.prompt_script_id = implMode.promptScriptId;

    if (String(stepDef.id || "") === "impl_be" && payload.execution_mode_requested === "structured_patch") {
      payload.expected_artifacts = ["impl/be_patch_bundle.json", "impl/be_notes.md", "handoff/be_to_fe.json"];
    }
    if (String(stepDef.id || "") === "impl_fe" && payload.execution_mode_requested === "structured_patch") {
      payload.expected_artifacts = ["impl/fe_patch_bundle.json", "impl/fe_notes.md"];
    }

    const effectiveStepDef = { ...stepDef, prompt_script_id: payload.prompt_script_id };
    const promptScript = effectiveStepDef.prompt_script_id
      ? promptScriptRegistry?.scripts?.[effectiveStepDef.prompt_script_id] || null
      : null;
    const promptBinding = validatePromptScriptBinding({ stepDef: effectiveStepDef, promptScriptRegistry, promptScript });
    if (!promptBinding.ok) {
      const err = new Error(promptBinding.detail || "prompt script binding invalid");
      err.code = promptBinding.code || "PROMPT_SCRIPT_BINDING_INVALID";
      throw err;
    }
    payload.prompt_script = promptScript;
    payload.llm_role = String(promptScript?.llm_role || effectiveStepDef.role || "");

    if (String(stepDef.id || "") === "impl_be") {
      const executionPacket = buildBackendExecutionPacket({
        stepDef: effectiveStepDef,
        payload,
        provider: input.provider || payload.provider || "",
        model: input.model || payload.model || "",
      });
      const checked = validateBackendExecutionPacket(executionPacket);
      if (!checked.ok) {
        const err = new Error(`backend execution packet invalid: ${checked.errors.join("; ")}`);
        err.code = "BACKEND_EXECUTION_PACKET_INVALID";
        throw err;
      }
      payload.execution_adapter_packet = executionPacket;
      const toolAdapterRequest = buildCodingExecutorRequest({
        provider: input.provider || payload.provider || "",
        payload,
        executionPacket,
        role: effectiveStepDef.role,
        stepId: effectiveStepDef.id,
      });
      const requestChecked = validateCodingExecutorRequest(toolAdapterRequest);
      if (!requestChecked.ok) {
        const err = new Error(`coding executor request invalid: ${requestChecked.errors.join("; ")}`);
        err.code = "CODING_EXECUTOR_REQUEST_INVALID";
        throw err;
      }
      payload.tool_adapter_request = toolAdapterRequest;
    }

    if (String(stepDef.id || "") === "impl_fe") {
      const executionPacket = buildFrontendExecutionPacket({
        stepDef: effectiveStepDef,
        payload,
        provider: input.provider || payload.provider || "",
        model: input.model || payload.model || "",
      });
      const checked = validateFrontendExecutionPacket(executionPacket);
      if (!checked.ok) {
        const err = new Error(`frontend execution packet invalid: ${checked.errors.join("; ")}`);
        err.code = "FRONTEND_EXECUTION_PACKET_INVALID";
        throw err;
      }
      payload.execution_adapter_packet = executionPacket;
      const toolAdapterRequest = buildCodingExecutorRequest({
        provider: input.provider || payload.provider || "",
        payload,
        executionPacket,
        role: effectiveStepDef.role,
        stepId: effectiveStepDef.id,
      });
      const requestChecked = validateCodingExecutorRequest(toolAdapterRequest);
      if (!requestChecked.ok) {
        const err = new Error(`coding executor request invalid: ${requestChecked.errors.join("; ")}`);
        err.code = "CODING_EXECUTOR_REQUEST_INVALID";
        throw err;
      }
      payload.tool_adapter_request = toolAdapterRequest;
    }

    if (stepDef.tool === "coding.delegate") {
      const runtimeByStep = { pm_spec: 120, arch_design: 180, impl_fe: 240, impl_be: 240, release_pack: 120 };
      payload.task_prompt = payload.task_prompt || buildStepPrompt({ run, stepDef: effectiveStepDef, input, payload, promptScript });
      if (Array.isArray(payload.target_file_context) && payload.target_file_context.length > 0) {
        payload.task_prompt = `${payload.task_prompt}${buildTargetFileContextBlock(payload.target_file_context)}`;
      }

      if (String(stepDef.id || "") === "arch_design") {
        const memoryProjectId = String(run.run_id || run.workflow_run_id || "default");
        const projectContext = getProjectContext(memoryProjectId);
        const priorADRs = getPriorADRs(memoryProjectId);
        const taskHistory = getTaskHistory(memoryProjectId, 5);
        const memoryContext = buildArchitectMemoryContext({ projectContext, priorADRs, taskHistory });
        if (memoryContext) {
          payload.task_prompt = `${payload.task_prompt}${memoryContext}`;
        }
      }

      if (fastMode) {
        const fastNote = [
          "",
          "[Fast Mode]",
          "- Keep output concise and directly actionable.",
          "- Prioritize required artifact paths first; avoid unnecessary long prose.",
        ].join("\n");
        payload.task_prompt = `${payload.task_prompt}${fastNote}`;
      }
      if (!payload.prompt) payload.prompt = payload.task_prompt;
      if (input.provider && !payload.provider) payload.provider = input.provider;
      if (input.model && !payload.model) payload.model = input.model;
      if (!Number.isFinite(Number(payload.max_runtime_s))) {
        const configured = Number(input.max_runtime_s || 0);
        payload.max_runtime_s = configured > 0 ? configured : (runtimeByStep[stepDef.id] || 240);
      }
      if ((stepDef.id === "impl_fe" || stepDef.id === "impl_be") && !Array.isArray(payload.target_paths)) {
        payload.target_paths = ["sandbox/crm_site/"];
      }
      if ((stepDef.id === "impl_be" || stepDef.id === "impl_fe") && payload.execution_adapter_packet) {
        payload.target_paths = payload.execution_adapter_packet.target_paths;
      }
    }

    if (stepDef.gate === "acceptance") {
      const suiteId = registry.project_types?.[run.project_type]?.acceptance_suite;
      const suite = suiteId ? registry.acceptance_suites?.[suiteId] : null;
      const commands = Array.isArray(suite?.commands) ? suite.commands.filter(Boolean) : [];
      if (!payload.command && commands.length > 0) payload.command = commands.join(" && ");
      payload.acceptance_suite_id = suiteId || payload.acceptance_suite_id || "";
      payload.required_reports = payload.required_reports || suite?.required_reports || [];
      payload.acceptance_context = {
        step_id: stepDef.id,
        role: stepDef.role,
        goal: String(input.goal || ""),
        required_artifacts: payload.expected_artifacts || [],
        prompt_script_id: effectiveStepDef.prompt_script_id || "",
        prompt_script: promptScript,
      };
    }

    return payload;
  }

  return { buildStepPayload };
}
