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
import { createRepoContextService } from "./repo_context_service.js";
import {
  applyWorkerCodingTemplateDefaults,
  loadWorkerCodingTemplateRegistryOrThrow,
} from "../worker_coding_templates.js";

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

function buildCodingContextBlock({ contextPacket = null, repoMap = null }) {
  if (!contextPacket || typeof contextPacket !== "object") return "";
  const lines = ["", "[Coding Context Packet]"];
  lines.push(`Step: ${String(contextPacket.step_id || "")}`);
  lines.push(`Role: ${String(contextPacket.role || "")}`);
  if (Array.isArray(contextPacket.target_paths) && contextPacket.target_paths.length > 0) {
    lines.push(`Target Paths: ${contextPacket.target_paths.join(", ")}`);
  }
  if (Array.isArray(contextPacket.entrypoints) && contextPacket.entrypoints.length > 0) {
    lines.push(`Entrypoints: ${contextPacket.entrypoints.join(", ")}`);
  }
  if (Array.isArray(contextPacket.related_tests) && contextPacket.related_tests.length > 0) {
    lines.push(`Related Tests: ${contextPacket.related_tests.join(", ")}`);
  }
  if (Array.isArray(contextPacket.recent_changed_files) && contextPacket.recent_changed_files.length > 0) {
    lines.push(`Recent Changed Files: ${contextPacket.recent_changed_files.join(", ")}`);
  }
  if (Array.isArray(contextPacket.memory_hints) && contextPacket.memory_hints.length > 0) {
    lines.push("Memory Hints:");
    for (const hint of contextPacket.memory_hints) lines.push(`- ${String(hint)}`);
  }
  if (repoMap && Array.isArray(repoMap.candidate_files) && repoMap.candidate_files.length > 0) {
    lines.push(`Candidate Files: ${repoMap.candidate_files.slice(0, 12).join(", ")}`);
  }
  if (repoMap && Array.isArray(repoMap.key_config_files) && repoMap.key_config_files.length > 0) {
    lines.push(`Key Config Files: ${repoMap.key_config_files.join(", ")}`);
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
  const requestedLane = String(payload?.execution_lane || "").trim();
  // stable_local_lane uses a local ollama model with a limited context window.
  // Structured patch (diff-first) requires injecting target file content which
  // can overflow that budget. Force full_file_fallback to keep the prompt small.
  if (["impl_be", "impl_fe"].includes(String(stepDef?.id || "")) && requestedLane === "stable_local_lane") {
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

function clampInt(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.trunc(n)));
}

function shouldEnforceStableCodingLane({ run, stepDef }) {
  return String(run?.workflow_id || "") === "coding_team_v0"
    && String(run?.project_type || "") === "webapp_crm"
    && String(stepDef?.tool || "") === "coding.delegate";
}

function applyStableCodingLaneDefaults({ run, stepDef, payload, input, runtimeConfig }) {
  const runtimeWorkerCoder = runtimeConfig?.worker_coder || {};
  const laneRegistry = runtimeWorkerCoder.execution_lanes && typeof runtimeWorkerCoder.execution_lanes === "object"
    ? runtimeWorkerCoder.execution_lanes
    : {};
  if (!shouldEnforceStableCodingLane({ run, stepDef })) {
    return;
  }

  const configuredLane = String(
    payload.execution_lane
      || input.execution_lane
      || runtimeWorkerCoder.execution_lane_default
      || ""
  ).trim();
  if (!configuredLane) {
    return;
  }

  const laneConfig = laneRegistry[configuredLane] && typeof laneRegistry[configuredLane] === "object"
    ? laneRegistry[configuredLane]
    : null;
  payload.execution_lane = configuredLane;
  if (laneConfig?.provider) {
    payload.provider = String(laneConfig.provider);
  } else if (!payload.provider && input.provider) {
    payload.provider = input.provider;
  }
  if (laneConfig?.model) {
    payload.model = String(laneConfig.model);
  } else if (!payload.model && input.model) {
    payload.model = input.model;
  }
  if (!payload.wall_clock_timeout_s && runtimeWorkerCoder.wall_clock_timeout_s_default) {
    payload.wall_clock_timeout_s = Number(runtimeWorkerCoder.wall_clock_timeout_s_default);
  }
  if (!payload.max_attempts && runtimeWorkerCoder.max_attempts_default) {
    payload.max_attempts = Number(runtimeWorkerCoder.max_attempts_default);
  }
  if (!payload.same_error_repeat_limit && runtimeWorkerCoder.same_error_repeat_limit_default) {
    payload.same_error_repeat_limit = Number(runtimeWorkerCoder.same_error_repeat_limit_default);
  }
}

function inferVerificationCommand({ stepId, targetPaths = [], workspaceRoot }) {
  const safeStepId = String(stepId || "");
  const safeTargets = Array.isArray(targetPaths) ? targetPaths : [];
  for (const targetPath of safeTargets) {
    const rel = String(targetPath || "").replace(/\\/g, "/").replace(/^\/+/, "");
    if (!rel) continue;
    const abs = path.resolve(workspaceRoot, rel);
    if (!abs.startsWith(path.resolve(workspaceRoot)) || !fs.existsSync(abs)) continue;
    if (fs.statSync(abs).isDirectory()) {
      const entries = fs.readdirSync(abs, { withFileTypes: true });
      for (const entry of entries) {
        if (!entry.isFile()) continue;
        const relFile = path.posix.join(rel, entry.name);
        if (entry.name.endsWith(".js") && safeStepId === "impl_fe") return `node --check ${relFile}`;
        if (entry.name.endsWith(".js") && safeStepId === "impl_be") return `node --check ${relFile}`;
        if (entry.name.endsWith(".py")) return `python -m py_compile ${relFile}`;
      }
    } else if (fs.statSync(abs).isFile()) {
      if (rel.endsWith(".js")) return `node --check ${rel}`;
      if (rel.endsWith(".py")) return `python -m py_compile ${rel}`;
    }
  }
  return "";
}

function findNearestPackageJson({ workspaceRoot, targetPaths = [] }) {
  const workspaceAbs = path.resolve(workspaceRoot);
  const candidates = new Set();
  for (const targetPath of targetPaths) {
    const rel = String(targetPath || "").replace(/\\/g, "/").replace(/^\/+/, "");
    if (!rel) continue;
    let current = path.resolve(workspaceAbs, rel);
    if (fs.existsSync(current) && fs.statSync(current).isFile()) {
      current = path.dirname(current);
    }
    while (current.startsWith(workspaceAbs)) {
      candidates.add(path.join(current, "package.json"));
      if (current === workspaceAbs) break;
      current = path.dirname(current);
    }
  }
  candidates.add(path.join(workspaceAbs, "package.json"));
  for (const candidate of candidates) {
    try {
      if (!fs.existsSync(candidate)) continue;
      const parsed = JSON.parse(fs.readFileSync(candidate, "utf8"));
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return {
          path: candidate,
          dir: path.dirname(candidate),
          json: parsed,
        };
      }
    } catch {
      continue;
    }
  }
  return null;
}

function resolveTemplateVerificationPlan({ stepId, payload, workspaceRoot }) {
  const tiers = Array.isArray(payload?.template_verification_tiers)
    ? payload.template_verification_tiers.map((item) => String(item || "").trim().toLowerCase()).filter(Boolean)
    : [];
  const targetPaths = Array.isArray(payload?.target_paths) ? payload.target_paths : [];
  const packageInfo = findNearestPackageJson({ workspaceRoot, targetPaths });
  const scripts = packageInfo?.json?.scripts && typeof packageInfo.json.scripts === "object"
    ? packageInfo.json.scripts
    : {};
  const plan = [];
  const syntaxCommand = inferVerificationCommand({ stepId, targetPaths, workspaceRoot });
  if (syntaxCommand) {
    plan.push({
      tier: "syntax_check",
      command: syntaxCommand,
      required: true,
      source: "inferred_target_paths",
    });
  }

  const scriptMappings = {
    lint: ["lint"],
    type_check: ["typecheck", "type-check", "check-types"],
    unit_test: ["test", "unit", "unit:test"],
    build: ["build"],
  };

  for (const tier of tiers) {
    const scriptNames = scriptMappings[tier] || [];
    const matchedScript = scriptNames.find((name) => typeof scripts[name] === "string" && scripts[name].trim());
    const relPackageDir = packageInfo
      ? path.relative(workspaceRoot, packageInfo.dir).replace(/\\/g, "/")
      : "";
    const scriptCommand = matchedScript
      ? relPackageDir && relPackageDir !== "."
        ? `npm --prefix ${relPackageDir} run ${matchedScript}`
        : `npm run ${matchedScript}`
      : "";
    plan.push({
      tier,
      command: scriptCommand,
      required: false,
      source: matchedScript
        ? `package_json:${path.relative(workspaceRoot, packageInfo.path).replace(/\\/g, "/")}`
        : "template_declared_unresolved",
    });
  }
  return plan;
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
  const repoContextService = createRepoContextService({ workspaceRoot });
  const workerCodingTemplateRegistry = loadWorkerCodingTemplateRegistryOrThrow();
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
    applyStableCodingLaneDefaults({ run, stepDef, payload, input, runtimeConfig });
    applyWorkerCodingTemplateDefaults({
      payload,
      templateRegistry: workerCodingTemplateRegistry,
      stepDef,
    });

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
    payload.context_packet = null;
    payload.repo_map = null;

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
    }

    if (stepDef.tool === "coding.delegate") {
      const runtimeByStep = { pm_spec: 360, arch_design: 480, impl_fe: 360, impl_be: 240, release_pack: 120 };
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
      const runtimeWorkerCoder = runtimeConfig?.worker_coder || {};
      if ((stepDef.id === "impl_fe" || stepDef.id === "impl_be") && !Array.isArray(payload.target_paths)) {
        payload.target_paths = ["sandbox/crm_site/"];
      }
      if ((stepDef.id === "impl_be" || stepDef.id === "impl_fe") && payload.execution_adapter_packet) {
        payload.target_paths = payload.execution_adapter_packet.target_paths;
      }
      if (["impl_be", "impl_fe"].includes(String(stepDef.id || ""))) {
        if (!Array.isArray(payload.verification_plan) || payload.verification_plan.length === 0) {
          payload.verification_plan = resolveTemplateVerificationPlan({
            stepId: stepDef.id,
            payload,
            workspaceRoot,
          });
        }
        if (!String(payload.verification_command || "").trim()) {
          const runtimeDefaultCommand = String(runtimeWorkerCoder.verification_command_default || "").trim();
          const firstPlanCommand = Array.isArray(payload.verification_plan)
            ? payload.verification_plan.find((item) => String(item?.command || "").trim())
            : null;
          payload.verification_command = runtimeDefaultCommand || String(firstPlanCommand?.command || "").trim() || inferVerificationCommand({
            stepId: stepDef.id,
            targetPaths: payload.target_paths,
            workspaceRoot,
          });
        }
        payload.max_attempts = clampInt(
          payload.max_attempts ?? input.max_attempts ?? runtimeWorkerCoder.max_attempts_default,
          1,
          3,
          2,
        );
        payload.same_error_repeat_limit = clampInt(
          payload.same_error_repeat_limit ?? input.same_error_repeat_limit ?? runtimeWorkerCoder.same_error_repeat_limit_default,
          1,
          3,
          2,
        );
        payload.wall_clock_timeout_s = clampInt(
          payload.wall_clock_timeout_s ?? input.wall_clock_timeout_s ?? runtimeWorkerCoder.wall_clock_timeout_s_default,
          Math.max(30, Number(payload.max_runtime_s || 30)),
          3600,
          Math.max(Number(payload.max_runtime_s || 30), 300),
        );
      }
      if (["impl_be", "impl_fe", "qa_verify"].includes(String(stepDef.id || ""))) {
        const memoryProjectId = String(run.run_id || run.workflow_run_id || "default");
        const recentTaskHistory = getTaskHistory(memoryProjectId, 3);
        const memoryHints = recentTaskHistory.map((entry) => {
          const stepId = String(entry?.step_id || entry?.task_id || "unknown");
          const status = String(entry?.status || "unknown");
          const summary = String(entry?.summary || entry?.result || entry?.note || "").trim();
          return summary ? `${stepId} | ${status} | ${summary}` : `${stepId} | ${status}`;
        });
        payload.repo_map = repoContextService.buildRepoMap({
          targetPaths: payload.target_paths,
          recentChangedFiles: payload.target_paths,
        });
        payload.context_packet = repoContextService.buildContextPacket({
          role: effectiveStepDef.role,
          stepId: effectiveStepDef.id,
          targetPaths: payload.target_paths,
          recentChangedFiles: payload.target_paths,
          memoryHints,
        });
        payload.task_prompt = `${payload.task_prompt}${buildCodingContextBlock({
          contextPacket: payload.context_packet,
          repoMap: payload.repo_map,
        })}`;
      }
      if (["impl_be", "impl_fe"].includes(String(stepDef.id || "")) && payload.execution_adapter_packet) {
        const toolAdapterRequest = buildCodingExecutorRequest({
          provider: input.provider || payload.provider || "",
          payload,
          executionPacket: payload.execution_adapter_packet,
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
