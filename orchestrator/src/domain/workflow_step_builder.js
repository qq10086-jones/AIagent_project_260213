/**
 * workflow_step_builder.js
 *
 * Step payload construction helpers.
 * Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "../schema_lite_validator.js";
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

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CODING_TEAM_WORKPLAN_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(MODULE_DIR, "..", "..", "contracts", "coding_team_workplan.schema.json"), "utf8")
);

export function pathForRunArtifacts(run_id) {
  return `runtime/artifacts/release/${run_id || "unknown-run"}`;
}

function defaultCodingTargetPaths(projectType) {
  if (String(projectType || "") === "webapp_crm") return ["workspace/sandbox/crm_site/"];
  if (String(projectType || "") === "single_file_html") return ["workspace/sandbox/html_site/"];
  if (String(projectType || "") === "generic_app") return ["workspace/sandbox/app/"];
  return ["workspace/sandbox/project/"];
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

function buildArchHandoffBlock(handoff, { stepId = "" } = {}) {
  if (!handoff || typeof handoff !== "object") return "";
  const lines = [
    "",
    "[Architecture Handoff — THIS IS YOUR PRIMARY SPECIFICATION]",
    "IMPORTANT: Implement EXACTLY what this handoff specifies. Do NOT copy patterns from existing sandbox files if they differ from this spec.",
  ];
  if (Array.isArray(handoff.modules) && handoff.modules.length > 0) {
    lines.push(`Modules: ${handoff.modules.map((m) => String(m)).join(", ")}`);
  }
  if (Array.isArray(handoff.interfaces) && handoff.interfaces.length > 0) {
    lines.push("API interfaces to implement:");
    for (const iface of handoff.interfaces) lines.push(`- ${String(iface)}`);
  }
  if (Array.isArray(handoff.decisions) && handoff.decisions.length > 0) {
    lines.push("Architecture decisions (MUST follow these):");
    for (const dec of handoff.decisions) {
      const parts = [dec?.adr_id, dec?.title, dec?.status].filter(Boolean).map(String);
      lines.push(`- ${parts.join(" | ")}`);
    }
  }
  if (Array.isArray(handoff.risks) && handoff.risks.length > 0) {
    lines.push("Risks:");
    for (const risk of handoff.risks) lines.push(`- ${String(risk)}`);
  }
  // R1: Include workplan tasks so LLM knows exactly what to build
  const wp = handoff.workplan;
  if (wp && typeof wp === "object") {
    const side = stepId === "impl_fe" ? "fe_tasks" : "be_tasks";
    const tasks = Array.isArray(wp[side]) ? wp[side] : [];
    if (tasks.length > 0) {
      lines.push(`\nWorkplan tasks to implement (${side}):`);
      for (const t of tasks) {
        const id = String(t?.id || "").trim();
        const desc = String(t?.description || "").trim();
        const verify = String(t?.verify || "").trim();
        lines.push(`- ${id || "TASK"}: ${desc}${verify ? ` [verify: ${verify}]` : ""}`);
      }
      lines.push("Execute these tasks IN ORDER. Each task's verify condition must pass before proceeding.");
    }
  }
  return lines.length > 3 ? lines.join("\n") : "";
}

function buildStructuredWorkplanBlock(tasks = [], sectionLabel = "") {
  const safeTasks = Array.isArray(tasks) ? tasks.filter((item) => item && typeof item === "object") : [];
  if (safeTasks.length === 0) return "";
  const lines = [``, `[Structured Workplan — ${sectionLabel}]`];
  for (const task of safeTasks) {
    const id = String(task.id || "").trim();
    const description = String(task.description || "").trim();
    const verify = String(task.verify || "").trim();
    if (!id && !description) continue;
    lines.push(`- ${id || "TASK"}: ${description || "No description provided"}${verify ? ` | verify: ${verify}` : ""}`);
  }
  lines.push("Execute tasks in order. After completing each task, self-check against its verify condition before proceeding to the next.");
  return lines.join("\n");
}

function normalizeInjectedWorkplan(tasks = [], sectionLabel = "", sourcePath = "plan/workplan.json") {
  const safeTasks = Array.isArray(tasks)
    ? tasks
        .filter((item) => item && typeof item === "object")
        .map((item) => ({
          id: String(item.id || "").trim(),
          description: String(item.description || "").trim(),
          verify: String(item.verify || "").trim(),
        }))
        .filter((item) => item.id || item.description)
    : [];
  if (safeTasks.length === 0) return null;
  return {
    section: sectionLabel,
    source: String(sourcePath || "plan/workplan.json"),
    tasks: safeTasks,
  };
}

function readStructuredWorkplan({ workspaceRoot, artifactRoot, stepId }) {
  const workplanJsonPath = path.resolve(workspaceRoot, artifactRoot, "plan/workplan.json");
  if (!fs.existsSync(workplanJsonPath)) {
    return { block: "", injected: null, validation: { checked: false, ok: true, code: null, errors: [] } };
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(workplanJsonPath, "utf8"));
    const errors = validateJsonSchemaLite(CODING_TEAM_WORKPLAN_SCHEMA, parsed, "$");
    if (errors.length > 0) {
      return {
        block: "",
        injected: null,
        validation: {
          checked: true,
          ok: false,
          code: "WORKPLAN_JSON_INVALID",
          errors,
          path: path.relative(workspaceRoot, workplanJsonPath).replace(/\\/g, "/"),
        },
      };
    }
    const tasks = String(stepId || "") === "impl_be" ? parsed?.be_tasks : parsed?.fe_tasks;
    const sectionLabel = String(stepId || "") === "impl_be" ? "BE Tasks" : "FE Tasks";
    return {
      block: buildStructuredWorkplanBlock(tasks, sectionLabel),
      injected: normalizeInjectedWorkplan(
        tasks,
        sectionLabel,
        path.relative(workspaceRoot, workplanJsonPath).replace(/\\/g, "/"),
      ),
      validation: {
        checked: true,
        ok: true,
        code: null,
        errors: [],
        path: path.relative(workspaceRoot, workplanJsonPath).replace(/\\/g, "/"),
      },
    };
  } catch {
    return {
      block: "",
      injected: null,
      validation: {
        checked: true,
        ok: false,
        code: "WORKPLAN_JSON_PARSE_FAILED",
        errors: ["plan/workplan.json is not valid JSON"],
        path: path.relative(workspaceRoot, workplanJsonPath).replace(/\\/g, "/"),
      },
    };
  }
}

function buildWorkplanFallbackNotice(validation = {}) {
  if (!validation?.checked || validation?.ok) return "";
  const lines = [
    "",
    "[Workplan JSON Fallback]",
    `Structured workplan unavailable: ${String(validation.code || "WORKPLAN_JSON_INVALID")}`,
  ];
  if (Array.isArray(validation.errors) && validation.errors.length > 0) {
    lines.push(...validation.errors.slice(0, 5).map((item) => `- ${String(item)}`));
  }
  lines.push("Fallback to plan/workplan.md if present.");
  return lines.join("\n");
}

function buildCodingContextBlock({ contextPacket = null, repoMap = null, hasArchHandoff = false }) {
  if (!contextPacket || typeof contextPacket !== "object") return "";
  const lines = ["", "[Coding Context Packet — Reference Only]"];
  if (hasArchHandoff) {
    lines.push("NOTE: These are existing sandbox files for REFERENCE ONLY. The Architecture Handoff above is your PRIMARY specification. If sandbox files use a different data model or domain, build fresh per the handoff spec — do NOT copy old patterns.");
  }
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
  // stable_cloud_lane currently shows weak structured-patch reliability on impl steps.
  // primary_minimax_lane stays on full_file_fallback to keep prompts smaller and stable.
  // Force full_file_fallback on these lanes to keep prompts smaller and avoid empty patch bundles.
  if (["impl_be", "impl_fe"].includes(String(stepDef?.id || "")) && ["stable_local_lane", "stable_cloud_lane", "primary_minimax_lane"].includes(requestedLane)) {
    return {
      executionModeRequested: "full_file_fallback",
      promptScriptId: String(stepDef?.prompt_script_id || ""),
      targetFileContext: [],
      contextBudgetPreview: null,
    };
  }
  const diffFirstEnabled = runtimeConfig?.execution?.diff_first_enabled !== false;
  const targetPaths = Array.isArray(payload?.target_paths) ? payload.target_paths : defaultCodingTargetPaths(payload?.project_type);
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

function readTextSafe(absPath) {
  try {
    return fs.readFileSync(absPath, "utf8");
  } catch {
    return "";
  }
}

function readJsonSafe(absPath) {
  try {
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

function inferSmokeApiEndpoint({ workspaceRoot, artifactRoot }) {
  const rootAbs = path.resolve(workspaceRoot, String(artifactRoot || ""));
  const handoff = readJsonSafe(path.join(rootAbs, "handoff", "be_to_fe.json"));
  const contracts = Array.isArray(handoff?.api_contracts) ? handoff.api_contracts : [];
  const preferredGet = contracts.find((item) => String(item?.method || "").toUpperCase() === "GET" && String(item?.path || "").startsWith("/api/"));
  if (preferredGet) return String(preferredGet.path);
  const firstApi = contracts.find((item) => String(item?.path || "").startsWith("/api/"));
  if (firstApi) return String(firstApi.path);

  const interfacesText = readTextSafe(path.join(rootAbs, "plan", "interfaces.md"));
  const match = interfacesText.match(/^##\s+(GET|POST|PUT|PATCH|DELETE)\s+([^\s#]+)/im);
  if (match?.[2] && match[2].startsWith("/api/")) return String(match[2]);
  return "";
}

function buildSmokeTestCommand({ workspaceRoot, artifactRoot, port = 13099 }) {
  const workspaceScriptAbs = path.resolve(workspaceRoot, "orchestrator", "scripts", "run_smoke_test.mjs");
  const fallbackScriptAbs = path.resolve(MODULE_DIR, "..", "..", "scripts", "run_smoke_test.mjs");
  const scriptAbs = fs.existsSync(workspaceScriptAbs) ? workspaceScriptAbs : fallbackScriptAbs;
  const scriptRel = path.relative(workspaceRoot, scriptAbs).replace(/\\/g, "/");
  const apiEndpoint = inferSmokeApiEndpoint({ workspaceRoot, artifactRoot });
  const args = [
    "node",
    JSON.stringify(scriptRel),
    "--artifact-root",
    JSON.stringify(String(artifactRoot || "")),
    "--port",
    String(port),
  ];
  if (apiEndpoint) {
    args.push("--api-endpoint", JSON.stringify(apiEndpoint));
  }
  return args.join(" ");
}

function buildAcceptanceTestCommand({ workspaceRoot, artifactRoot }) {
  const workspaceScriptAbs = path.resolve(workspaceRoot, "orchestrator", "scripts", "run_acceptance_test.mjs");
  const fallbackScriptAbs = path.resolve(MODULE_DIR, "..", "..", "scripts", "run_acceptance_test.mjs");
  const scriptAbs = fs.existsSync(workspaceScriptAbs) ? workspaceScriptAbs : fallbackScriptAbs;
  const scriptRel = path.relative(workspaceRoot, scriptAbs).replace(/\\/g, "/");
  return [
    "node",
    JSON.stringify(scriptRel),
    "--artifact-root",
    JSON.stringify(String(artifactRoot || "")),
    "--workspace-root",
    JSON.stringify("."),
  ].join(" ");
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

function shouldEnforceGenericAppPrimaryLane({ run, stepDef }) {
  return String(run?.workflow_id || "") === "coding_team_v0"
    && ["generic_app", "single_file_html", "generic_coding_task"].includes(String(run?.project_type || ""))
    && ["impl_be", "impl_fe"].includes(String(stepDef?.id || ""))
    && String(stepDef?.tool || "") === "coding.delegate";
}

function applyGenericAppPrimaryLaneDefaults({ run, stepDef, payload, runtimeConfig }) {
  if (!shouldEnforceGenericAppPrimaryLane({ run, stepDef })) return;
  const runtimeWorkerCoder = runtimeConfig?.worker_coder || {};
  const laneRegistry = runtimeWorkerCoder.execution_lanes && typeof runtimeWorkerCoder.execution_lanes === "object"
    ? runtimeWorkerCoder.execution_lanes
    : {};
  const targetLane = "primary_minimax_lane";
  const laneConfig = laneRegistry[targetLane] && typeof laneRegistry[targetLane] === "object"
    ? laneRegistry[targetLane]
    : null;
  if (!laneConfig) return;
  payload.execution_lane = targetLane;
  if (laneConfig.provider) payload.provider = String(laneConfig.provider);
  if (laneConfig.model) payload.model = String(laneConfig.model);
}

function getExecutionLaneConfig(runtimeConfig, laneId) {
  const runtimeWorkerCoder = runtimeConfig?.worker_coder || {};
  const laneRegistry = runtimeWorkerCoder.execution_lanes && typeof runtimeWorkerCoder.execution_lanes === "object"
    ? runtimeWorkerCoder.execution_lanes
    : {};
  const laneConfig = laneRegistry[laneId] && typeof laneRegistry[laneId] === "object"
    ? laneRegistry[laneId]
    : null;
  return laneConfig;
}

function applyStepModelRoutingDefaults({ run, stepDef, payload, runtimeConfig }) {
  if (String(run?.workflow_id || "") !== "coding_team_v0") return;
  if (String(run?.project_type || "") !== "webapp_crm") return;

  const lowRiskCodingSteps = {
    release_pack: "primary_minimax_lane",
  };
  const targetLane = lowRiskCodingSteps[String(stepDef?.id || "")] || "";
  if (targetLane && String(stepDef?.tool || "") === "coding.delegate") {
    const laneConfig = getExecutionLaneConfig(runtimeConfig, targetLane);
    payload.execution_lane = targetLane;
    if (laneConfig?.provider) payload.provider = String(laneConfig.provider);
    const resolvedModel = String(laneConfig?.model || payload.model_override || payload.model || "minimax-coding-plan/MiniMax-M2.7");
    payload.model = resolvedModel;
    payload.model_override = resolvedModel;
    return;
  }

  if (String(stepDef?.id || "") === "deploy_preview") {
    const laneConfig = getExecutionLaneConfig(runtimeConfig, "primary_minimax_lane");
    // deploy_preview is currently an ops tool, not an LLM task. Keep the planned lane
    // as metadata so run artifacts remain aligned with the design intent.
    payload.model_override = String(laneConfig?.model || payload.model_override || "minimax-coding-plan/MiniMax-M2.7");
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
export function createStepBuilder({ registry, promptScriptRegistry, handoffContracts, workspaceRoot = ".", runtimeConfig = {}, redis = null }) {
  const contextBudgetService = createContextBudgetService();
  const repoContextService = createRepoContextService({ workspaceRoot, redis });
  const workerCodingTemplateRegistry = loadWorkerCodingTemplateRegistryOrThrow();
  async function buildStepPayload({ run, stepDef, stepIndex }) {
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
    // Phase 1.5-4: Extract lineage from run input for refinement re-entry
    const inputLineage = input.lineage && typeof input.lineage === "object" ? input.lineage : null;

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
      lineage: inputLineage,
      is_refinement: !!inputLineage,
    };
    applyStableCodingLaneDefaults({ run, stepDef, payload, input, runtimeConfig });
    applyGenericAppPrimaryLaneDefaults({ run, stepDef, payload, runtimeConfig });
    applyStepModelRoutingDefaults({ run, stepDef, payload, runtimeConfig });
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
    payload.workplan_validation = { checked: false, ok: true, code: null, errors: [] };
    payload.injected_workplan = null;

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

    // Set target_paths BEFORE building execution packets so they get the correct project-specific paths.
    if (["impl_be", "impl_fe"].includes(String(stepDef.id || "")) && !Array.isArray(payload.target_paths)) {
      payload.target_paths = defaultCodingTargetPaths(payload.project_type);
    }

    if (String(stepDef.id || "") === "impl_be") {
      const executionPacket = buildBackendExecutionPacket({
        stepDef: effectiveStepDef,
        payload,
        provider: input.provider || payload.provider || "",
        model: payload.model_override || input.model || payload.model || "",
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
        model: payload.model_override || input.model || payload.model || "",
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
      const runtimeByStep = { pm_spec: 360, arch_design: 480, impl_fe: 360, impl_be: 240, smoke_test: 120, qa_verify: 180, release_pack: 120 };
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
        // Reinforce handoff schema — LLM often drops workplan/risks fields
        const handoffReminder = [
          "",
          "[CRITICAL: handoff/architect_to_impl.json REQUIRED FIELDS]",
          "Your handoff JSON MUST have ALL of these top-level keys or validation WILL fail:",
          '  "from_step": "arch_design",',
          '  "to_steps": ["impl_be", "impl_fe"],',
          '  "modules": ["..."],        // array of strings',
          '  "interfaces": ["..."],     // array of strings',
          '  "decisions": [{...}],      // array of {adr_id, title, status}',
          '  "risks": ["..."],          // array of strings — DO NOT OMIT',
          '  "workplan": {              // DO NOT OMIT, DO NOT RENAME',
          '    "be_tasks": [{"id":"T-BE-1","description":"...","verify":"..."}],',
          '    "fe_tasks": [{"id":"T-FE-1","description":"...","verify":"..."}]',
          "  }",
          "Copy workplan content from plan/workplan.json into the handoff.",
        ].join("\n");
        payload.task_prompt = `${payload.task_prompt}${handoffReminder}`;
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
      if (payload.model_override && !payload.model) payload.model = payload.model_override;
      if (input.model && !payload.model) payload.model = input.model;
      if (!Number.isFinite(Number(payload.max_runtime_s))) {
        const configured = Number(input.max_runtime_s || 0);
        payload.max_runtime_s = configured > 0 ? configured : (runtimeByStep[stepDef.id] || 240);
      }
      const runtimeWorkerCoder = runtimeConfig?.worker_coder || {};
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
      // R3: For impl steps, inject handoff + workplan BEFORE context so LLM sees spec first
      if (["impl_be", "impl_fe"].includes(String(stepDef.id || ""))) {
        const archHandoffPath = path.resolve(workspaceRoot, artifactRoot, "handoff/architect_to_impl.json");
        if (fs.existsSync(archHandoffPath)) {
          try {
            const archHandoff = JSON.parse(fs.readFileSync(archHandoffPath, "utf8"));
            const archBlock = buildArchHandoffBlock(archHandoff, { stepId: stepDef.id });
            if (archBlock) payload.task_prompt = `${payload.task_prompt}${archBlock}`;
          } catch { /* ignore read errors */ }
        }
        const structuredWorkplan = readStructuredWorkplan({ workspaceRoot, artifactRoot, stepId: stepDef.id });
        payload.workplan_validation = structuredWorkplan.validation;
        payload.injected_workplan = structuredWorkplan.injected || null;
        if (structuredWorkplan.block) {
          payload.task_prompt = `${payload.task_prompt}${structuredWorkplan.block}`;
        } else {
          const fallbackNotice = buildWorkplanFallbackNotice(structuredWorkplan.validation);
          if (fallbackNotice) payload.task_prompt = `${payload.task_prompt}${fallbackNotice}`;
          const workplanPath = path.resolve(workspaceRoot, artifactRoot, "plan/workplan.md");
          if (fs.existsSync(workplanPath)) {
          try {
            const workplan = fs.readFileSync(workplanPath, "utf8");
            const sectionLabel = stepDef.id === "impl_be" ? "BE Tasks" : "FE Tasks";
            const sectionMatch = workplan.match(new RegExp(`##\\s*${sectionLabel}([\\s\\S]*?)(?=\\n##\\s|$)`));
            if (sectionMatch) {
              const taskList = sectionMatch[1].trim();
              if (taskList) {
                payload.task_prompt = `${payload.task_prompt}\n\n[Task List from plan/workplan.md — ${sectionLabel}]\n${taskList}\nExecute tasks in order. After completing each task, self-check against its verify condition before proceeding to the next.`;
              }
            }
          } catch { /* graceful fallback — workplan read failure does not block step */ }
          }
        }
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
        payload.repo_map = await repoContextService.buildRepoMap({
          targetPaths: payload.target_paths,
          recentChangedFiles: payload.target_paths,
        });
        payload.context_packet = await repoContextService.buildContextPacket({
          role: effectiveStepDef.role,
          stepId: effectiveStepDef.id,
          targetPaths: payload.target_paths,
          recentChangedFiles: payload.target_paths,
          memoryHints,
        });
        const hasArchHandoff = ["impl_be", "impl_fe"].includes(String(stepDef.id || ""))
          && fs.existsSync(path.resolve(workspaceRoot, artifactRoot, "handoff/architect_to_impl.json"));
        payload.task_prompt = `${payload.task_prompt}${buildCodingContextBlock({
          contextPacket: payload.context_packet,
          repoMap: payload.repo_map,
          hasArchHandoff,
        })}`;
      }
      if (String(stepDef.id || "") === "qa_verify") {
        const acceptancePath = path.resolve(workspaceRoot, artifactRoot, "plan/acceptance.json");
        if (fs.existsSync(acceptancePath)) {
          try {
            const acceptanceRaw = fs.readFileSync(acceptancePath, "utf8");
            const acceptanceParsed = JSON.parse(acceptanceRaw);
            const criteria = Array.isArray(acceptanceParsed?.criteria)
              ? acceptanceParsed.criteria
              : Array.isArray(acceptanceParsed)
              ? acceptanceParsed
              : [];
            if (criteria.length > 0) {
              const criteriaBlock = [
                "",
                "[Acceptance Criteria from plan/acceptance.json]",
                ...criteria.map((c, i) => `${i + 1}. ${typeof c === "string" ? c : String(c?.description || c?.criterion || JSON.stringify(c))}`),
                "Evaluate each criterion against the impl files. Include a checks entry for each.",
              ].join("\n");
              payload.task_prompt = `${payload.task_prompt}${criteriaBlock}`;
            }
          } catch { /* ignore parse errors */ }
        }
        // v3.3 Patch B: inject acceptance_result.json (deterministic test results)
        const acceptanceResultPath = path.resolve(workspaceRoot, artifactRoot, "verify/acceptance_result.json");
        if (fs.existsSync(acceptanceResultPath)) {
          try {
            const acceptResult = JSON.parse(fs.readFileSync(acceptanceResultPath, "utf8"));
            const detResults = Array.isArray(acceptResult?.criteria_results)
              ? acceptResult.criteria_results.filter((r) => r.verify_tier === "deterministic")
              : [];
            if (detResults.length > 0) {
              const detBlock = [
                "",
                "[Deterministic Acceptance Test Results — DO NOT override these verdicts]",
                `Verdict: ${acceptResult.verdict} (${acceptResult.deterministic_pass} pass, ${acceptResult.deterministic_fail} fail)`,
                ...detResults.map((r) => `- ${r.id}: ${r.status.toUpperCase()} — ${r.description}${r.error ? ` (error: ${r.error})` : ""}`),
                "",
                "IMPORTANT: If a deterministic criterion FAILED above, your qa_report must mark it as 'fail'. Do not override deterministic results.",
                "Focus your semantic review on criteria NOT covered by deterministic tests.",
                "You MUST include journey_checks (at least 1) and rubric_citations (at least 1) in your report.",
                "Include acceptance_test_results summary from the data above.",
              ].join("\n");
              payload.task_prompt = `${payload.task_prompt}${detBlock}`;
            }
            payload.acceptance_test_results = {
              verdict: acceptResult.verdict,
              deterministic_pass: acceptResult.deterministic_pass || 0,
              deterministic_fail: acceptResult.deterministic_fail || 0,
              semantic_pending: acceptResult.semantic_pending || 0,
            };
          } catch { /* ignore parse errors */ }
        }
        const implFeDir = path.resolve(workspaceRoot, artifactRoot, "impl/fe_changes");
        const implBeDir = path.resolve(workspaceRoot, artifactRoot, "impl/be_changes");
        const implFiles = [];
        for (const dir of [implFeDir, implBeDir]) {
          if (fs.existsSync(dir) && fs.statSync(dir).isDirectory()) {
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const entry of entries) {
              if (entry.isFile()) implFiles.push(path.relative(workspaceRoot, path.join(dir, entry.name)).replace(/\\/g, "/"));
            }
          }
        }
        if (implFiles.length > 0) {
          payload.task_prompt = `${payload.task_prompt}\n\n[Impl Files to Verify]\n${implFiles.map((f) => `- ${f}`).join("\n")}\nCheck each file exists and assess whether its content addresses the acceptance criteria.`;
        }
        const suiteId = registry.project_types?.[run.project_type]?.acceptance_suite;
        const suite = suiteId ? registry.acceptance_suites?.[suiteId] : null;
        const suiteCommands = Array.isArray(suite?.commands) ? suite.commands.filter(Boolean) : [];
        if (suiteCommands.length > 0) {
          payload.task_prompt = `${payload.task_prompt}\n\n[Acceptance Suite Commands]\nThe following commands are expected to pass for this project type:\n${suiteCommands.map((c) => `- ${c}`).join("\n")}\nInclude a deterministic check entry for each command result.`;
        }
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

    if (String(stepDef.tool || "") === "coding.execute" && String(stepDef.id || "") === "smoke_test") {
      const smokeCmd = buildSmokeTestCommand({ workspaceRoot, artifactRoot, port: 13099 });
      const acceptCmd = buildAcceptanceTestCommand({ workspaceRoot, artifactRoot });
      payload.command = `${smokeCmd} && ${acceptCmd}`;
      payload.max_runtime_s = clampInt(payload.max_runtime_s ?? 120, 30, 600, 120);
      // smoke_test 步骤现在同时产出 smoke + acceptance 结果
      if (Array.isArray(payload.expected_artifacts) && !payload.expected_artifacts.includes("verify/acceptance_result.json")) {
        payload.expected_artifacts.push("verify/acceptance_result.json");
      }
      // v3.3 Patch B: refinement re-entry — load regression baseline from parent run
      if (inputLineage?.parent_run_id) {
        const parentArtifactRoot = pathForRunArtifacts(inputLineage.parent_run_id);
        const baselinePath = path.resolve(workspaceRoot, parentArtifactRoot, "release", "regression_baseline.json");
        if (fs.existsSync(baselinePath)) {
          payload.regression_baseline_path = baselinePath.replace(/\\/g, "/");
        }
      }
    }

    if (String(stepDef.tool || "") === "ops.deploy_preview") {
      const defaultTargetPaths = Array.isArray(payload.target_paths) && payload.target_paths.length > 0
        ? payload.target_paths
        : defaultCodingTargetPaths(payload.project_type);
      payload.target_paths = defaultTargetPaths;
      payload.project_root_candidates = [
        `${artifactRoot}/impl/be_changes`,
        `${artifactRoot}/impl/fe_changes/public`,
        ...defaultTargetPaths,
      ];
      payload.release_manifest_path = `${artifactRoot}/meta/run_manifest.json`;
      payload.release_notes_path = `${artifactRoot}/release/release_notes.md`;
      payload.preview_ttl_hours = clampInt(
        payload.preview_ttl_hours ?? input.preview_ttl_hours ?? 24,
        1,
        168,
        24,
      );
      payload.preview_provider = String(payload.preview_provider || input.preview_provider || "local");
      payload.render_service_id = String(payload.render_service_id || input.render_service_id || "");
      // GitHub delivery params (pass-through from input or environment)
      payload.github_repo = String(payload.github_repo || input.github_repo || process.env.GITHUB_REPO || "");
      payload.github_token = String(payload.github_token || input.github_token || process.env.GITHUB_TOKEN || "");
      payload.branch_prefix = String(payload.branch_prefix || input.branch_prefix || "nexus/");
      payload.goal = String(input.goal || "");
    }

    return payload;
  }

  return { buildStepPayload };
}


