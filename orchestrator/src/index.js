import express from "express";
import { v4 as uuidv4 } from "uuid";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { parseIntent, translate, CURRENT_QWEN_MODEL, setQwenModel } from "./nlp/router.js";
import { createDiscordGateway } from "./adapters/discord_gateway.js";
import { analyzeTaskRisk } from "./policy.js";
import { handleExecuteTool } from "./ingress.js";
import { getDefaultRegistryPath, loadRegistryOrThrow, validateTaskInputAgainstRegistry } from "./registry.js";
import { createWorkflowEngine } from "./workflow_engine.js";
import { getDefaultPromptScriptRegistryPath, loadPromptScriptRegistryOrThrow, validatePromptScriptsAgainstAgents } from "./prompt_script_registry.js";
import { getDefaultAgentRegistryDir, loadAgentContractsOrThrow } from "./agent_contract_registry.js";
import { getDefaultHandoffContractPath, loadHandoffContractsOrThrow } from "./handoff_contract_registry.js";
import { buildRouteContractResponse } from "./vnext/route_contract.js";
import { makeErrorResponse } from "./vnext/response_protocol.js";
import { createExecuteVNextDispatch } from "./vnext/runtime_dispatch.js";
import { createHandleApiChat, generateBrainDirectReply as generateChatDirectReply } from "./vnext/chat_entrypoint.js";
import { createHandleApproveTask, createHandleRejectTask } from "./vnext/approval_entrypoint.js";
import { deliverWorkflowRuntimeNotification } from "./vnext/workflow_notification_delivery.js";
import { assertDispatchErrorResponse } from "./vnext/contract_validator.js";
import { callBrainWithRetry } from "./vnext/local_llm_client.js";
import { dispatch as dispatchLlm } from "./vnext/llm_dispatcher.js";
import { planCompositeWorkflowFromText, buildForcedIntentFromRule, detectLanguageQuick, formatCodingDelegateResult, summarizeOutputBrief } from "./vnext/composite_planner.js";
import { createDiscordMessageHandler } from "./adapters/discord_message_handler.js";
import { createResultConsumer } from "./vnext/result_consumer.js";
import { createTaskWatchdog } from "./vnext/task_watchdog.js";
import { registerCronSchedules } from "./vnext/cron_scheduler.js";
import { createTaskEnqueuer } from "./vnext/task_enqueuer.js";
import { createRuntimeConnections } from "./infra/runtime_connections.js";
import {
  upsertTask as _upsertTask, countPendingTasksForRun as _countPendingTasksForRun,
  countFailedTasksForRun as _countFailedTasksForRun, findRunIdByTaskId as _findRunIdByTaskId,
  findTaskIdByIdempotencyKey as _findTaskIdByIdempotencyKey, getTaskForApproval as _getTaskForApproval,
  getTaskForRejection as _getTaskForRejection, getTaskStream as _getTaskStream,
  listStaleQueuedTasks as _listStaleQueuedTasks, listStaleRunningTasks as _listStaleRunningTasks,
  listTasksForRunStatus as _listTasksForRunStatus, listTasksForRunTimeline as _listTasksForRunTimeline,
  listPendingApprovalTasks as _listPendingApprovalTasks, markTaskRunning as _markTaskRunning,
  markTaskApprovalRejected as _markTaskApprovalRejected, markTaskQueued as _markTaskQueued,
  recordTaskEvent as _recordTaskEvent, updateTaskTerminalResult as _updateTaskTerminalResult,
  forceTaskFailed as _forceTaskFailed, failQueuedTaskIfStillQueued as _failQueuedTaskIfStillQueued,
  failRunningTaskIfStillRunning as _failRunningTaskIfStillRunning,
} from "./data/task_repository.js";
import { listEventsForTaskIds as _listEventsForTaskIds } from "./data/event_repository.js";
import { ensureOrchestratorSchema as _ensureOrchestratorSchema } from "./data/schema_repository.js";
import { listRecentMemoryItemsForProject as _listRecentMemoryItemsForProject, insertMemoryItem as _insertMemoryItem } from "./data/memory_store_repository.js";
import { completeRunWithCostLedger as _completeRunWithCostLedger, ensureRun as _ensureRun, findRunIdByClientMsgId as _findRunIdByClientMsgId, getRunById as _getRunById, getRunInputText as _getRunInputText, updateRunStatus as _updateRunStatus, updateRunStatusIfNotFailed as _updateRunStatusIfNotFailed } from "./data/run_repository.js";
import { insertRule as _insertRule, listLatestRulesForProject as _listLatestRulesForProject } from "./data/rule_repository.js";
import { enqueueToStream as _enqueueToStream } from "./data/stream_repository.js";
import { getTraceProjectAndAction as _getTraceProjectAndAction, getTraceProjectId as _getTraceProjectId, insertTrace as _insertTrace, updateTraceFeedback as _updateTraceFeedback } from "./data/trace_repository.js";
import { countFailedWorkflowRunsForRun as _countFailedWorkflowRunsForRun, insertWorkflowDefinition as _insertWorkflowDefinition } from "./data/workflow_repository.js";

// --- Env vars ---
const {
  REDIS_URL, PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE,
  STREAM_TASK = "stream:task", STREAM_TASK_CODING = "stream:task:coding",
  STREAM_RESULT = "stream:result", GROUP_TASK = "cg:workers", GROUP_RESULT = "cg:orchestrator",
  DISCORD_TOKEN, MINIO_ENDPOINT = "http://nexus-minio:9000",
  MINIO_ACCESS_KEY = "nexus", MINIO_SECRET_KEY = "nexuspassword",
  AUTO_REPORT_CHANNEL_ID, AUTO_REPORT_TIMEZONE = "Asia/Shanghai",
  APPROVAL_TOKEN = "dev-approval-token", TOOLS_CONFIG_PATH = "configs/tools.json",
  REGISTRY_PATH = "", RESUME_TOKEN_SECRET = "dev-resume-secret",
  RESUME_TOKEN_TTL_SEC = "86400", WORKSPACE_ROOT = "/workspace",
  STREAM_TASK_DLQ = "stream:task:dlq", TASK_RUNNING_TIMEOUT_SEC = "900",
  TASK_QUEUED_TIMEOUT_SEC = "21600", TASK_WATCHDOG_INTERVAL_SEC = "30",
  TASK_TIMEOUT_AUTO_DLQ = "1", RUNTIME_CONFIG_PATH = "configs/runtime/runtime_defaults.json",
  RELEASE_PACK_ARCHIVE_TO_MINIO = "1", RELEASE_PACK_BUCKET = "nexus-artifacts",
  WORKFLOW_STEP_ARTIFACT_AUDIT = "", WORKFLOW_STRICT_STEP_ARTIFACTS = "",
} = process.env;

// --- Runtime config ---
function loadRuntimeConfig() {
  const candidates = [String(RUNTIME_CONFIG_PATH || "").trim(), path.resolve("configs/runtime/runtime_defaults.json"), path.resolve("../configs/runtime/runtime_defaults.json")].filter(Boolean);
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) {
        const parsed = JSON.parse(fs.readFileSync(p, "utf-8"));
        if (parsed && typeof parsed === "object") return { config: parsed, path: p };
      }
    } catch (err) { console.warn(`[runtime-config] failed to load '${p}': ${err.message}`); }
  }
  return { config: {}, path: null };
}
const RUNTIME_CONFIG_LOADED = loadRuntimeConfig();
const RUNTIME_CONFIG = RUNTIME_CONFIG_LOADED.config || {};
const RUNTIME_ORCH = RUNTIME_CONFIG.orchestrator || {};
const RUNTIME_EXECUTION = RUNTIME_CONFIG.execution || {};
const RUNTIME_WATCHDOG = RUNTIME_CONFIG.watchdog || {};
const RUNTIME_STREAMS = RUNTIME_CONFIG.streams || {};
const QWEN_BASE = process.env.QWEN_BASE_URL || RUNTIME_ORCH.qwen_base_url || "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";
const QWEN_MODEL = process.env.QWEN_MODEL || RUNTIME_ORCH.qwen_model || "qwen-plus";
const CODER_PROVIDER_DEFAULT = String(process.env.CODER_PROVIDER_DEFAULT || RUNTIME_ORCH.coder_provider_default || "opencode").toLowerCase();
const CODER_MODEL_DEFAULT = String(process.env.CODER_MODEL_DEFAULT || RUNTIME_ORCH.coder_model_default || "minimax-m2.5");
const DEFAULT_LOCAL_MODEL = process.env.QUANT_LLM_MODEL || RUNTIME_ORCH.quant_llm_model || "deepseek-r1:32b";
const RESOLVED_WORKFLOW_STEP_ARTIFACT_AUDIT = String(WORKFLOW_STEP_ARTIFACT_AUDIT || (RUNTIME_ORCH.workflow_step_artifact_audit ? "1" : "0")) !== "0";
const RESOLVED_WORKFLOW_STRICT_STEP_ARTIFACTS = String(WORKFLOW_STRICT_STEP_ARTIFACTS || (RUNTIME_ORCH.workflow_strict_step_artifacts ? "1" : "0")) !== "0";
const RESOLVED_STREAM_TASK_DLQ = String(STREAM_TASK_DLQ || RUNTIME_STREAMS.task_dlq || "stream:task:dlq");
const RESOLVED_TASK_RUNNING_TIMEOUT_SEC = Number(TASK_RUNNING_TIMEOUT_SEC || RUNTIME_WATCHDOG.running_timeout_sec || 900);
const RESOLVED_TASK_QUEUED_TIMEOUT_SEC = Number(TASK_QUEUED_TIMEOUT_SEC || RUNTIME_WATCHDOG.queued_timeout_sec || 21600);
const RESOLVED_TASK_WATCHDOG_INTERVAL_SEC = Number(TASK_WATCHDOG_INTERVAL_SEC || RUNTIME_WATCHDOG.interval_sec || 30);
const RESOLVED_TASK_TIMEOUT_AUTO_DLQ = String(TASK_TIMEOUT_AUTO_DLQ || (RUNTIME_WATCHDOG.auto_dlq ? "1" : "0")) !== "0";

// Mutable app state (model/mode selection via Discord commands)
const appState = { forceLocalLlm: false, currentLocalModel: DEFAULT_LOCAL_MODEL };

// --- Registry/tool config ---
function loadToolsConfig() {
  try {
    const raw = fs.readFileSync(path.resolve(TOOLS_CONFIG_PATH), "utf-8");
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed ? parsed : {};
  } catch (err) { console.warn("[orchestrator] tools.json load failed:", err.message); return {}; }
}
const TOOLS_CONFIG = loadToolsConfig();
export function getToolSpec(toolName) { return TOOLS_CONFIG?.[toolName] || {}; }

function loadRegistryWithFallback() {
  const explicit = REGISTRY_PATH && String(REGISTRY_PATH).trim() ? String(REGISTRY_PATH).trim() : "";
  if (explicit) {
    try { return { path: explicit, registry: loadRegistryOrThrow(explicit) }; } catch (err) {
      const fallbackPath = getDefaultRegistryPath();
      if (path.resolve(explicit) === path.resolve(fallbackPath)) throw err;
      console.warn(`[registry] explicit REGISTRY_PATH failed ('${explicit}'): ${err.message}`);
      console.warn(`[registry] falling back to default registry path '${fallbackPath}'`);
      return { path: fallbackPath, registry: loadRegistryOrThrow(fallbackPath) };
    }
  }
  const fallbackPath = getDefaultRegistryPath();
  return { path: fallbackPath, registry: loadRegistryOrThrow(fallbackPath) };
}
const REGISTRY_LOADED = loadRegistryWithFallback();
const REGISTRY_CONFIG_PATH = REGISTRY_LOADED.path;
const REGISTRY = REGISTRY_LOADED.registry;
const PROMPT_SCRIPT_REGISTRY = loadPromptScriptRegistryOrThrow(getDefaultPromptScriptRegistryPath());
const AGENT_REGISTRY = loadAgentContractsOrThrow(getDefaultAgentRegistryDir());
const HANDOFF_CONTRACTS = loadHandoffContractsOrThrow(getDefaultHandoffContractPath());
const PROMPT_AGENT_BINDING_CHECK = validatePromptScriptsAgainstAgents({ promptRegistry: PROMPT_SCRIPT_REGISTRY, agentRegistry: AGENT_REGISTRY });
if (!PROMPT_AGENT_BINDING_CHECK.ok) {
  throw new Error(`prompt script/agent binding invalid: ${PROMPT_AGENT_BINDING_CHECK.errors.join("; ")}`);
}
const channelMemory = new Map();

// --- Infra ---
export const { redis, pool, s3 } = createRuntimeConnections({
  redisUrl: REDIS_URL,
  pgHost: PGHOST,
  pgPort: PGPORT,
  pgUser: PGUSER,
  pgPassword: PGPASSWORD,
  pgDatabase: PGDATABASE,
  minioEndpoint: MINIO_ENDPOINT,
  minioAccessKey: MINIO_ACCESS_KEY,
  minioSecretKey: MINIO_SECRET_KEY,
});

const discordGateway = createDiscordGateway({ translate });
const { client: discord, taskToContext, runToContext, workflowRunToContext, replyChunked, safeTranslate, bindTaskToContext, sendStepTransitionNotification, createResultEmbed, createBinaryAttachment, registerHandlers: registerDiscordHandlers, login: loginDiscordGateway } = discordGateway;

// --- Utilities ---
function makeIdempotencyKey(run_id, tool_name, payload = {}) {
  return crypto.createHash("sha256").update(`${run_id}|${tool_name}|${JSON.stringify(payload)}`).digest("hex").slice(0, 48);
}
export async function recordEvent(task_id, event_type, payload = {}) { return _recordTaskEvent(pool, task_id, event_type, payload); }

function normalizeErrorCode(status, errorCode, output) {
  const raw = String(errorCode || "").trim();
  if (raw) return raw;
  if (status === "succeeded") return null;
  const fromOutput = String(output?.error_code || output?.code || "").trim();
  if (fromOutput) return fromOutput;
  if (status === "failed") return "TASK_FAILED";
  if (status === "aborted") return "TASK_ABORTED";
  return "TASK_ERROR";
}

function normalizeResultPayload(status, output, errorCode) {
  const safe = output && typeof output === "object" ? output : { raw: String(output || "") };
  return { ok: status === "succeeded", status, error_code: errorCode || null, output: safe, updated_at: new Date().toISOString() };
}

function listFilesRecursive(rootDir, maxFiles = 400) {
  if (!rootDir || !fs.existsSync(rootDir)) return [];
  const out = [];
  const stack = [rootDir];
  while (stack.length > 0 && out.length < maxFiles) {
    const cur = stack.pop();
    let ents = [];
    try { ents = fs.readdirSync(cur, { withFileTypes: true }); } catch { continue; }
    for (const ent of ents) {
      const full = path.join(cur, ent.name);
      if (ent.isDirectory()) { stack.push(full); }
      else if (ent.isFile()) {
        try { const st = fs.statSync(full); out.push({ path: full.replace(/\\/g, "/"), bytes: st.size, mtime: st.mtime.toISOString() }); } catch {}
      }
      if (out.length >= maxFiles) break;
    }
  }
  return out.sort((a, b) => String(a.path).localeCompare(String(b.path)));
}

// --- DB wrappers ---
async function ensureRun(run_id, opts) { return _ensureRun(pool, run_id, opts); }
async function updateRunStatus(run_id, status) { return _updateRunStatus(pool, run_id, status); }
async function updateRunStatusIfNotFailed(run_id, status) { return _updateRunStatusIfNotFailed(pool, run_id, status); }
async function completeRunWithCostLedger(run_id, costLedger = {}) { return _completeRunWithCostLedger(pool, run_id, costLedger); }
async function findRunIdByClientMsgId(client_msg_id) { return _findRunIdByClientMsgId(pool, client_msg_id); }
async function getRunById(run_id) { return _getRunById(pool, run_id); }
async function getRunInputText(run_id) { return _getRunInputText(pool, run_id); }
export async function upsertTask(task) { return _upsertTask(pool, task); }
export function getTaskStream(tool_name) { return _getTaskStream(pool, tool_name, { streamTask: STREAM_TASK, streamTaskCoding: STREAM_TASK_CODING }); }

// --- LLM context helpers ---
function detectProject(text) {
  const lower = String(text || "").toLowerCase();
  if (lower.includes("openclaw") || lower.includes("nexus")) return "openclaw";
  if (lower.includes("quant") || lower.includes("\u4ea4\u6613") || lower.includes("\u9009\u80a1")) return "quant";
  return "general";
}

async function buildContext(project) {
  try {
    let contextStr = "";
    const ruleRows = await _listLatestRulesForProject(pool, project, 5);
    if (ruleRows.length > 0) {
      contextStr += "- Soft Rules / Guidelines:\n";
      ruleRows.forEach((r, idx) => { try { const o = JSON.parse(r.rule_json); if (o.message) contextStr += `  ${idx + 1}. ${o.message}\n`; } catch {} });
    }
    const memRows = await _listRecentMemoryItemsForProject(pool, project, 3);
    if (memRows.length > 0) {
      contextStr += "\n- Approved SOPs / Memories:\n";
      memRows.forEach(m => { try { contextStr += `  * ${JSON.stringify(JSON.parse(m.content))}\n`; } catch { contextStr += `  * ${m.content}\n`; } });
    }
    return contextStr.trim();
  } catch (err) { console.warn("[learning] Failed to build context:", err.message); return ""; }
}

async function generateBrainDirectReply(rawInput, modelPreference = "auto") {
  const project = detectProject(rawInput);
  const projectContext = await buildContext(project);
  return generateChatDirectReply({
    rawInput,
    modelPreference,
    forceLocalLlm: appState.forceLocalLlm,
    hasQwenKey: Boolean(process.env.QWEN_API_KEY),
    qwenBase: QWEN_BASE,
    qwenModel: QWEN_MODEL,
    currentLocalModel: appState.currentLocalModel,
    projectContext,
  });
}

// --- Task enqueuer ---
const taskEnqueuer = createTaskEnqueuer({
  pool, redis, registry: REGISTRY, analyzeTaskRisk, validateTaskInputAgainstRegistry, getToolSpec,
  upsertTask, getTaskStream, findTaskIdByIdempotencyKey: _findTaskIdByIdempotencyKey,
  enqueueToStream: _enqueueToStream, bindTaskToContext, recordEvent,
  insertWorkflowDefinition: _insertWorkflowDefinition, makeIdempotencyKey,
  groupTask: GROUP_TASK,
});
const { enqueueTask, enqueueWorkflow } = taskEnqueuer;

// --- Workflow engine ---
const workflowEngine = createWorkflowEngine({
  pool, registry: REGISTRY, promptScriptRegistry: PROMPT_SCRIPT_REGISTRY,
  handoffContracts: HANDOFF_CONTRACTS, enqueueTask, recordEvent, makeIdempotencyKey,
  resumeTokenSecret: String(RESUME_TOKEN_SECRET || "dev-resume-secret"),
  resumeTokenTtlSec: Number(RESUME_TOKEN_TTL_SEC || 86400),
  workspaceRoot: String(WORKSPACE_ROOT || "/workspace"),
  auditStepArtifacts: RESOLVED_WORKFLOW_STEP_ARTIFACT_AUDIT,
  strictStepArtifacts: RESOLVED_WORKFLOW_STRICT_STEP_ARTIFACTS,
  runtimeConfig: {
    ...RUNTIME_CONFIG,
    execution: {
      diff_first_enabled: RUNTIME_EXECUTION.diff_first_enabled !== false,
    },
  },
  minio: {
    enabled: String(RELEASE_PACK_ARCHIVE_TO_MINIO || "1") !== "0",
    bucket: String(RELEASE_PACK_BUCKET || "nexus-artifacts"),
    endpoint: String(MINIO_ENDPOINT || "http://nexus-minio:9000"),
    accessKey: String(MINIO_ACCESS_KEY || "nexus"),
    secretKey: String(MINIO_SECRET_KEY || "nexuspassword"),
  },
  onStepTransition: sendStepTransitionNotification,
});

// --- vNext services ---
const executeVNextDispatch = createExecuteVNextDispatch({
  ensureRun, parseIntent, registry: REGISTRY, generateBrainDirectReply, pool,
  updateRunStatus: async (_pool, run_id, status) => updateRunStatus(run_id, status),
  enqueueTask, workflowEngine, coderProviderDefault: CODER_PROVIDER_DEFAULT, coderModelDefault: CODER_MODEL_DEFAULT,
});

const handleApiChat = createHandleApiChat({
  uuidv4, ensureRun, planCompositeWorkflowFromText, pool,
  updateRunStatus: async (_pool, run_id, status) => updateRunStatus(run_id, status),
  completeRunWithCostLedger: async (_pool, run_id, costLedger) => completeRunWithCostLedger(run_id, costLedger),
  enqueueWorkflow, parseIntent, buildForcedIntentFromRule, executeVNextDispatch,
  buildVNextDispatchInput: (args) => ({ source: args.source || "api", raw_input: args.rawInput || "", ...args.payload }),
  forceLocalLlm: appState.forceLocalLlm,
  callBrainWithRetry,
  currentLocalModel: appState.currentLocalModel,
  currentQwenModel: CURRENT_QWEN_MODEL,
});

const handleApproveTask = createHandleApproveTask({
  approvalToken: APPROVAL_TOKEN, pool,
  getTaskForApproval: async (_pool, task_id) => _getTaskForApproval(pool, task_id),
  markTaskQueued: async (_pool, task_id) => _markTaskQueued(pool, task_id),
  recordEvent, workflowEngine, getTaskStream, redis,
});

const handleRejectTask = createHandleRejectTask({
  approvalToken: APPROVAL_TOKEN, pool,
  updateRunStatus: async (_pool, run_id, status) => updateRunStatus(run_id, status),
  getTaskForRejection: async (_pool, task_id) => _getTaskForRejection(pool, task_id),
  markTaskApprovalRejected: async (_pool, task_id, resultPayload) => _markTaskApprovalRejected(pool, task_id, resultPayload),
  countPendingTasksForRun: async (_pool, run_id) => _countPendingTasksForRun(pool, run_id),
  recordEvent, workflowEngine, normalizeResultPayload, taskToContext, runToContext,
});

// --- Discord handlers ---
const { handleDiscordMessage, handleDiscordReaction } = createDiscordMessageHandler({
  redis, discord, approvalToken: APPROVAL_TOKEN,
  coderProviderDefault: CODER_PROVIDER_DEFAULT, coderModelDefault: CODER_MODEL_DEFAULT,
  ensureRun, updateRunStatus, findRunIdByClientMsgId, enqueueTask, makeIdempotencyKey,
  getToolSpec, executeVNextDispatch, appState,
  currentQwenModel: () => CURRENT_QWEN_MODEL, setQwenModel,
  translate, safeTranslate, replyChunked, runToContext, workflowRunToContext,
});
registerDiscordHandlers({ onMessage: handleDiscordMessage, onReaction: handleDiscordReaction });
loginDiscordGateway(DISCORD_TOKEN);

// --- Cron schedules ---
registerCronSchedules({ ensureRun, enqueueTask, runToContext, autoReportChannelId: AUTO_REPORT_CHANNEL_ID, autoReportTimezone: AUTO_REPORT_TIMEZONE });

// --- Result consumer ---
const resultConsumer = createResultConsumer({
  pool, redis, workflowEngine, normalizeResultPayload, normalizeErrorCode,
  getRunInputText, findRunIdByTaskId: (pool, id) => _findRunIdByTaskId(pool, id),
  countPendingTasksForRun: (pool, id) => _countPendingTasksForRun(pool, id),
  countFailedWorkflowRunsForRun: (pool, id) => _countFailedWorkflowRunsForRun(pool, id),
  countFailedTasksForRun: (pool, id) => _countFailedTasksForRun(pool, id),
  updateRunStatusIfNotFailed, updateTaskTerminalResult: _updateTaskTerminalResult,
  forceTaskFailed: _forceTaskFailed, markTaskRunning: _markTaskRunning, recordEvent,
  taskToContext, runToContext, discord, replyChunked, safeTranslate,
  createResultEmbed, createBinaryAttachment,
  insertTrace: _insertTrace, s3,
  callLocalOllamaReply: async (prompt) => {
    const result = await dispatchLlm(
      "qa",
      [{ role: "user", content: prompt }],
      { provider: "local_ollama", model: appState.currentLocalModel, secondary_model: "" }
    );
    return result.content;
  },
  detectProject, formatCodingDelegateResult, summarizeOutputBrief,
  deliverWorkflowRuntimeNotification, streamResult: STREAM_RESULT, groupResult: GROUP_RESULT,
});

// --- Task watchdog ---
const taskWatchdog = createTaskWatchdog({
  pool, redis, workflowEngine, normalizeResultPayload, recordEvent,
  listStaleRunningTasks: _listStaleRunningTasks, listStaleQueuedTasks: _listStaleQueuedTasks,
  failRunningTaskIfStillRunning: _failRunningTaskIfStillRunning,
  failQueuedTaskIfStillQueued: _failQueuedTaskIfStillQueued,
  enqueueToStream: _enqueueToStream, groupTask: GROUP_TASK,
  runningTimeoutSec: Math.max(60, RESOLVED_TASK_RUNNING_TIMEOUT_SEC),
  queuedTimeoutSec: Math.max(300, RESOLVED_TASK_QUEUED_TIMEOUT_SEC),
  intervalMs: Math.max(5000, RESOLVED_TASK_WATCHDOG_INTERVAL_SEC * 1000),
  autoDlq: RESOLVED_TASK_TIMEOUT_AUTO_DLQ, streamTaskDlq: RESOLVED_STREAM_TASK_DLQ,
});

// --- Express app ---
const app = express();
app.use(express.json());
app.get("/health", (_, res) => res.send("ok"));

app.post("/vnext/route", async (req, res) => {
  try {
    let analyzerResult = null;
    try {
      const raw = String(req.body?.raw_input || req.body?.message || req.body?.text || "");
      analyzerResult = raw ? await parseIntent(raw, {}) : null;
    } catch (err) { console.warn("[vnext] parseIntent failed, fallback to heuristic router:", err?.message || err); }
    const result = buildRouteContractResponse({ body: req.body || {}, analyzerResult, registry: REGISTRY });
    return res.json(result);
  } catch (err) {
    const code = String(err?.code || "");
    const badReq = code === "TASK_ENVELOPE_INVALID" || code === "BAD_REQUEST";
    return res.status(badReq ? 400 : 500).json(assertDispatchErrorResponse(makeErrorResponse({ error: err.message || "vnext route failed", error_code: code || "UNKNOWN_ERROR", task_envelope: null })));
  }
});

app.post("/vnext/dispatch", async (req, res) => {
  const run_id = String(req.body?.run_id || uuidv4()).trim();
  try {
    const result = await executeVNextDispatch({ requestBody: req.body || {}, run_id, client_msg_id: `vnext-${run_id}` });
    if (!result.ok) return res.status(400).json(result);
    return res.json(result);
  } catch (err) {
    await updateRunStatus(run_id, "failed").catch(() => {});
    const code = String(err?.code || "");
    const badReq = ["TASK_ENVELOPE_INVALID", "REGISTRY_INVALID", "BAD_REQUEST", "DISPATCH_SUCCESS_CONTRACT_INVALID", "DISPATCH_ERROR_CONTRACT_INVALID"].includes(code);
    return res.status(badReq ? 400 : 500).json(assertDispatchErrorResponse(makeErrorResponse({
      run_id, error: err.message || "vnext dispatch failed",
      error_code: ["TASK_ENVELOPE_INVALID", "REGISTRY_INVALID", "BAD_REQUEST"].includes(code) ? code : "UNKNOWN_ERROR",
      task_envelope: null,
    })));
  }
});

app.get("/runtime/config", (_, res) => res.json({
  ok: true,
  runtime_config_path: RUNTIME_CONFIG_LOADED.path || null,
  resolved: {
    qwen_base_url: QWEN_BASE, qwen_model: QWEN_MODEL, coder_provider_default: CODER_PROVIDER_DEFAULT,
    coder_model_default: CODER_MODEL_DEFAULT, quant_llm_model: DEFAULT_LOCAL_MODEL,
    workflow_step_artifact_audit: RESOLVED_WORKFLOW_STEP_ARTIFACT_AUDIT, workflow_strict_step_artifacts: RESOLVED_WORKFLOW_STRICT_STEP_ARTIFACTS,
    stream_task_dlq: RESOLVED_STREAM_TASK_DLQ, task_running_timeout_sec: RESOLVED_TASK_RUNNING_TIMEOUT_SEC,
    task_queued_timeout_sec: RESOLVED_TASK_QUEUED_TIMEOUT_SEC, task_watchdog_interval_sec: RESOLVED_TASK_WATCHDOG_INTERVAL_SEC,
    task_timeout_auto_dlq: RESOLVED_TASK_TIMEOUT_AUTO_DLQ,
  },
  source_priority: ["environment variables", "runtime_defaults.json", "hardcoded fallback"],
}));

app.post("/tasks/:task_id/approve", handleApproveTask);
app.post("/tasks/:task_id/reject", handleRejectTask);

app.post("/workflow-runs/start", async (req, res) => {
  const workflow_id = String(req.body?.workflow_id || "").trim();
  const project_type = String(req.body?.project_type || "").trim();
  const input = req.body?.input && typeof req.body.input === "object" ? req.body.input : {};
  const run_id = String(req.body?.run_id || uuidv4()).trim();
  if (!workflow_id) return res.status(400).json({ ok: false, error: "workflow_id is required" });
  try {
    await ensureRun(run_id, { client_msg_id: `workflow-run-${run_id}`, user_id: "workflow", status: "running", input_text: `workflow_run:${workflow_id}` });
    const started = await workflowEngine.startWorkflowRun({ workflow_id, project_type: project_type || undefined, run_id, input, context: null });
    return res.json({ ok: true, ...started });
  } catch (err) {
    const code = String(err?.code || "");
    const badReq = ["WORKFLOW_NOT_FOUND", "PROJECT_TYPE_NOT_FOUND", "WORKFLOW_PROJECT_TYPE_MISMATCH", "WORKFLOW_EMPTY"].includes(code);
    return res.status(badReq ? 400 : 500).json({ ok: false, error: err.message || "workflow run start failed", error_code: code || undefined });
  }
});

app.get("/workflow-runs/:workflow_run_id", async (req, res) => {
  try {
    const state = await workflowEngine.getWorkflowRunStatus(req.params.workflow_run_id);
    if (!state) return res.status(404).json({ ok: false, error: "workflow_run not found" });
    return res.json({ ok: true, ...state });
  } catch (err) { return res.status(500).json({ ok: false, error: err.message || "workflow run query failed" }); }
});

app.post("/workflow-runs/:workflow_run_id/resume-token", async (req, res) => {
  try {
    const issued = await workflowEngine.issueResumeToken(req.params.workflow_run_id);
    return res.json({ ok: true, workflow_run_id: req.params.workflow_run_id, ...issued });
  } catch (err) {
    const code = String(err?.code || "");
    return res.status(code === "WORKFLOW_RUN_NOT_FOUND" || code === "RESUME_INVALID" ? 400 : 500).json({ ok: false, error: err.message || "issue resume token failed", error_code: code || undefined });
  }
});

app.post("/workflow-runs/:workflow_run_id/resume", async (req, res) => {
  const resume_token = String(req.body?.resume_token || "").trim();
  if (!resume_token) return res.status(400).json({ ok: false, error: "resume_token is required", error_code: "RESUME_INVALID" });
  try {
    return res.json(await workflowEngine.resumeFromToken(req.params.workflow_run_id, resume_token, null));
  } catch (err) {
    const code = String(err?.code || "");
    return res.status(code === "WORKFLOW_RUN_NOT_FOUND" || code === "RESUME_INVALID" ? 400 : 500).json({ ok: false, error: err.message || "resume failed", error_code: code || undefined });
  }
});

app.get("/workflow-runs/:workflow_run_id/validate-pack", async (req, res) => {
  try {
    const result = await workflowEngine.validateRunArtifactPack(req.params.workflow_run_id);
    return res.json({ ok: true, validation: result });
  } catch (err) {
    const code = String(err?.code || "");
    return res.status(code === "WORKFLOW_RUN_NOT_FOUND" ? 404 : 500).json({ ok: false, error: err.message || "artifact pack validate failed", error_code: code || undefined });
  }
});

app.post("/workflow-runs/:workflow_run_id/archive-pack", async (req, res) => {
  try {
    return res.json(await workflowEngine.archiveRunArtifactPack(req.params.workflow_run_id));
  } catch (err) {
    const code = String(err?.code || "");
    return res.status(code === "WORKFLOW_RUN_NOT_FOUND" || code === "ARTIFACT_INCOMPLETE" ? 400 : 500).json({ ok: false, error: err.message || "archive pack failed", error_code: code || undefined });
  }
});

app.get("/runs/:run_id/status", async (req, res) => {
  try {
    const run = await getRunById(req.params.run_id);
    if (!run) return res.status(404).json({ ok: false, error: "run not found" });
    const tasks = await _listTasksForRunStatus(pool, req.params.run_id);
    const counts = { queued: 0, running: 0, waiting_approval: 0, succeeded: 0, failed: 0, other: 0 };
    for (const t of tasks) { const s = String(t.status || ""); if (counts[s] !== undefined) counts[s] += 1; else counts.other += 1; }
    return res.json({ ok: true, run: { run_id: run.run_id, status: run.status, created_at: run.created_at, input_text: run.input_text }, counts, tasks });
  } catch (err) { return res.status(500).json({ ok: false, error: err.message || "run status query failed" }); }
});

app.get("/runs/:run_id/timeline", async (req, res) => {
  try {
    const tasks = await _listTasksForRunTimeline(pool, req.params.run_id);
    if (tasks.length === 0) return res.status(404).json({ ok: false, error: "run not found" });
    const events = await _listEventsForTaskIds(pool, tasks.map(r => r.task_id));
    return res.json({ ok: true, run_id: req.params.run_id, tasks, events });
  } catch (err) { return res.status(500).json({ ok: false, error: err.message || "timeline query failed" }); }
});

app.get("/runs/:run_id/artifacts", async (req, res) => {
  try {
    const run_id = req.params.run_id;
    const releaseDir = path.join(WORKSPACE_ROOT, "artifacts", "release", run_id);
    const runtimeDir = path.join(WORKSPACE_ROOT, "artifacts", "runs", run_id);
    return res.json({ ok: true, run_id, roots: { release: releaseDir.replace(/\\/g, "/"), runtime: runtimeDir.replace(/\\/g, "/") }, release_files: listFilesRecursive(releaseDir), runtime_files: listFilesRecursive(runtimeDir) });
  } catch (err) { return res.status(500).json({ ok: false, error: err.message || "artifacts query failed" }); }
});

app.get("/approvals/pending", async (req, res) => {
  try {
    const limit = Math.max(1, Math.min(Number(req.query.limit || 50), 200));
    const tasks = await _listPendingApprovalTasks(pool, limit);
    return res.json({ ok: true, count: tasks.length, tasks });
  } catch (err) { return res.status(500).json({ ok: false, error: err.message || "pending approval query failed" }); }
});

app.post("/chat", handleApiChat);

app.post("/traces", async (req, res) => {
  const { project_id, task_type, context_digest, action_json, metrics_json } = req.body;
  const trace_id = uuidv4();
  try {
    await _insertTrace(pool, { trace_id, project_id: project_id || "general", task_type: task_type || "unknown", context_digest: context_digest || "", action_json: action_json || {}, metrics_json: metrics_json || {} });
    return res.json({ ok: true, trace_id });
  } catch (err) { return res.status(500).json({ ok: false, error: err.message }); }
});

app.post("/traces/:trace_id/feedback", async (req, res) => {
  const { trace_id } = req.params;
  const { feedback, reason, rating } = req.body;
  try {
    await _updateTraceFeedback(pool, trace_id, { feedback, reason, rating });
    if (feedback === "\u274c" && reason) {
      const project_id = (await _getTraceProjectId(pool, trace_id)) || "general";
      await _insertRule(pool, { rule_id: uuidv4(), project_id, scope: "task", rule_type: "soft", rule_json: { condition: "feedback_based", message: reason }, weight: 1 });
    }
    if (feedback === "\u2705") {
      const row = await _getTraceProjectAndAction(pool, trace_id);
      if (row) await _insertMemoryItem(pool, { mem_id: uuidv4(), project_id: row.project_id || "general", type: "sop", content: row.action_json || {}, tags: "auto_generated" });
    }
    return res.json({ ok: true, trace_id, message: "Feedback recorded and rules/memories updated if applicable." });
  } catch (err) { return res.status(500).json({ ok: false, error: err.message }); }
});

async function main() {
  try { await _ensureOrchestratorSchema(pool); } catch (err) { console.warn("[orchestrator] schema ensure failed:", err.message); }
  try { await redis.xgroup("CREATE", STREAM_TASK, GROUP_TASK, "$", "MKSTREAM"); } catch {}
  try { await redis.xgroup("CREATE", STREAM_RESULT, GROUP_RESULT, "$", "MKSTREAM"); } catch {}
  resultConsumer.start();
  taskWatchdog.start();
  app.listen(3000, () => console.log("Orchestrator listening on :3000"));
}

main().catch(console.error);
