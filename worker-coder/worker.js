import Redis from "ioredis";
import pg from "pg";
import crypto from "crypto";
import { pathToFileURL } from "url";
import { CodingService } from "./coding_service.js";
import { createTaskLifecycle } from "./task_lifecycle.js";
import { validateSafeCommand } from "./verification_runner.js";
import { loadRuntimeConfig } from "./runtime_config.js";
import { validateRuntimePreflight } from "./provider_preflight.js";
import { pushBranchToGitHub } from "./git_side_effects.js";
import { validateWorkerResult } from "../shared/contracts/worker_result.js";
import { createAuditHooks } from "../shared/contracts/audit_hooks.js";
import { isSingleAgentPayload, validateSingleAgentPayloadGuardrails, validateSingleAgentWorkerResultGuardrails } from "../shared/contracts/single_agent_guardrails.js";

const {
  REDIS_URL = "redis://localhost:6379",
  PGHOST = "localhost",
  PGPORT = 5432,
  PGUSER = "nexus",
  PGPASSWORD = "nexuspassword",
  PGDATABASE = "nexus",
  WORKSPACE_ROOT = "/workspace"
} = process.env;

const STREAM_TASK = process.env.STREAM_TASK || "stream:task:coding";
const STREAM_RESULT = "stream:result";
const GROUP = process.env.GROUP_TASK || "cg:workers:coding";
const CONSUMER = `coder-${crypto.randomUUID().slice(0, 8)}`;
const RUNTIME = loadRuntimeConfig();
const RUNTIME_CODER = RUNTIME.config?.worker_coder || {};

function parseBool(value, fallback = false) {
  if (typeof value === "boolean") return value;
  const safe = String(value || "").trim().toLowerCase();
  if (!safe) return fallback;
  if (["1", "true", "yes", "on"].includes(safe)) return true;
  if (["0", "false", "no", "off"].includes(safe)) return false;
  return fallback;
}

const DEFAULT_PROVIDER = String(process.env.CODER_PROVIDER_DEFAULT || RUNTIME_CODER.provider_default || "auto").toLowerCase();
const DEFAULT_MODEL = String(process.env.CODER_MODEL_DEFAULT || RUNTIME_CODER.model_default || "");
const DEFAULT_EXECUTION_LANE = String(process.env.CODER_EXECUTION_LANE_DEFAULT || RUNTIME_CODER.execution_lane_default || "").trim();
const DEFAULT_ALLOW_PROVIDER_FALLBACK = parseBool(
  process.env.CODER_ALLOW_PROVIDER_FALLBACK,
  parseBool(RUNTIME_CODER.allow_provider_fallback_default, false),
);
const GLOBAL_TASK_TIMEOUT_MS = Math.max(30000, Number(process.env.CODER_GLOBAL_TASK_TIMEOUT_MS || RUNTIME_CODER.global_task_timeout_ms || 900000));
// stream_batch_size controls how many Redis messages are fetched per loop iteration.
// Set to 1 for single-GPU (ollama) setups to prevent concurrent LLM saturation.
// Increase for multi-GPU or cloud provider setups.
const STREAM_BATCH_SIZE = Math.max(1, Math.min(20, Number(process.env.CODER_STREAM_BATCH_SIZE || RUNTIME_CODER.stream_batch_size || 1)));
import {
  MAX_RESULT_OUTPUT_BYTES,
  STALE_TASK_MIN_IDLE_MS as STALE_IDLE_MS_CONST,
} from './constants.js';

console.log(
  `[runtime-config] path=${RUNTIME.path || "none"} provider_default=${DEFAULT_PROVIDER} model_default=${DEFAULT_MODEL || "none"} execution_lane_default=${DEFAULT_EXECUTION_LANE || "none"} allow_provider_fallback=${DEFAULT_ALLOW_PROVIDER_FALLBACK} global_timeout_ms=${GLOBAL_TASK_TIMEOUT_MS} stream_batch_size=${STREAM_BATCH_SIZE}`
);
const STARTUP_PREFLIGHT = validateRuntimePreflight({
  defaultProvider: DEFAULT_PROVIDER,
  defaultModel: DEFAULT_MODEL,
  defaultExecutionLane: DEFAULT_EXECUTION_LANE,
  runtimeCoderConfig: RUNTIME_CODER,
});
if (STARTUP_PREFLIGHT.ok) {
  console.log("[startup-preflight] ok");
} else {
  for (const issue of STARTUP_PREFLIGHT.issues) {
    console.warn(`[startup-preflight] ${issue.severity} lane=${issue.lane} code=${issue.code} message=${issue.message}`);
  }
  if (STARTUP_PREFLIGHT.fatal) {
    throw new Error("startup preflight failed with fatal runtime configuration issues");
  }
}

const redis = new Redis(REDIS_URL);
const pool = new pg.Pool({
  host: PGHOST,
  port: Number(PGPORT),
  user: PGUSER,
  password: PGPASSWORD,
  database: PGDATABASE,
});
const AUDIT_HOOKS = createAuditHooks({ pool, logger: console });

async function emitResult(task_id, status, output, error) {
  const msg = { task_id, status };
  if (output) msg.output = serializeResultOutput(output);
  if (error) msg.error = String(error);
  await redis.xadd(STREAM_RESULT, "*", ...Object.entries(msg).flat());
}

async function emitPiSessionUpdate(task_id, tool_name, event) {
  if (!event || typeof event !== "object") return;
  await emitResult(task_id, "pi_session_update", {
    tool_name,
    event,
  });
}

function truncateString(value, maxLen = 2000) {
  const safe = String(value || "");
  if (safe.length <= maxLen) return safe;
  return `${safe.slice(0, maxLen)}...<truncated>`;
}

function compactResultOutput(output) {
  if (!output || typeof output !== "object") {
    return output;
  }
  const diagnostics = output.diagnostics && typeof output.diagnostics === "object"
    ? output.diagnostics
    : {};
  return {
    ok: Boolean(output.ok),
    error: output.error ? truncateString(output.error, 1000) : null,
    summary: output.summary ? truncateString(output.summary, 1000) : null,
    provider_used: output.provider_used || null,
    provider_requested: output.provider_requested || null,
    model_used: output.model_used || null,
    execution_lane: output.execution_lane || null,
    command_source: output.command_source || null,
    files_changed: Array.isArray(output.files_changed) ? output.files_changed.slice(0, 20) : [],
    diff_stats: output.diff_stats || null,
    test_result: output.test_result || null,
    artifacts: output.artifacts || null,
    worker_result: output.worker_result || null,
    diagnostics: {
      error_code: diagnostics.error_code || null,
      provider_error_class: diagnostics.provider_error_class || null,
      model_provider: diagnostics.model_provider || null,
      model_name: diagnostics.model_name || null,
      fallback_taken: diagnostics.fallback_taken || false,
      fallback_from: diagnostics.fallback_from || null,
      fallback_target: diagnostics.fallback_target || null,
      failure_attribution: diagnostics.failure_attribution || null,
      retry_summary: diagnostics.retry_summary || null,
      promotion: diagnostics.promotion || null,
      final_failure_summary: diagnostics.final_failure_summary || null,
    },
  };
}

function normalizePermissionDecision(permissionAdvice = {}, { toolName = "", source = "permission_council" } = {}) {
  if (!permissionAdvice || typeof permissionAdvice !== "object") return null;
  const riskScoreRaw = Number(permissionAdvice.risk_score);
  return {
    source,
    tool_name: toolName || null,
    recommendation: String(permissionAdvice.council_advice || "review"),
    safety_verdict: permissionAdvice.safety_verdict ? String(permissionAdvice.safety_verdict) : null,
    context_verdict: permissionAdvice.context_verdict ? String(permissionAdvice.context_verdict) : null,
    risk_level: permissionAdvice.risk_level ? String(permissionAdvice.risk_level) : null,
    risk_score: Number.isFinite(riskScoreRaw) ? riskScoreRaw : null,
    summary: permissionAdvice.advisory_summary ? String(permissionAdvice.advisory_summary) : null,
  };
}

function buildWorkerResultEnvelope({ runId, toolName, ok, output, error, payload = {} }) {
  const boundedValidation = Array.isArray(output?.metadata?.bounded_validation) && output.metadata.bounded_validation.length > 0
    ? output.metadata.bounded_validation
    : [{
      name: "result_shape",
      ok: Boolean(output && typeof output === "object"),
      detail: output && typeof output === "object" ? "object_result_emitted" : "non_object_result",
    }];
  const advisoryDecision = normalizePermissionDecision(payload?.permission_advisory, { toolName });
  const permissionDecisions = Array.isArray(output?.metadata?.permission_decisions)
    ? output.metadata.permission_decisions
    : Array.isArray(output?.permission_decisions)
      ? output.permission_decisions
      : advisoryDecision
        ? [advisoryDecision]
        : [];
  const validation = validateWorkerResult({
    run_id: runId,
    worker_name: "worker-coder",
    status: ok ? "succeeded" : "failed",
    ok,
    output,
    error,
    metadata: {
      duration_ms: Number(output?.duration_ms || 0),
      tool_calls: Number(output?.diagnostics?.tool_calls || output?.tool_calls || 0),
      permission_decisions: permissionDecisions,
      evidence_id: output?.metadata?.evidence_id || payload?.evidence_id || payload?.task_envelope?.evidence_id || "",
      replay_tag: output?.metadata?.replay_tag || payload?.replay_tag || payload?.task_envelope?.replay_tag || "",
      bounded_validation: boundedValidation,
      source_tool: toolName,
    },
  });
  return validation.value;
}

function serializeResultOutput(output) {
  const full = JSON.stringify(output);
  if (Buffer.byteLength(full, "utf8") <= MAX_RESULT_OUTPUT_BYTES) {
    return full;
  }
  const compact = JSON.stringify({
    truncated: true,
    payload: compactResultOutput(output),
  });
  if (Buffer.byteLength(compact, "utf8") <= MAX_RESULT_OUTPUT_BYTES) {
    return compact;
  }
  return JSON.stringify({
    truncated: true,
    payload: {
      ok: Boolean(output?.ok),
      error: truncateString(output?.error || "", 1000),
      provider_used: output?.provider_used || null,
      execution_lane: output?.execution_lane || null,
      diagnostics: {
        error_code: output?.diagnostics?.error_code || null,
        provider_error_class: output?.diagnostics?.provider_error_class || null,
      },
    },
  });
}

async function writeFact(run_id, agent_name, payload) {
  try {
    const fact_id = crypto.randomUUID();
    await pool.query(
      "INSERT INTO fact_items (fact_id, run_id, agent_name, kind, payload_json) VALUES ($1, $2, $3, $4, $5)",
      [fact_id, run_id, agent_name, "tool_result", JSON.stringify(payload)]
    );
  } catch (err) {
    console.error("[worker] Failed to write fact:", err.message);
  }
}

async function processTask(msgId, task, lifecycle) {
  const { task_id, tool_name, run_id, payload: rawPayload } = task;

  const HANDLED_TOOLS = new Set(["ops.deploy_preview"]);
  if (!tool_name.startsWith("coding.") && !HANDLED_TOOLS.has(tool_name)) {
    return false;
  }

  let payload = {};
  try {
    payload = JSON.parse(rawPayload || "{}");
  } catch (parseErr) {
    console.warn(`[worker] Failed to parse task payload for ${task_id}:`, parseErr.message);
  }

  let output = {};
  let error = null;
  let isSuccess = false;

  try {
    const payloadGuardrails = validateSingleAgentPayloadGuardrails(payload);
    if (!payloadGuardrails.ok) {
      throw new Error(`SINGLE_AGENT_GUARDRAILS_INVALID: ${payloadGuardrails.errors.join("; ")}`);
    }
    const permissionAdvice = payload?.permission_advisory && typeof payload.permission_advisory === "object"
      ? payload.permission_advisory
      : null;
    console.log(`[worker] Claimed task ${task_id} [${tool_name}]`);
    await lifecycle.emitClaimed();
    await emitResult(task_id, "tool_call", {
      tool_name,
      input_summary: truncateString(payload?.task_prompt || payload?.prompt || payload?.command || "", 240),
    });
    await AUDIT_HOOKS.onToolCall(run_id, tool_name, payload, {
      taskId: task_id,
      workerName: "worker-coder",
      toolName: tool_name,
      permissionAdvice,
    });
    if (tool_name === "coding.patch") {
      const result = await CodingService.applyPatch({
        workspaceRoot: WORKSPACE_ROOT,
        file_path: payload.file_path,
        edit_block: payload.edit_block,
        target_paths: Array.isArray(payload.target_paths) ? payload.target_paths : [],
        task_id,
        run_id
      });
      output = result;
      isSuccess = result.success;
      if (!isSuccess) error = result.message;
    } else if (tool_name === "coding.execute") {
      const command = String(payload.command || "").trim();
      const checked = validateSafeCommand(command);
      if (!checked.ok) throw new Error(checked.error);

      const result = await CodingService.executeCommand({
        workspaceRoot: WORKSPACE_ROOT,
        command,
        artifact_root: payload.artifact_root || "",
        expected_artifacts: Array.isArray(payload.expected_artifacts) ? payload.expected_artifacts : [],
        step_id: payload.step_id || "",
        task_prompt: payload.task_prompt || payload.prompt || "",
        run_id,
        task_id
      });
      output = result;
      isSuccess = result.ok;
      if (!isSuccess) error = result.error;
    } else if (tool_name === "coding.delegate") {
      const hasFallbackFlag = Object.prototype.hasOwnProperty.call(payload, "allow_provider_fallback");
      const requestedModel = payload.model_override || payload.model || DEFAULT_MODEL || null;
      const result = await CodingService.delegateTask({
        workspaceRoot: WORKSPACE_ROOT,
        task_prompt: payload.task_prompt || payload.prompt,
        artifact_root: payload.artifact_root || "",
        expected_artifacts: Array.isArray(payload.expected_artifacts) ? payload.expected_artifacts : [],
        step_id: payload.step_id || "",
        target_paths: Array.isArray(payload.target_paths) ? payload.target_paths : [],
        provider: payload.provider || DEFAULT_PROVIDER,
        model: requestedModel,
        execution_lane: payload.execution_lane || DEFAULT_EXECUTION_LANE || null,
        allow_provider_fallback: hasFallbackFlag
          ? parseBool(payload.allow_provider_fallback, DEFAULT_ALLOW_PROVIDER_FALLBACK)
          : DEFAULT_ALLOW_PROVIDER_FALLBACK,
        runtime_coder_config: RUNTIME_CODER,
        run_id,
        task_id,
        max_runtime_s: payload.max_runtime_s || 600,
        max_attempts: payload.max_attempts || 1,
        same_error_repeat_limit: payload.same_error_repeat_limit || 2,
        wall_clock_timeout_s: payload.wall_clock_timeout_s || 0,
        codex_command: Array.isArray(payload.codex_command) ? payload.codex_command : null,
        opencode_command: Array.isArray(payload.opencode_command) ? payload.opencode_command : null,
        verification_command: payload.verification_command || "",
        verification_plan: Array.isArray(payload.verification_plan) ? payload.verification_plan : null,
        execution_adapter_packet: payload.execution_adapter_packet || null,
        context_packet: payload.context_packet || null,
        repo_map: payload.repo_map || null,
        task_class: payload.task_class || null,
        beta_template_id: payload.beta_template_id || null,
        context_envelope: payload.context_envelope || null,
        on_runtime_event: async (event) => {
          await emitPiSessionUpdate(task_id, tool_name, event);
        },
        redis,
      });
      output = result;
      isSuccess = !!result.ok;
      if (!isSuccess) error = result.error || "coding.delegate failed";
    } else if (tool_name === "ops.deploy_preview") {
      const result = await pushBranchToGitHub({
        workspaceRoot: WORKSPACE_ROOT,
        runId: run_id,
        goal: payload.goal || payload.task_prompt || "",
        githubRepo: payload.github_repo || process.env.GITHUB_REPO,
        githubToken: payload.github_token || process.env.GITHUB_TOKEN,
        branchPrefix: payload.branch_prefix || "nexus/",
      });
      output = result;
      isSuccess = result.ok;
      if (!isSuccess) error = result.message || result.reason || "GitHub delivery failed";
    } else {
      throw new Error(`Unknown tool: ${tool_name}`);
    }

    if (output && typeof output === "object" && !output.worker_result) {
      output = {
        ...output,
        worker_result: buildWorkerResultEnvelope({
          runId: run_id,
          toolName: tool_name,
          ok: isSuccess,
          output,
          error,
          payload,
        }),
      };
    }
    if (isSingleAgentPayload(payload)) {
      const workerResultGuardrails = validateSingleAgentWorkerResultGuardrails(output?.worker_result, { requireSingleAgent: true });
      if (!workerResultGuardrails.ok) {
        throw new Error(`SINGLE_AGENT_GUARDRAILS_INVALID: ${workerResultGuardrails.errors.join("; ")}`);
      }
    }

    await AUDIT_HOOKS.onToolResult(run_id, tool_name, {
      ok: isSuccess,
      error,
      worker_result: output?.worker_result || null,
    }, {
      taskId: task_id,
      workerName: "worker-coder",
      toolName: tool_name,
    });
    await emitResult(task_id, "tool_result", {
      tool_name,
      ok: isSuccess,
      summary: truncateString(
        output?.summary
        || output?.error
        || output?.diagnostics?.final_failure_summary
        || output?.command_source
        || "",
        240,
      ),
      result_summary: output?.worker_result?.metadata?.bounded_validation?.[0]?.detail || "",
    });

    await lifecycle.finalizeResult({
      ok: isSuccess,
      output,
      error,
    });

  } catch (err) {
    console.error(`[worker] Task failed:`, err);
    await lifecycle.finalizeExecutionFailure(err);
  }
  return true;
}

export async function main() {
  console.log(`[worker] Starting Worker-Coder (${CONSUMER})...`);

  try {
    await redis.xgroup("CREATE", STREAM_TASK, GROUP, "0", "MKSTREAM");
  } catch (e) {
    if (!e.message.includes("BUSYGROUP")) throw e;
  }

async function processMessages(messages) {
    const promises = messages.map(async ([id, fieldValues]) => {
      const task = {};
      for (let i = 0; i < fieldValues.length; i += 2) {
        task[fieldValues[i]] = fieldValues[i + 1];
      }
      let parsedPayload = {};
      try {
        parsedPayload = JSON.parse(task.payload || "{}");
      } catch (parseErr) {
        console.warn(`[worker] Failed to parse stream payload for ${task.task_id}:`, parseErr.message);
        parsedPayload = {};
      }
      
      const WORKER_HANDLED = task.tool_name && (
        task.tool_name.startsWith("coding.") || task.tool_name === "ops.deploy_preview"
      );
      if (WORKER_HANDLED) {
        console.log(`[worker] Processing task ${task.task_id} (${task.tool_name})...`);
        const lifecycle = createTaskLifecycle({
          taskId: task.task_id,
          msgId: id,
          toolName: task.tool_name,
          runId: task.run_id,
          emitResult,
          writeFact,
          ackMessage: async (messageId) => {
            await redis.xack(STREAM_TASK, GROUP, messageId);
          },
          auditHooks: AUDIT_HOOKS,
          workerName: "worker-coder",
          taskEnvelope: {
            task_id: task.task_id,
            tool_name: task.tool_name,
            evidence_id: parsedPayload.evidence_id || parsedPayload.task_envelope?.evidence_id || null,
            replay_tag: parsedPayload.replay_tag || parsedPayload.task_envelope?.replay_tag || null,
            task_envelope: parsedPayload.task_envelope || null,
            payload: parsedPayload,
          },
        });
        try {
          const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error("GLOBAL_TASK_TIMEOUT")), GLOBAL_TASK_TIMEOUT_MS));
          await Promise.race([processTask(id, task, lifecycle), timeoutPromise]);
          console.log(`[worker] Successfully processed and acknowledged task ${task.task_id}`);
        } catch (err) {
          console.error(`[worker] Critical error processing task ${task.task_id}:`, err.message);
          await lifecycle.finalizeTimeout(err);
        }
      } else {
        console.warn(`[worker] Received non-coding task ${task.task_id} (${task.tool_name}), acknowledging and skipping.`);
        await redis.xack(STREAM_TASK, GROUP, id);
      }
    });
    await Promise.allSettled(promises);
  }

  let claimStartId = "0-0";
  const STALE_TASK_MIN_IDLE_MS = STALE_IDLE_MS_CONST;

  while (true) {
    try {
      if (typeof redis.xautoclaim === "function") {
        const claimRes = await redis.xautoclaim(STREAM_TASK, GROUP, CONSUMER, STALE_TASK_MIN_IDLE_MS, claimStartId, "COUNT", STREAM_BATCH_SIZE);
        if (claimRes) {
          claimStartId = Array.isArray(claimRes) ? String(claimRes[0] || "0-0") : "0-0";
          const claimedMessages = Array.isArray(claimRes?.[1]) ? claimRes[1] : [];
          if (claimedMessages.length > 0) {
            console.log(`[worker] XAUTOCLAIM recovered ${claimedMessages.length} stale messages.`);
            await processMessages(claimedMessages);
            continue;
          }
        }
      }

      const res = await redis.xreadgroup("GROUP", GROUP, CONSUMER, "COUNT", STREAM_BATCH_SIZE, "BLOCK", 5000, "STREAMS", STREAM_TASK, ">");
      if (res && res.length > 0) {
        const stream = res[0];
        const messages = stream[1];
        await processMessages(messages);
      }
    } catch (e) {
      console.error("[worker] Loop error:", e.message);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(console.error);
}
