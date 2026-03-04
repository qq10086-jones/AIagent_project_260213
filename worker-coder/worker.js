import Redis from "ioredis";
import pg from "pg";
import { v4 as uuidv4 } from "uuid";
import { CodingService } from "./coding_service.js";
import { loadRuntimeConfig } from "./runtime_config.js";

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
const CONSUMER = `coder-${uuidv4().slice(0, 8)}`;
const RUNTIME = loadRuntimeConfig();
const RUNTIME_CODER = RUNTIME.config?.worker_coder || {};
const DEFAULT_PROVIDER = String(process.env.CODER_PROVIDER_DEFAULT || RUNTIME_CODER.provider_default || "auto").toLowerCase();
const DEFAULT_MODEL = String(process.env.CODER_MODEL_DEFAULT || RUNTIME_CODER.model_default || "");
const GLOBAL_TASK_TIMEOUT_MS = Math.max(30000, Number(process.env.CODER_GLOBAL_TASK_TIMEOUT_MS || RUNTIME_CODER.global_task_timeout_ms || 900000));
console.log(
  `[runtime-config] path=${RUNTIME.path || "none"} provider_default=${DEFAULT_PROVIDER} model_default=${DEFAULT_MODEL || "none"} global_timeout_ms=${GLOBAL_TASK_TIMEOUT_MS}`
);

const redis = new Redis(REDIS_URL);
const pool = new pg.Pool({
  host: PGHOST,
  port: Number(PGPORT),
  user: PGUSER,
  password: PGPASSWORD,
  database: PGDATABASE,
});

const ALLOWED_CMD_PREFIXES = new Set([
  "python",
  "pytest",
  "npm",
  "node",
  "git",
  "ls",
  "cat",
  "echo",
  "pwd",
  "grep",
  "rg",
  "fd",
  "ruff",
  "black",
]);

function splitCommandChain(command) {
  return String(command || "")
    .split("&&")
    .map((x) => x.trim())
    .filter(Boolean);
}

function validateExecuteCommand(command) {
  const raw = String(command || "").trim();
  if (!raw) return { ok: false, error: "Command blocked: empty command." };
  // Only allow chained commands through "&&". Block other shell-control/meta chars.
  if (/[;|><`$(){}[\]\\*?~\n\r]/.test(raw)) {
    return { ok: false, error: "Command blocked: forbidden shell meta-character detected." };
  }
  // Reject a standalone '&' that is not part of a "&&" chain.
  if (/(^|[^&])&([^&]|$)/.test(raw)) {
    return { ok: false, error: "Command blocked: unsupported '&' operator." };
  }
  const segments = splitCommandChain(raw);
  if (segments.length === 0) {
    return { ok: false, error: "Command blocked: empty command segment." };
  }
  for (const segment of segments) {
    const cmdPrefix = segment.trim().split(/\s+/)[0];
    if (!ALLOWED_CMD_PREFIXES.has(cmdPrefix)) {
      return { ok: false, error: `Command blocked: '${cmdPrefix}' is not whitelisted.` };
    }
  }
  return { ok: true };
}

async function emitResult(task_id, status, output, error) {
  const msg = { task_id, status };
  if (output) msg.output = JSON.stringify(output);
  if (error) msg.error = String(error);
  await redis.xadd(STREAM_RESULT, "*", ...Object.entries(msg).flat());
}

async function writeFact(run_id, agent_name, payload) {
  try {
    const fact_id = uuidv4();
    await pool.query(
      "INSERT INTO fact_items (fact_id, run_id, agent_name, kind, payload_json) VALUES ($1, $2, $3, $4, $5)",
      [fact_id, run_id, agent_name, "tool_result", JSON.stringify(payload)]
    );
  } catch (err) {
    console.error("[worker] Failed to write fact:", err.message);
  }
}

async function processTask(msgId, task) {
  const { task_id, tool_name, run_id, payload: rawPayload } = task;
  
  if (!tool_name.startsWith("coding.")) {
    return false; // not my job
  }

  let payload = {};
  try {
    payload = JSON.parse(rawPayload || "{}");
  } catch {}

  console.log(`[worker] Claimed task ${task_id} [${tool_name}]`);
  await emitResult(task_id, "claimed");

  let output = {};
  let error = null;
  let isSuccess = false;

  try {
    if (tool_name === "coding.patch") {
      const result = await CodingService.applyPatch({
        workspaceRoot: WORKSPACE_ROOT,
        file_path: payload.file_path,
        edit_block: payload.edit_block,
        task_id,
        run_id
      });
      output = result;
      isSuccess = result.success;
      if (!isSuccess) error = result.message;
    } else if (tool_name === "coding.execute") {
      const command = String(payload.command || "").trim();
      const checked = validateExecuteCommand(command);
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
      const result = await CodingService.delegateTask({
        workspaceRoot: WORKSPACE_ROOT,
        task_prompt: payload.task_prompt || payload.prompt,
        artifact_root: payload.artifact_root || "",
        expected_artifacts: Array.isArray(payload.expected_artifacts) ? payload.expected_artifacts : [],
        step_id: payload.step_id || "",
        provider: payload.provider || DEFAULT_PROVIDER,
        model: payload.model || DEFAULT_MODEL || null,
        run_id,
        task_id,
        max_runtime_s: payload.max_runtime_s || 600,
        codex_command: Array.isArray(payload.codex_command) ? payload.codex_command : null,
        opencode_command: Array.isArray(payload.opencode_command) ? payload.opencode_command : null,
      });
      output = result;
      isSuccess = !!result.ok;
      if (!isSuccess) error = result.error || "coding.delegate failed";
    } else {
      throw new Error(`Unknown tool: ${tool_name}`);
    }

    // Write fact so Brain can consume it
    await writeFact(run_id, "coder", { tool_name, output, success: isSuccess });
    await emitResult(task_id, isSuccess ? "succeeded" : "failed", output, error);

  } catch (err) {
    console.error(`[worker] Task failed:`, err);
    await writeFact(run_id, "coder", { tool_name, error: err.message, success: false });
    await emitResult(task_id, "failed", { error: err.message, plan: "failed_during_execution" }, err.message);
  }

  await redis.xack(STREAM_TASK, GROUP, msgId);
  return true;
}

async function main() {
  console.log(`[worker] Starting Worker-Coder (${CONSUMER})...`);
  
  try {
    await redis.xgroup("CREATE", STREAM_TASK, GROUP, "0", "MKSTREAM");
  } catch (e) {
    if (!e.message.includes("BUSYGROUP")) throw e;
  }

  while (true) {
    try {
      const res = await redis.xreadgroup("GROUP", GROUP, CONSUMER, "COUNT", 1, "BLOCK", 5000, "STREAMS", STREAM_TASK, ">");
      if (res && res.length > 0) {
        const stream = res[0];
        const messages = stream[1];
        for (const [id, fieldValues] of messages) {
          const task = {};
          for (let i = 0; i < fieldValues.length; i += 2) {
            task[fieldValues[i]] = fieldValues[i + 1];
          }
          
          if (task.tool_name && task.tool_name.startsWith("coding.")) {
            console.log(`[worker] Processing task ${task.task_id} (${task.tool_name})...`);
            try {
              const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error("GLOBAL_TASK_TIMEOUT")), GLOBAL_TASK_TIMEOUT_MS));
              await Promise.race([processTask(id, task), timeoutPromise]);
              // After processTask finishes, it should have already called emitResult and xack inside.
              console.log(`[worker] Successfully processed and acknowledged task ${task.task_id}`);
            } catch (err) {
              console.error(`[worker] Critical error processing task ${task.task_id}:`, err.message);
              await emitResult(task.task_id, "failed", { error: err.message }, err.message);
              await redis.xack(STREAM_TASK, GROUP, id);
            }
          } else {
            // NOT a coding task. In a shared queue model, we MUST acknowledge it so it doesn't stay pending for US.
            // Ideally, Orchestrator should only send relevant tasks to this stream.
            console.warn(`[worker] Received non-coding task ${task.task_id} (${task.tool_name}), acknowledging and skipping.`);
            await redis.xack(STREAM_TASK, GROUP, id);
          }
        }
      }
    } catch (e) {
      console.error("[worker] Loop error:", e.message);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}

main().catch(console.error);
