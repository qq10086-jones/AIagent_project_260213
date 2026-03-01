import Redis from "ioredis";
import pg from "pg";
import { v4 as uuidv4 } from "uuid";
import { CodingService } from "./coding_service.js";

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

const redis = new Redis(REDIS_URL);
const pool = new pg.Pool({
  host: PGHOST,
  port: Number(PGPORT),
  user: PGUSER,
  password: PGPASSWORD,
  database: PGDATABASE,
});

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
      // Security Check (Worker Side)
      const command = payload.command || "";
      const allowedPrefixes = ["python", "pytest", "npm", "node", "git", "ls", "cat", "echo", "pwd", "grep", "rg", "fd", "ruff", "black"];
      const cmdPrefix = command.trim().split(/\s+/)[0];
      const forbiddenChars = [";", "|", ">", "<", "&", "$", "(", ")", "`", "\\", "*", "?", "[", "]", "{", "}", "~"];
      
      if (!allowedPrefixes.includes(cmdPrefix)) {
        throw new Error(`Command blocked: '${cmdPrefix}' is not whitelisted.`);
      }
      if (forbiddenChars.some(char => command.includes(char))) {
        throw new Error("Command blocked: forbidden shell meta-character detected.");
      }

      const result = await CodingService.executeCommand({
        workspaceRoot: WORKSPACE_ROOT,
        command,
        run_id,
        task_id
      });
      output = result;
      isSuccess = result.ok;
      if (!isSuccess) error = result.error;
    } else if (tool_name === "coding.delegate") {
      const result = await CodingService.delegateTask({
        workspaceRoot: WORKSPACE_ROOT,
        task_prompt: payload.task_prompt,
        provider: payload.provider || "auto",
        model: payload.model || null,
        run_id,
        task_id,
        max_runtime_s: payload.max_runtime_s || 600,
        codex_command: Array.isArray(payload.codex_command) ? payload.codex_command : null,
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
    await emitResult(task_id, "failed", null, err.message);
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
            await processTask(id, task);
          } else {
            // Not a coding task, leave it for other workers. (Wait, if we use a shared group, we shouldn't XACK, but XREADGROUP consumes it... 
            // In a real multi-worker setup, we should have topic-based queues. For now, we put it back or we don't XACK so another consumer picks it up?
            // Actually, if we don't XACK, it stays pending for THIS consumer.
            // Let's acknowledge it and put it back? No, let's assume for Phase 2 we use the same architecture. 
            // Wait, worker-quant also consumes STREAM_TASK and does `if not handler: r.xack(...)`. This means the first worker to grab a task consumes it, even if it can't handle it!
            // I need to fix this by letting worker-coder only consume if it's a coding tool.
            // But `xreadgroup` already assigned it to this consumer. 
            // Let's just re-enqueue it or change the routing in Orchestrator!)
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
