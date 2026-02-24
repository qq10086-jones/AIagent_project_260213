import express from "express";
import Redis from "ioredis";
import pg from "pg";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";

const {
  REDIS_URL,
  PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE,
  STREAM_TASK = "stream:task",
  STREAM_RESULT = "stream:result",
  GROUP_TASK = "cg:workers",
  GROUP_RESULT = "cg:orchestrator",
  APPROVAL_TOKEN = "dev-approval-token",
} = process.env;

const redis = new Redis(REDIS_URL);
const pool = new pg.Pool({
  host: PGHOST, port: Number(PGPORT || 5432), user: PGUSER, password: PGPASSWORD, database: PGDATABASE,
});

async function dbEvent(task_id, event_type, payload = {}) {
  await pool.query("INSERT INTO event_log(task_id, event_type, payload_json) VALUES ($1,$2,$3)", [task_id, event_type, JSON.stringify(payload)]);
}

async function upsertTask(task) {
  await pool.query(`INSERT INTO tasks(task_id, tool_name, status, risk_level, payload_json) VALUES ($1,$2,$3,$4,$5) ON CONFLICT (task_id) DO UPDATE SET status=EXCLUDED.status, updated_at=NOW()`, [task.task_id, task.tool_name, task.status, task.risk_level, JSON.stringify(task.payload)]);
}

async function handleNextWorkflowStep(workflowId, prevStepIndex, prevOutput) {
  const w = await pool.query("SELECT * FROM workflows WHERE workflow_id=$1", [workflowId]);
  if (!w.rows[0]) return;
  const definition = JSON.parse(w.rows[0].definition_json);
  const nextStepIndex = prevStepIndex + 1;
  const nextStep = definition.steps[nextStepIndex];
  if (!nextStep) return;
  const task_id = uuidv4();
  const payload = { ...nextStep.payload, prev_output: prevOutput };
  const risk_level = nextStep.risk_level || "low";
  
  // Guard check: if high/medium, start as waiting_approval
  const status = (risk_level === "high" || risk_level === "medium") ? "waiting_approval" : "queued";
  
  await upsertTask({ task_id, tool_name: nextStep.tool_name, status, risk_level, payload });
  await dbEvent(task_id, "task.created", { tool_name: nextStep.tool_name, workflow_id: workflowId, step_index: nextStepIndex });
  
  if (status === "queued") {
    await redis.xadd(STREAM_TASK, "*", "task_id", task_id, "tool_name", nextStep.tool_name, "payload", JSON.stringify(payload), "risk_level", risk_level, "workflow_id", workflowId, "step_index", String(nextStepIndex));
  } else {
    await dbEvent(task_id, "approval.requested", { reason: `risk_level=${risk_level}` });
  }
}

async function startResultConsumer() {
  const consumer = "orchestrator-1";
  while (true) {
    try {
      const res = await redis.xreadgroup("GROUP", GROUP_RESULT, consumer, "BLOCK", 5000, "COUNT", 20, "STREAMS", STREAM_RESULT, ">");
      if (!res) continue;
      for (const [, messages] of res) {
        for (const [id, kv] of messages) {
          const obj = {};
          for (let i = 0; i < kv.length; i += 2) obj[kv[i]] = kv[i + 1];
          const task_id = obj.task_id;
          const status = obj.status || "succeeded";
          console.log(`[result] task=${task_id} status=${status} wf=${obj.workflow_id} step=${obj.step_index}`);
          const output = obj.output ? JSON.parse(obj.output) : null;
          if (status === "claimed") {
            await pool.query("UPDATE tasks SET status=$1, updated_at=NOW() WHERE task_id=$2", ["running", task_id]);
          } else {
            await pool.query("UPDATE tasks SET status=$1, updated_at=NOW() WHERE task_id=$2", [status, task_id]);
            await dbEvent(task_id, `task.${status}`, { output });

            // --- Asset Persistence Logic (Robust) ---
            let realOutput = output;
            if (typeof output === 'string') {
              try { realOutput = JSON.parse(output); } catch(e) {}
            }
            
            if (status === "succeeded" && realOutput && Array.isArray(realOutput.artifacts)) {
              for (const art of realOutput.artifacts) {
                await pool.query(
                  "INSERT INTO assets(task_id, object_key, sha256, mime_type, file_size, metadata_json) VALUES ($1,$2,$3,$4,$5,$6)",
                  [task_id, art.object_key, art.sha256, art.mime, art.size, JSON.stringify({ name: art.name })]
                );
                console.log(`[asset] recorded: ${art.name} -> ${art.object_key}`);
              }
            }
          }
          await redis.xack(STREAM_RESULT, GROUP_RESULT, id);
          if (status === "succeeded" && obj.workflow_id) {
            await handleNextWorkflowStep(obj.workflow_id, parseInt(obj.step_index), output);
          }
        }
      }
    } catch (e) { await new Promise(r => setTimeout(r, 1000)); }
  }
}

const app = express();
app.use(express.json());
app.get("/health", (_, res) => res.send("ok"));

app.post("/tasks/:id/approve", async (req, res) => {
  const tok = req.header("X-Approval-Token");
  if (tok !== APPROVAL_TOKEN) return res.status(401).send("unauthorized");
  const t = await pool.query("SELECT * FROM tasks WHERE task_id=$1", [req.params.id]);
  if (!t.rows[0]) return res.status(404).send("not found");
  await pool.query("UPDATE tasks SET status='queued', updated_at=NOW() WHERE task_id=$1", [req.params.id]);
  await dbEvent(req.params.id, "approval.approved", {});
  const row = t.rows[0];
  // 重新获取该任务关联的 workflow 信息（如果有）
  const e = await pool.query("SELECT payload_json FROM event_log WHERE task_id=$1 AND event_type='task.created'", [req.params.id]);
  const meta = JSON.parse(e.rows[0]?.payload_json || "{}");
  await redis.xadd(STREAM_TASK, "*", "task_id", req.params.id, "tool_name", row.tool_name, "payload", row.payload_json, "risk_level", row.risk_level, "workflow_id", meta.workflow_id || "", "step_index", String(meta.step_index || "0"));
  res.json({ ok: true });
});

app.post("/workflows", async (req, res) => {
  const { name, definition } = req.body;
  const workflow_id = uuidv4();
  await pool.query("INSERT INTO workflows(workflow_id, name, definition_json) VALUES ($1,$2,$3)", [workflow_id, name, JSON.stringify(definition)]);
  await handleNextWorkflowStep(workflow_id, -1, null);
  res.json({ workflow_id, status: "started" });
});

async function main() {
  try { await redis.xgroup("CREATE", STREAM_TASK, GROUP_TASK, "$", "MKSTREAM"); } catch (e) {}
  try { await redis.xgroup("CREATE", STREAM_RESULT, GROUP_RESULT, "$", "MKSTREAM"); } catch (e) {}
  startResultConsumer();
  app.listen(3000, () => console.log("Orchestrator listening on :3000"));
}
main().catch(console.error);