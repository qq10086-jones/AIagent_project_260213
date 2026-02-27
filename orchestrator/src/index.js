import express from "express";
import Redis from "ioredis";
import pg from "pg";
import { v4 as uuidv4 } from "uuid";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { parseIntent, translate, qwenChat, CURRENT_QWEN_MODEL, setQwenModel } from "./nlp/router.js";
import { Client, GatewayIntentBits, EmbedBuilder, AttachmentBuilder } from "discord.js";
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import cron from "node-cron";

const {
  REDIS_URL,
  PGHOST,
  PGPORT,
  PGUSER,
  PGPASSWORD,
  PGDATABASE,
  STREAM_TASK = "stream:task",
  STREAM_RESULT = "stream:result",
  GROUP_TASK = "cg:workers",
  GROUP_RESULT = "cg:orchestrator",
  DISCORD_TOKEN,
  MINIO_ENDPOINT = "http://nexus-minio:9000",
  MINIO_ACCESS_KEY = "nexus",
  MINIO_SECRET_KEY = "nexuspassword",
  AUTO_REPORT_CHANNEL_ID,
  AUTO_REPORT_TIMEZONE = "Asia/Shanghai",
  APPROVAL_TOKEN = "dev-approval-token",
  TOOLS_CONFIG_PATH = "configs/tools.json",
} = process.env;

const QWEN_BASE = process.env.QWEN_BASE_URL || "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";
const QWEN_MODEL = process.env.QWEN_MODEL || "qwen-plus";
const RE_COMPOSITE_CUE = /(?:\u7136\u540e|\u4e26\u4e14|\u540c\u65f6|\u63a5\u7740|\u968f\u540e|\u53e6\u5916|\u4ee5\u53ca|;|\uff1b|\n)/i;
const RE_PRECLOSE = /(?:\u76d8\u5c3e|\u76e4\u5c3e|\u6536\u76d8\u524d|\u6536\u76e4\u524d|preclose)/i;
const RE_POSTCLOSE = /(?:\u76d8\u540e|\u76e4\u5f8c|\u95ea\u8baf|\u9583\u8a0a|tdnet|postclose|post-close)/i;
const RE_NEWS_DAILY = /(?:\u65e5\u62a5|daily report|\u5e02\u573a\u65b0\u95fb|news report)/i;
const RE_DISCOVERY_CUE = /(?:\u5efa\u4ed3|\u5efa\u5009|\u4ed3\u4f4d|\u5009\u4f4d|\u9009\u80a1|\u9078\u80a1|\u6a19\u7684|\u6807\u7684|\u627e.*\u6807\u7684|\u5206\u6279|position plan|portfolio plan|discovery|build[- ]?position|entry plan|staged entry|candidates?|stock picks?|find .*stocks?|find .*candidates?|allocation)/i;
const RE_DISCOVERY_INDEX = /(?:\u5efa\u4ed3|\u5efa\u5009|\u9009\u80a1|\u9078\u80a1|\u6807\u7684|\u6a19\u7684|\u5206\u6279|discovery|position plan|portfolio plan|build[- ]?position|entry plan|staged entry|candidates?|stock picks?|allocation)/i;

function loadToolsConfig() {
  try {
    const resolved = path.resolve(TOOLS_CONFIG_PATH);
    const raw = fs.readFileSync(resolved, "utf-8");
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed ? parsed : {};
  } catch (err) {
    console.warn("[orchestrator] tools.json load failed:", err.message);
    return {};
  }
}

const TOOLS_CONFIG = loadToolsConfig();
const channelMemory = new Map();

function getToolSpec(toolName) {
  return TOOLS_CONFIG?.[toolName] || {};
}

const redis = new Redis(REDIS_URL);
const pool = new pg.Pool({
  host: PGHOST,
  port: Number(PGPORT || 5432),
  user: PGUSER,
  password: PGPASSWORD,
  database: PGDATABASE,
});

const s3 = new S3Client({
  endpoint: MINIO_ENDPOINT,
  credentials: { accessKeyId: MINIO_ACCESS_KEY, secretAccessKey: MINIO_SECRET_KEY },
  region: "us-east-1",
  forcePathStyle: true,
});

const discord = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent] });

discord.on("error", err => console.error("[discord] Client error:", err.message));

const taskToContext = new Map();
const runToContext = new Map();
const DISCORD_MAX_CONTENT = 1900;

function makeIdempotencyKey(run_id, tool_name, payload = {}) {
  const raw = `${run_id}|${tool_name}|${JSON.stringify(payload)}`;
  return crypto.createHash("sha256").update(raw).digest("hex").slice(0, 48);
}

async function recordEvent(task_id, event_type, payload = {}) {
  try {
    await pool.query(
      "INSERT INTO event_log(task_id, event_type, payload_json) VALUES ($1,$2,$3)",
      [task_id, event_type, JSON.stringify(payload || {})]
    );
  } catch (err) {
    console.warn(`[orchestrator] event_log insert failed (${event_type}):`, err.message);
  }
}

function parseOutputField(rawOutput) {
  if (!rawOutput) return null;
  try {
    return JSON.parse(rawOutput);
  } catch {
    return { raw: String(rawOutput) };
  }
}

function splitForDiscord(text, maxLen = DISCORD_MAX_CONTENT) {
  const normalized = String(text || "").replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];
  if (normalized.length <= maxLen) return [normalized];

  const out = [];
  let rest = normalized;
  while (rest.length > maxLen) {
    let cut = rest.lastIndexOf("\n\n", maxLen);
    if (cut < Math.floor(maxLen * 0.5)) cut = rest.lastIndexOf("\n", maxLen);
    if (cut < Math.floor(maxLen * 0.5)) cut = rest.lastIndexOf(" ", maxLen);
    if (cut < Math.floor(maxLen * 0.5)) cut = maxLen;
    out.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trimStart();
  }
  if (rest) out.push(rest);
  return out;
}

async function replyChunked(msg, text, header = "") {
  const merged = header ? `${header}\n${text || ""}` : String(text || "");
  const chunks = splitForDiscord(merged);
  if (chunks.length === 0) return;
  for (const chunk of chunks) {
    await msg.reply(chunk);
  }
}

function markdownToSimpleHtml(markdownText, title = "NEXUS Report") {
  const safe = String(markdownText || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return `<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>${title}</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; line-height: 1.5; color: #222; }
    pre { white-space: pre-wrap; word-break: break-word; background: #f7f7f7; padding: 16px; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>${title}</h1>
  <pre>${safe}</pre>
</body>
</html>`;
}

async function readS3ObjectBuffer(bucket, key) {
  const s3Res = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  const chunks = [];
  for await (const chunk of s3Res.Body || []) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function bindTaskToContext(task_id, context, tool_name) {
  if (!context) return;
  if (!context.pendingTaskIds) context.pendingTaskIds = new Set();
  if (!Number.isFinite(context.totalTaskCount) || context.totalTaskCount <= 0) context.totalTaskCount = 1;
  context.pendingTaskIds.add(task_id);
  runToContext.set(context.run_id, context);
  taskToContext.set(task_id, {
    channelId: context.channelId,
    startTime: context.startTime || Date.now(),
    lang: context.lang || "zh",
    run_id: context.run_id,
    closeRunOnTaskResult: Boolean(context.closeRunOnTaskResult),
    pendingTaskIds: context.pendingTaskIds,
    totalTaskCount: context.totalTaskCount || 1,
    tool_name: tool_name || context.tool_name || "unknown",
  });
}

async function ensureRun(run_id, { client_msg_id, user_id, status, input_text }) {
  await pool.query(
    `INSERT INTO runs(run_id, client_msg_id, user_id, status, input_text)
     VALUES ($1, $2, $3, $4, $5)
     ON CONFLICT (run_id) DO UPDATE
     SET status = EXCLUDED.status,
         input_text = COALESCE(NULLIF(EXCLUDED.input_text, ''), runs.input_text)`,
    [run_id, client_msg_id, user_id, status, input_text]
  );
}

async function callQwenChat(messages) {
  const QWEN_KEY = process.env.QWEN_API_KEY;
  if (!QWEN_KEY) throw new Error("QWEN_API_KEY is not set");
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);
  const response = await fetch(`${QWEN_BASE}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${QWEN_KEY}` },
    body: JSON.stringify({ model: QWEN_MODEL, messages }),
    signal: controller.signal,
  }).finally(() => clearTimeout(timeoutId));

  if (!response.ok) {
    const errText = await response.text().catch(() => "");
    throw new Error(`Qwen API error ${response.status} ${response.statusText} ${errText}`.trim());
  }
  const data = await response.json();
  return data.choices?.[0]?.message?.content?.trim() || "";
}

async function upsertTask(task) {
  await pool.query(
    `INSERT INTO tasks(task_id, tool_name, status, risk_level, payload_json, run_id, idempotency_key, workflow_id, step_index)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
     ON CONFLICT (task_id) DO UPDATE SET status=EXCLUDED.status, updated_at=NOW()`,
    [
      task.task_id,
      task.tool_name,
      task.status,
      task.risk_level || "low",
      JSON.stringify(task.payload),
      task.run_id,
      task.idempotency_key || null,
      task.workflow_id || null,
      Number.isFinite(task.step_index) ? task.step_index : null,
    ]
  );
}

async function enqueueTask({ tool_name, payload, run_id, risk_level = null, idempotency_key, context }) {
  const fullPayload = { ...(payload || {}), run_id };
  const idem = idempotency_key || makeIdempotencyKey(run_id, tool_name, fullPayload);
  const spec = getToolSpec(tool_name);
  const finalRisk = risk_level || spec?.default_risk || "low";
  const requiresApproval = Boolean(spec?.requires_approval);

  const existing = await pool.query("SELECT task_id FROM tasks WHERE idempotency_key=$1 LIMIT 1", [idem]);
  if (existing.rows.length > 0) {
    const task_id = existing.rows[0].task_id;
    bindTaskToContext(task_id, context, tool_name);
    return { task_id, deduplicated: true };
  }

  const task_id = uuidv4();
  await upsertTask({
    task_id,
    tool_name,
    status: requiresApproval ? "waiting_approval" : "queued",
    risk_level: finalRisk,
    payload: fullPayload,
    run_id,
    idempotency_key: idem,
    workflow_id: payload?.workflow_id,
    step_index: payload?.step_index,
  });

  await recordEvent(task_id, "task.created", { tool_name, run_id, risk_level: finalRisk });
  if (requiresApproval) {
    await recordEvent(task_id, "approval.requested", { tool_name, run_id });
  } else {
    await redis.xadd(
      STREAM_TASK,
      "*",
      "task_id",
      task_id,
      "run_id",
      run_id,
      "tool_name",
      tool_name,
      "payload",
      JSON.stringify(fullPayload),
      "workflow_id",
      payload?.workflow_id || "",
      "step_index",
      Number.isFinite(payload?.step_index) ? String(payload.step_index) : ""
    );
  }

  bindTaskToContext(task_id, context, tool_name);
  return { task_id, deduplicated: false, waiting_approval: requiresApproval };
}

async function enqueueWorkflow({ name, steps, run_id, context = null }) {
  const normalizedSteps = Array.isArray(steps) ? steps.filter(s => s && s.tool_name) : [];
  if (normalizedSteps.length === 0) {
    return { ok: false, error: "No valid steps." };
  }

  const workflow_id = uuidv4();
  await pool.query(
    `INSERT INTO workflows(workflow_id, name, definition_json)
     VALUES ($1,$2,$3)`,
    [workflow_id, String(name || "chat-workflow"), JSON.stringify({ steps: normalizedSteps })]
  );

  if (context) {
    context.totalTaskCount = normalizedSteps.length;
    context.completedTaskCount = 0;
  }

  const tasks = [];
  for (let i = 0; i < normalizedSteps.length; i++) {
    const step = normalizedSteps[i];
    const payload = { ...(step.payload || {}), workflow_id, step_index: i };
    const enq = await enqueueTask({
      tool_name: step.tool_name,
      payload,
      run_id,
      risk_level: step.risk_level,
      idempotency_key: makeIdempotencyKey(run_id, step.tool_name, payload),
      context,
    });
    tasks.push({ task_id: enq.task_id, tool_name: step.tool_name, waiting_approval: enq.waiting_approval });
  }
  return { ok: true, workflow_id, run_id, tasks };
}

function hasCompositeCue(text) {
  const s = String(text || "");
  return RE_COMPOSITE_CUE.test(s);
}

function splitCompositeClauses(text) {
  const s = String(text || "")
    .replace(/[；;]/g, "|")
    .replace(/\n+/g, "|")
    .replace(/(?:\u7136\u540e|\u4e26\u4e14|\u540c\u65f6|\u63a5\u7740|\u968f\u540e|\u53e6\u5916|\u4ee5\u53ca)/g, "|");
  return s
    .split("|")
    .map(x => x.trim())
    .filter(x => x.length >= 2);
}

function fallbackRouteClause(clause) {
  const raw = String(clause || "").trim();
  const lower = raw.toLowerCase();
  if (!raw) return null;

  if (RE_PRECLOSE.test(raw) || lower.includes("pre-close")) {
    return { tool_name: "news.preclose_brief_jp", payload: {} };
  }
  if (RE_POSTCLOSE.test(raw)) {
    return { tool_name: "news.tdnet_close_flash", payload: {} };
  }
  if (RE_NEWS_DAILY.test(raw) || (lower.includes("news") && lower.includes("report"))) {
    return { tool_name: "news.daily_report", payload: {} };
  }
  if (hasDiscoveryCue(raw)) {
    return { tool_name: "quant.discovery_workflow", payload: buildDiscoveryPayloadFromText(raw) };
  }
  if (/设置.*资金|设置.*本金|set.*capital|set account/i.test(raw)) {
    const m = raw.match(/([0-9]+(?:\.[0-9]+)?)/);
    const capital = m ? Number(m[1]) : null;
    if (capital) return { tool_name: "portfolio.set_account", payload: { starting_capital: capital, ccy: /usd/i.test(raw) ? "USD" : "JPY" } };
  }
  return null;
}

function hasDiscoveryCue(text) {
  const s = String(text || "");
  return RE_DISCOVERY_CUE.test(s);
}

function parseCapitalJpy(text) {
  const s = String(text || "");
  let m = s.match(/([0-9]+(?:\.[0-9]+)?)\s*(?:w|W|万)/);
  if (m) return Math.round(Number(m[1]) * 10000);
  m = s.match(/([0-9]{2,9}(?:\.[0-9]+)?)\s*(?:日元|円|JPY)/i);
  if (m) return Math.round(Number(m[1]));
  return null;
}

function extractGoalText(text) {
  const s = String(text || "").trim();
  if (!s) return "";
  const m1 = s.match(/(?:\u76ee\u6807)[:\uff1a]?\s*([^\uff0c\u3002\uff1b;\n]+)/);
  if (m1 && m1[1]) return `目标${m1[1].trim()}`;
  const m2 = s.match(/([0-9]{1,2}\s*(?:\u4e2a\u6708|\u500b\u6708|\u6708)[^\uff0c\u3002\uff1b;\n]{0,40}?[0-9]{1,2}(?:\.[0-9]+)?\s*%[^\uff0c\u3002\uff1b;\n]{0,20})/);
  if (m2 && m2[1]) return m2[1].trim();
  return "";
}

function buildDiscoveryPayloadFromText(text) {
  const s = String(text || "").trim();
  const payload = {};

  const capital = parseCapitalJpy(s);
  if (capital && Number.isFinite(capital) && capital > 0) {
    payload.capital_base_jpy = capital;
  }

  if (/(?:\u4f4e\u98ce\u9669|\u4f4e\u98a8\u96aa|\u7a33\u5065|\u7a69\u5065|\u4fdd\u5b88|conservative|low risk)/i.test(s)) {
    payload.risk_profile = "low";
  } else if (/(?:\u9ad8\u98ce\u9669|\u9ad8\u98a8\u96aa|\u6fc0\u8fdb|\u6fc0\u9032|\u8fdb\u53d6|\u9032\u53d6|aggressive|high risk)/i.test(s)) {
    payload.risk_profile = "high";
  } else if (/(?:\u4e2d\u98ce\u9669|\u4e2d\u98a8\u96aa|\u5e73\u8861|balanced|medium risk)/i.test(s)) {
    payload.risk_profile = "medium";
  }

  const jp = /(?:\bJP\b|\u65e5\u80a1|\u65e5\u672c|\u4e1c\u4eac|\u6771\u4eac)/i.test(s);
  const us = /(?:\bUS\b|\u7f8e\u80a1|\u7f8e\u56fd|\u7f8e\u570b)/i.test(s);
  if (jp && us) payload.market = "ALL";
  else if (jp) payload.market = "JP";
  else if (us) payload.market = "US";

  const mMonth = s.match(/([0-9]{1,2})\s*(?:\u4e2a\u6708|\u500b\u6708|\u6708)/);
  if (mMonth) payload.horizon_days = Number(mMonth[1]) * 30;

  const mRet = s.match(/([0-9]{1,2}(?:\.[0-9]+)?)\s*%/);
  if (mRet) payload.target_return_pct = Number(mRet[1]);

  const goalText = extractGoalText(s);
  if (goalText) {
    payload.goal = goalText;
  } else if (/(?:\u76ee\u6807|\u589e\u503c|\u56de\u62a5|\u6536\u76ca|\u5efa\u4ed3\u8ba1\u5212|\u5efa\u5009\u8a08\u5283|\u5206\u6279)/.test(s)) {
    payload.goal = s.slice(0, 120);
  }
  return payload;
}

function extractRuleBasedStepsFromText(text) {
  const s = String(text || "");
  const out = [];
  const add = (idx, tool_name, payload = {}) => {
    if (idx < 0) return;
    out.push({ idx, tool_name, payload });
  };

  const idxPre = s.search(RE_PRECLOSE);
  add(idxPre, "news.preclose_brief_jp", {});

  const idxPost = s.search(RE_POSTCLOSE);
  add(idxPost, "news.tdnet_close_flash", {});

  const idxDaily = s.search(RE_NEWS_DAILY);
  add(idxDaily, "news.daily_report", {});

  const idxDisc = hasDiscoveryCue(s)
    ? s.search(RE_DISCOVERY_INDEX)
    : -1;
  if (idxDisc >= 0) {
    add(idxDisc, "quant.discovery_workflow", buildDiscoveryPayloadFromText(s));
  }

  out.sort((a, b) => a.idx - b.idx);
  return out.map(({ tool_name, payload }) => ({ tool_name, payload }));
}

async function planCompositeWorkflowFromText(userInput, memory = {}) {
  const ruleSteps = extractRuleBasedStepsFromText(userInput);
  // If rule-based detector already finds multiple tasks, use it directly for stability.
  if (ruleSteps.length >= 2) {
    return {
      name: `chat-composite-${Date.now()}`,
      steps: ruleSteps,
    };
  }

  if (!hasCompositeCue(userInput)) return null;
  const clauses = splitCompositeClauses(userInput);
  if (clauses.length < 2) return null;

  const steps = [];
  const seen = new Set();
  const pushStep = (step) => {
    if (!step?.tool_name) return;
    const key = `${step.tool_name}|${JSON.stringify(step.payload || {})}`;
    if (seen.has(key)) return;
    seen.add(key);
    steps.push({ tool_name: step.tool_name, payload: step.payload || {} });
  };
  const localMemory = { ...(memory || {}) };
  for (const clause of clauses) {
    let intent = null;
    try {
      intent = await parseIntent(clause, localMemory);
    } catch {
      intent = null;
    }
    if (intent?.payload?.symbol) localMemory.last_symbol = intent.payload.symbol;
    if (intent?.requires_tools && intent?.tool_name && intent?.confidence >= 0.55) {
      pushStep({
        tool_name: intent.tool_name,
        payload: intent.payload || {},
      });
      continue;
    }

    const fallback = fallbackRouteClause(clause);
    if (fallback?.tool_name) {
      pushStep({
        tool_name: fallback.tool_name,
        payload: fallback.payload || {},
      });
    }
  }

  for (const step of ruleSteps) {
    pushStep(step);
  }

  if (steps.length < 2) return null;
  return {
    name: `chat-composite-${Date.now()}`,
    steps,
  };
}

function detectLanguageQuick(text) {
  const s = String(text || "");
  if (/[\u4e00-\u9fff]/.test(s)) return "zh";
  if (/[\u3040-\u30ff]/.test(s)) return "ja";
  return "en";
}

function summarizeOutputBrief(output) {
  if (!output || typeof output !== "object") return "Done";
  const raw = output.analysis || output.summary || output.message || output.stdout || output.raw || "Done";
  const oneLine = String(raw).replace(/\s+/g, " ").trim();
  return oneLine.slice(0, 120);
}

async function callBrainWithRetry(payload, retries = 2) {
  for (let i = 0; i <= retries; i++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 240000);

      const res = await fetch("http://brain:5000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!res.ok) throw new Error(`Brain returned ${res.status}`);
      return await res.json();
    } catch (err) {
      if (i === retries) throw err;
      console.log(`[orchestrator] brain call failed, retrying (${i + 1}/${retries})...`);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}

cron.schedule(
  "0 16 * * *",
  async () => {
    if (!AUTO_REPORT_CHANNEL_ID) return;

    const dailyDate = new Date().toISOString().slice(0, 10);
    const dailyTasks = [
      { tool: "news.daily_report", payload: { max_items: 20, date: dailyDate } },
      { tool: "quant.run_optimized_pipeline", payload: { date: dailyDate } },
    ];

    for (const t of dailyTasks) {
      const run_id = uuidv4();
      const context = {
        run_id,
        channelId: AUTO_REPORT_CHANNEL_ID,
        startTime: Date.now(),
        lang: "zh",
        closeRunOnTaskResult: true,
      };
      runToContext.set(run_id, context);

      await ensureRun(run_id, {
        client_msg_id: `cron-${dailyDate}-${t.tool}-${run_id.slice(0, 8)}`,
        user_id: "system-cron",
        status: "running",
        input_text: `daily:${t.tool}`,
      });

      await enqueueTask({
        tool_name: t.tool,
        payload: t.payload,
        run_id,
        idempotency_key: `${dailyDate}:${t.tool}`,
        context,
      });
    }
  },
  { timezone: AUTO_REPORT_TIMEZONE }
);

// JP Market Pre-Close Brief at 15:15 JST (Mon-Fri)
cron.schedule(
  "15 15 * * 1-5",
  async () => {
    if (!AUTO_REPORT_CHANNEL_ID) return;
    const dailyDate = new Date().toISOString().slice(0, 10);
    const tool = "news.preclose_brief_jp";
    
    const run_id = uuidv4();
    const context = {
      run_id,
      channelId: AUTO_REPORT_CHANNEL_ID,
      startTime: Date.now(),
      lang: "zh",
      closeRunOnTaskResult: true,
    };
    runToContext.set(run_id, context);

    await ensureRun(run_id, {
      client_msg_id: `cron-${dailyDate}-${tool}-${run_id.slice(0, 8)}`,
      user_id: "system-cron",
      status: "running",
      input_text: `cron:${tool}`,
    });

    await enqueueTask({
      tool_name: tool,
      payload: { date: dailyDate, type: "preclose" },
      run_id,
      idempotency_key: `${dailyDate}:${tool}`,
      context,
    });
  },
  { timezone: "Asia/Tokyo" }
);

// JP Market TDnet Close Flash at 15:35 JST (Mon-Fri)
cron.schedule(
  "35 15 * * 1-5",
  async () => {
    if (!AUTO_REPORT_CHANNEL_ID) return;
    const dailyDate = new Date().toISOString().slice(0, 10);
    const tool = "news.tdnet_close_flash";
    
    const run_id = uuidv4();
    const context = {
      run_id,
      channelId: AUTO_REPORT_CHANNEL_ID,
      startTime: Date.now(),
      lang: "zh",
      closeRunOnTaskResult: true,
    };
    runToContext.set(run_id, context);

    await ensureRun(run_id, {
      client_msg_id: `cron-${dailyDate}-${tool}-${run_id.slice(0, 8)}`,
      user_id: "system-cron",
      status: "running",
      input_text: `cron:${tool}`,
    });

    await enqueueTask({
      tool_name: tool,
      payload: { date: dailyDate, type: "postclose_flash" },
      run_id,
      idempotency_key: `${dailyDate}:${tool}`,
      context,
    });
  },
  { timezone: "Asia/Tokyo" }
);

discord.on("clientReady", () => console.log(`[discord] Logged in as ${discord.user.tag}`));

discord.on("messageCreate", async msg => {
  if (msg.author.bot) return;
  const rawInput = msg.content || "";

  if (rawInput.trim() === "/model" || rawInput.trim().startsWith("/model:") || rawInput.trim().startsWith("/model ")) {
    const parts = rawInput.trim().split(/[: ]/);
    const newModel = parts.length > 1 ? parts.slice(1).join(" ").trim() : "";
    
    if (newModel) {
      setQwenModel(newModel);
      await msg.reply(`[NEXUS] 模型已成功切换为: **${newModel}**`);
    } else {
      await msg.reply(`[NEXUS] 当前模型是: **${CURRENT_QWEN_MODEL}**`);
    }
    return;
  }

  const userInput = rawInput.replace(/@api\b/gi, "").replace(/@32b\b/gi, "").trim();
  if (!userInput) return;

  const client_msg_id = msg.id;
  const run_id = uuidv4();

  try {
    const existingRun = await pool.query("SELECT run_id FROM runs WHERE client_msg_id = $1", [client_msg_id]);
    if (existingRun.rows.length > 0) return;

    await msg.channel.sendTyping();
    await ensureRun(run_id, {
      client_msg_id,
      user_id: msg.author.id,
      status: "starting",
      input_text: userInput,
    });

    const memory = channelMemory.get(msg.channelId) || {};
    const compositePlan = await planCompositeWorkflowFromText(userInput, memory);
    if (compositePlan) {
      const lang = detectLanguageQuick(userInput);
      const context = {
        channelId: msg.channel.id,
        startTime: Date.now(),
        lang,
        run_id,
        closeRunOnTaskResult: true,
        totalTaskCount: compositePlan.steps.length,
        completedTaskCount: 0,
      };
      runToContext.set(run_id, context);
      await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["running", run_id]);

      const planningText = await translate(
        `识别到复合指令，已拆分为 ${compositePlan.steps.length} 个子任务，开始执行。`,
        lang
      );
      await msg.reply(`[NEXUS] ${planningText}`);

      const wf = await enqueueWorkflow({
        name: compositePlan.name,
        steps: compositePlan.steps,
        run_id,
        context,
      });
      if (!wf?.ok) {
        throw new Error(wf?.error || "Failed to enqueue workflow.");
      }

      const stepNames = compositePlan.steps.map((s, i) => `${i + 1}.${s.tool_name}`).join(" | ");
      const enqText = await translate(`工作流已创建。run_id=${run_id}。步骤: ${stepNames}`, lang);
      await msg.reply(`[NEXUS] ${enqText}`);
      return;
    }

    const intent = await parseIntent(userInput, memory);
    const lang = intent.language || "zh";

    // Update memory if a symbol was found
    if (intent.payload && intent.payload.symbol) {
      memory.last_symbol = intent.payload.symbol;
      channelMemory.set(msg.channelId, memory);
    }

    let model_preference = "local_small";
    if (rawInput.includes("@api")) model_preference = "api";
    if (rawInput.includes("@32b")) model_preference = "local_large";

    // 1. CHAT MODE: If intent analyzer suggests chat or no tools are required
    if (intent.mode_suggested === "chat" || !intent.requires_tools || intent.confidence < 0.6 || !intent.tool_name) {
      if (process.env.QWEN_API_KEY || model_preference === "api") {
        const reply = await callQwenChat([{ role: "user", content: userInput }]);
        await replyChunked(msg, reply || "I didn't understand that.");
      } else {
        const chatRes = await fetch(`${process.env.OLLAMA_BASE_URL || "http://host.docker.internal:11434"}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: process.env.QUANT_LLM_MODEL || "deepseek-r1:1.5b",
            messages: [{ role: "user", content: userInput }],
            stream: false,
          }),
        });
        const chatData = await chatRes.json();
        await replyChunked(msg, chatData.message?.content || "I didn't understand that.");
      }
      return;
    }

    // 2. RUN MODE: Tools are required
    const isBrainControlled = intent.tool_name === "quant.deep_analysis" || intent.tool_name === "quant.discovery_workflow";
    const mode = intent.tool_name === "quant.discovery_workflow" ? "discovery" : "analysis";
    
    const context = {
      channelId: msg.channel.id,
      startTime: Date.now(),
      lang,
      run_id,
      closeRunOnTaskResult: !isBrainControlled,
    };
    runToContext.set(run_id, context);
    await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["running", run_id]);

    if (isBrainControlled) {
      const progressMap = {
        "discovery": "正在为您搜寻全球金融情报并由专业模型进行量化筛选，请稍候...",
        "analysis": "已开始查找并生成深度分析，请稍候..."
      };
      const progressText = await translate(progressMap[mode], lang);
      await msg.reply(`[NEXUS] ${progressText}`);

      const brainData = await callBrainWithRetry({
        symbol: intent.payload.symbol || "unknown",
        run_id,
        model_preference,
        mode: mode,
        tool_name: intent.tool_name,
        tool_payload: intent.payload || {},
        qwen_model: CURRENT_QWEN_MODEL
      });

      const report = (brainData?.narrative || "").trim();
      const reportMarkdown = (brainData?.report_markdown || "").trim();
      const reportHtmlObjectKey = brainData?.report_html_object_key || "";
      await pool.query("UPDATE runs SET status=$1, cost_ledger_json=$2 WHERE run_id=$3", [
        "completed",
        JSON.stringify(brainData?.cost_ledger || {}),
        run_id,
      ]);
      runToContext.delete(run_id);

      const needAttachment = report.length > DISCORD_MAX_CONTENT * 2 || reportMarkdown.length > DISCORD_MAX_CONTENT * 2;
      if (report || reportMarkdown) {
        if (!needAttachment) {
          await replyChunked(msg, report || reportMarkdown, "[NEXUS Report]");
        } else {
          const preview = (report || reportMarkdown).slice(0, DISCORD_MAX_CONTENT - 120);
          if (preview) await replyChunked(msg, preview, "[NEXUS Report Preview]");

          const files = [];
          if (reportHtmlObjectKey) {
            try {
              const htmlBuffer = await readS3ObjectBuffer("nexus-artifacts", reportHtmlObjectKey);
              files.push(new AttachmentBuilder(htmlBuffer, { name: `nexus_report_${run_id.slice(0, 8)}.html` }));
            } catch (err) {
              console.error("S3 report download error:", err);
            }
          }
          if (files.length === 0 && reportMarkdown) {
            const htmlFallback = markdownToSimpleHtml(reportMarkdown, `NEXUS Report ${run_id.slice(0, 8)}`);
            files.push(new AttachmentBuilder(Buffer.from(htmlFallback, "utf-8"), { name: `nexus_report_${run_id.slice(0, 8)}.html` }));
            files.push(new AttachmentBuilder(Buffer.from(reportMarkdown, "utf-8"), { name: `nexus_report_${run_id.slice(0, 8)}.md` }));
          }
          if (files.length > 0) {
            await msg.reply({ content: "[NEXUS] 完整报告见附件（HTML/Markdown）。", files });
          } else {
            await replyChunked(msg, report || reportMarkdown, "[NEXUS Report]");
          }
        }
      } else {
        const fallback = await translate("任务完成，但未生成正文报告。", lang);
        await replyChunked(msg, `[NEXUS] ${fallback}`);
      }

      const elapsedSec = ((Date.now() - context.startTime) / 1000).toFixed(1);
      const doneRaw = `任务已完成。run_id=${run_id}，耗时=${elapsedSec}s`;
      const doneText = await translate(doneRaw, lang);
      await msg.reply(`[NEXUS] ${doneText}`);
    } else {
      const actionMap = {
        "news.daily_report": "正在生成全市场新闻日报，请稍候...",
        "news.preclose_brief_jp": "正在获取日本市场盘尾情报简报，请稍候...",
        "news.tdnet_close_flash": "正在扫描TDnet盘后公告闪讯，请稍候...",
        "github.skills_daily_report": "正在为您扫描最新AI智能体技能，请稍候...",
        "portfolio.set_account": "正在为您设置资金账户参数，请稍候...",
        "portfolio.record_fill": "正在记录您的成交数据并更新持仓，请稍候..."
      };
      const defaultMsg = `已识别指令 [${intent.tool_name}]，正在分配给对应Agent...`;
      const progressText = await translate(actionMap[intent.tool_name] || defaultMsg, lang);
      await msg.reply(`[NEXUS] ${progressText}`);
      
      await enqueueTask({
        tool_name: intent.tool_name,
        payload: intent.payload || {},
        run_id,
        idempotency_key: makeIdempotencyKey(run_id, intent.tool_name, intent.payload || {}),
        context,
      });
    }

  } catch (e) {
    console.error("[orchestrator] Error:", e);
    await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["failed", run_id]).catch(() => {});
    runToContext.delete(run_id);
    await replyChunked(msg, `Error: ${e.message}`);
  }
});

if (DISCORD_TOKEN) discord.login(DISCORD_TOKEN).catch(console.error);

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
          const output = parseOutputField(obj.output);

          if (status === "claimed") {
            await pool.query("UPDATE tasks SET status=$1, updated_at=NOW() WHERE task_id=$2", ["running", task_id]);
            await recordEvent(task_id, "task.claimed", { task_id });
          } else {
            await pool.query("UPDATE tasks SET status=$1, updated_at=NOW() WHERE task_id=$2", [status, task_id]);
            if (status === "succeeded") {
              await recordEvent(task_id, "task.succeeded", { task_id });
            } else if (status === "failed") {
              await recordEvent(task_id, "task.failed", { task_id });
            }

            const ctx = taskToContext.get(task_id);
            if (ctx) {
              const channel = await discord.channels.fetch(ctx.channelId).catch(() => null);
              if (!Array.isArray(ctx.resultItems)) ctx.resultItems = [];
              if (channel && typeof channel.send === "function") {
                const duration = ((Date.now() - ctx.startTime) / 1000).toFixed(1);
                const lang = ctx.lang || "zh";
                const title = await translate(status === "succeeded" ? "Task Completed" : "Task Failed", lang);
                const embed = new EmbedBuilder()
                  .setTitle(title)
                  .setColor(status === "succeeded" ? 0x00ff00 : 0xff0000)
                  .setDescription(`**Tool:** ${ctx.tool_name}\n**Duration:** ${duration}s`)
                  .setTimestamp();

                if (output) {
                  const summaryRaw = output.analysis || output.summary || output.stdout || output.raw || "Done";
                  const summary = await translate(String(summaryRaw), lang);
                  embed.addFields({ name: "Result", value: summary.slice(0, 1024) });
                }

                const attachments = [];
                if (status === "succeeded" && output && Array.isArray(output.artifacts)) {
                  for (const art of output.artifacts) {
                    const isSupported = (typeof art?.mime === "string") && (
                      art.mime.startsWith("image/") || 
                      art.mime === "text/html" || 
                      art.mime === "text/markdown"
                    );
                    if (isSupported && art.object_key) {
                      try {
                        const s3Res = await s3.send(new GetObjectCommand({ Bucket: "nexus-artifacts", Key: art.object_key }));
                        const chunks = [];
                        // Handle both ReadableStream and AsyncIterable
                        const body = s3Res.Body;
                        if (body) {
                          for await (const chunk of body) chunks.push(chunk);
                          const buffer = Buffer.concat(chunks);
                          attachments.push(new AttachmentBuilder(buffer, { name: art.name || "artifact" }));
                        }
                      } catch (err) {
                        console.error("S3 Download Error:", err);
                      }
                    }
                  }
                }

                await channel.send({ embeds: [embed], files: attachments });
              }

              ctx.resultItems.push({
                tool: ctx.tool_name,
                status,
                summary: summarizeOutputBrief(output),
              });

              taskToContext.delete(task_id);
              if (ctx.pendingTaskIds) {
                ctx.pendingTaskIds.delete(task_id);
                const total = Number(ctx.totalTaskCount || 1);
                const done = Math.max(0, total - ctx.pendingTaskIds.size);
                if (total > 1 && channel && typeof channel.send === "function") {
                  const progressRaw = `任务进度：${done}/${total}（run_id=${ctx.run_id}）`;
                  const progressText = await translate(progressRaw, ctx.lang || "zh");
                  await channel.send(`[NEXUS] ${progressText}`);
                }
                if (ctx.pendingTaskIds.size === 0 && ctx.closeRunOnTaskResult) {
                  await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", [status === "succeeded" ? "completed" : "failed", ctx.run_id]).catch(() => {});
                  if (channel && typeof channel.send === "function" && total > 1) {
                    const lines = (ctx.resultItems || []).map((it, idx) => {
                      const okMark = it.status === "succeeded" ? "OK" : "FAIL";
                      return `${idx + 1}. [${okMark}] ${it.tool}: ${it.summary}`;
                    });
                    const summaryBlock = lines.length ? lines.join("\n") : "No task details.";
                    await channel.send(`[NEXUS] 任务总览（run_id=${ctx.run_id}）\n${summaryBlock}`);
                  }
                  const doneRaw = status === "succeeded"
                    ? `本轮任务已全部完成。run_id=${ctx.run_id}`
                    : `本轮任务已结束，但存在失败任务。run_id=${ctx.run_id}`;
                  const doneText = await translate(doneRaw, ctx.lang || "zh");
                  if (channel && typeof channel.send === "function") {
                    await channel.send(`[NEXUS] ${doneText}`);
                  }
                  runToContext.delete(ctx.run_id);
                }
              }
            } else {
              const runRes = await pool.query("SELECT run_id FROM tasks WHERE task_id=$1", [task_id]);
              const run_id = runRes.rows[0]?.run_id;
              if (run_id) {
                const pendingRes = await pool.query(
                  "SELECT COUNT(1)::int AS c FROM tasks WHERE run_id=$1 AND status IN ('queued','running','waiting_approval')",
                  [run_id]
                );
                if ((pendingRes.rows[0]?.c || 0) === 0) {
                  await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", [status === "succeeded" ? "completed" : "failed", run_id]).catch(() => {});
                }
              }
            }
          }

          await redis.xack(STREAM_RESULT, GROUP_RESULT, id);
        }
      }
    } catch {
      await new Promise(r => setTimeout(r, 1000));
    }
  }
}

const app = express();
app.use(express.json());
app.get("/health", (_, res) => res.send("ok"));

app.post("/debug/plan", async (req, res) => {
  try {
    const message = String(req.body?.message || "");
    const composite = await planCompositeWorkflowFromText(message, {});
    return res.json({
      ok: true,
      hasCompositeCue: hasCompositeCue(message),
      ruleSteps: extractRuleBasedStepsFromText(message),
      composite,
    });
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message || "debug plan failed" });
  }
});

app.post("/execute-tool", async (req, res) => {
  const { tool_name, payload, run_id } = req.body;
  if (!tool_name || !run_id) {
    return res.status(400).json({ ok: false, error: "tool_name and run_id are required" });
  }

  const context = runToContext.get(run_id);
  const inputText = payload?.symbol ? `tool:${tool_name}:${payload.symbol}` : `tool:${tool_name}`;

  await ensureRun(run_id, {
    client_msg_id: `tool-${run_id}`,
    user_id: "brain-agent",
    status: "running",
    input_text: inputText,
  });

  const queued = await enqueueTask({
    tool_name,
    payload,
    run_id,
    idempotency_key: req.body.idempotency_key || makeIdempotencyKey(run_id, tool_name, payload || {}),
    context,
  });

  console.log(`[orchestrator] queued tool ${tool_name} for run ${run_id} task ${queued.task_id}`);
  return res.json({ ok: true, task_id: queued.task_id, deduplicated: queued.deduplicated });
});

app.post("/tasks/:task_id/approve", async (req, res) => {
  const token = req.header("X-Approval-Token") || "";
  if (token !== APPROVAL_TOKEN) {
    return res.status(403).json({ ok: false, error: "invalid approval token" });
  }

  const task_id = req.params.task_id;
  const row = await pool.query(
    "SELECT task_id, tool_name, payload_json, run_id, status, workflow_id, step_index FROM tasks WHERE task_id=$1",
    [task_id]
  );
  if (row.rows.length === 0) {
    return res.status(404).json({ ok: false, error: "task not found" });
  }
  const task = row.rows[0];
  if (task.status !== "waiting_approval") {
    return res.status(409).json({ ok: false, error: `task status is ${task.status}` });
  }

  await pool.query("UPDATE tasks SET status=$1, updated_at=NOW() WHERE task_id=$2", ["queued", task_id]);
  await recordEvent(task_id, "approval.approved", { task_id });

  let payload = {};
  try {
    payload = JSON.parse(task.payload_json || "{}");
  } catch {
    payload = {};
  }

  await redis.xadd(
    STREAM_TASK,
    "*",
    "task_id",
    task_id,
    "run_id",
    task.run_id || "",
    "tool_name",
    task.tool_name,
    "payload",
    JSON.stringify(payload),
    "workflow_id",
    task.workflow_id || "",
    "step_index",
    Number.isFinite(task.step_index) ? String(task.step_index) : ""
  );

  return res.json({ ok: true, task_id });
});

app.post("/workflows", async (req, res) => {
  const { name, definition } = req.body || {};
  const steps = definition?.steps;
  if (!name || !Array.isArray(steps) || steps.length === 0) {
    return res.status(400).json({ ok: false, error: "name and definition.steps are required" });
  }

  const run_id = uuidv4();

  await ensureRun(run_id, {
    client_msg_id: `workflow-${run_id}`,
    user_id: "workflow",
    status: "running",
    input_text: `workflow:${name}`,
  });

  try {
    const wf = await enqueueWorkflow({
      name: String(name),
      steps,
      run_id,
      context: null,
    });
    if (!wf?.ok) {
      return res.status(500).json({ ok: false, error: wf?.error || "workflow enqueue failed" });
    }
    return res.json({ ok: true, workflow_id: wf.workflow_id, run_id, tasks: wf.tasks });
  } catch (err) {
    await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["failed", run_id]).catch(() => {});
    return res.status(500).json({ ok: false, error: err.message || "workflow enqueue error" });
  }
});

app.post("/chat", async (req, res) => {
  const { message } = req.body;
  const run_id = uuidv4();

  try {
    await ensureRun(run_id, {
      client_msg_id: `api-${run_id}`,
      user_id: "api-user",
      status: "starting",
      input_text: message || "",
    });
  } catch (e) {
    return res.status(500).json({ ok: false, error: `Failed to initialize run: ${e.message}` });
  }

  const compositePlan = await planCompositeWorkflowFromText(message || "", {});

  if (compositePlan) {
    try {
      await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["running", run_id]);
      const wf = await enqueueWorkflow({
        name: compositePlan.name,
        steps: compositePlan.steps,
        run_id,
        context: null,
      });
      if (!wf?.ok) throw new Error(wf?.error || "workflow enqueue failed");
      return res.json({
        ok: true,
        mode: "workflow",
        workflow_id: wf.workflow_id,
        run_id,
        tasks: wf.tasks,
      });
    } catch (e) {
      await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["failed", run_id]).catch(() => {});
      return res.status(500).json({ ok: false, error: e.message });
    }
  }

  const intent = await parseIntent(message || "");

  if (intent.confidence > 0.6 && intent.tool_name) {
    try {
      await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["running", run_id]);
      const brainData = await callBrainWithRetry({
        symbol: intent.payload.symbol || "unknown",
        run_id,
        mode: intent.tool_name === "quant.discovery_workflow" ? "discovery" : "analysis",
        tool_name: intent.tool_name,
        tool_payload: intent.payload || {},
        model_preference: "local_small",
        qwen_model: CURRENT_QWEN_MODEL
      });
      await pool.query("UPDATE runs SET status=$1, cost_ledger_json=$2 WHERE run_id=$3", [
        "completed",
        JSON.stringify(brainData?.cost_ledger || {}),
        run_id,
      ]);
      return res.json({
        ok: true,
        narrative: brainData.narrative,
        report_markdown: brainData.report_markdown || "",
        report_html_object_key: brainData.report_html_object_key || "",
        run_id,
      });
    } catch (e) {
      await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["failed", run_id]).catch(() => {});
      return res.status(500).json({ ok: false, error: e.message });
    }
  }

  await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["completed", run_id]).catch(() => {});
  return res.json({ ok: false, run_id });
});

async function main() {
  try {
    await pool.query("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS workflow_id TEXT");
    await pool.query("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS step_index INT");
  } catch (err) {
    console.warn("[orchestrator] schema ensure failed:", err.message);
  }
  try {
    await redis.xgroup("CREATE", STREAM_TASK, GROUP_TASK, "$", "MKSTREAM");
  } catch {}

  try {
    await redis.xgroup("CREATE", STREAM_RESULT, GROUP_RESULT, "$", "MKSTREAM");
  } catch {}

  startResultConsumer();
  app.listen(3000, () => console.log("Orchestrator listening on :3000"));
}

main().catch(console.error);
