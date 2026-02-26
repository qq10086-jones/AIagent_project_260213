import express from "express";
import Redis from "ioredis";
import pg from "pg";
import { v4 as uuidv4 } from "uuid";
import crypto from "crypto";
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
} = process.env;

const QWEN_BASE = process.env.QWEN_BASE_URL || "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";
const QWEN_MODEL = process.env.QWEN_MODEL || "qwen-plus";

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
  context.pendingTaskIds.add(task_id);
  runToContext.set(context.run_id, context);
  taskToContext.set(task_id, {
    channelId: context.channelId,
    startTime: context.startTime || Date.now(),
    lang: context.lang || "zh",
    run_id: context.run_id,
    closeRunOnTaskResult: Boolean(context.closeRunOnTaskResult),
    pendingTaskIds: context.pendingTaskIds,
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
    `INSERT INTO tasks(task_id, tool_name, status, risk_level, payload_json, run_id, idempotency_key)
     VALUES ($1,$2,$3,$4,$5,$6,$7)
     ON CONFLICT (task_id) DO UPDATE SET status=EXCLUDED.status, updated_at=NOW()`,
    [
      task.task_id,
      task.tool_name,
      task.status,
      task.risk_level || "low",
      JSON.stringify(task.payload),
      task.run_id,
      task.idempotency_key || null,
    ]
  );
}

async function enqueueTask({ tool_name, payload, run_id, risk_level = "low", idempotency_key, context }) {
  const fullPayload = { ...(payload || {}), run_id };
  const idem = idempotency_key || makeIdempotencyKey(run_id, tool_name, fullPayload);

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
    status: "queued",
    risk_level,
    payload: fullPayload,
    run_id,
    idempotency_key: idem,
  });

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
    JSON.stringify(fullPayload)
  );

  bindTaskToContext(task_id, context, tool_name);
  return { task_id, deduplicated: false };
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

    const intent = await parseIntent(userInput);
    const lang = intent.language || "zh";

    let model_preference = "local_small";
    if (rawInput.includes("@api")) model_preference = "api";
    if (rawInput.includes("@32b")) model_preference = "local_large";

    // 1. CHAT MODE: If intent analyzer suggests chat or no tools are required
    if (intent.mode_suggested === "chat" || !intent.requires_tools || intent.confidence < 0.6 || !intent.tool_name) {
      if (model_preference === "api" && process.env.QWEN_API_KEY) {
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
          } else {
            await pool.query("UPDATE tasks SET status=$1, updated_at=NOW() WHERE task_id=$2", [status, task_id]);

            const ctx = taskToContext.get(task_id);
            if (ctx) {
              const channel = await discord.channels.fetch(ctx.channelId).catch(() => null);
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

              taskToContext.delete(task_id);
              if (ctx.pendingTaskIds) {
                ctx.pendingTaskIds.delete(task_id);
                if (ctx.pendingTaskIds.size === 0 && ctx.closeRunOnTaskResult) {
                  await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", [status === "succeeded" ? "completed" : "failed", ctx.run_id]).catch(() => {});
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

  const intent = await parseIntent(message || "");

  if (intent.confidence > 0.6 && intent.tool_name) {
    try {
      await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", ["running", run_id]);
      const brainData = await callBrainWithRetry({
        symbol: intent.payload.symbol || "unknown",
        run_id,
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
    await redis.xgroup("CREATE", STREAM_TASK, GROUP_TASK, "$", "MKSTREAM");
  } catch {}

  try {
    await redis.xgroup("CREATE", STREAM_RESULT, GROUP_RESULT, "$", "MKSTREAM");
  } catch {}

  startResultConsumer();
  app.listen(3000, () => console.log("Orchestrator listening on :3000"));
}

main().catch(console.error);
