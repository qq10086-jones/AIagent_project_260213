#!/usr/bin/env node
/**
 * inject_live_traffic.js
 *
 * Controlled traffic injector for accelerated M6 validation.
 *
 * Default mode is tuned for a 30-minute compressed evidence window:
 * - 90 requests
 * - 20s spacing
 * - coding_team_v0 / crm-like prompts
 *
 * Usage:
 *   node scripts/inject_live_traffic.js
 *   node scripts/inject_live_traffic.js --count 30 --interval-ms 5000
 *   node scripts/inject_live_traffic.js --duration-min 30 --interval-ms 10000
 *   node scripts/inject_live_traffic.js --endpoint /vnext/dispatch --approval-safe
 */

import http from "http";

const TASK_TEMPLATES = [
  "Build a CRM customer list page with edit form and API integration",
  "Implement a CRM dashboard with frontend UI and backend endpoint wiring",
  "Refactor a webapp CRM workflow to improve frontend and backend coordination",
  "Create a CRM reporting screen with data table, filters, and service integration",
  "Add a customer detail panel with form validation and backend persistence",
];

const WORKFLOW_HEAVY_TEMPLATES = [
  "Design and implement an end-to-end CRM workflow across PM spec, architecture, backend API, frontend UI, QA verification, and release packaging",
  "Plan and deliver a full CRM feature spanning architecture decisions, backend service changes, frontend implementation, QA flow, and release artifacts",
  "Build a complete CRM module as a coordinated multi-step project covering specification, architecture, backend, frontend, QA, and release readiness",
];

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const cur = argv[i];
    if (!cur.startsWith("--")) continue;
    const key = cur.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      out[key] = true;
    } else {
      out[key] = next;
      i += 1;
    }
  }
  return out;
}

function toInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function buildConfig(args) {
  const durationMin = toInt(args["duration-min"], 30);
  const intervalMs = toInt(args["interval-ms"], 20000);
  const explicitCount = toInt(args.count, 0);
  const totalRequests = explicitCount || Math.max(1, Math.floor((durationMin * 60 * 1000) / intervalMs));
  return {
    hostname: String(args.host || "localhost"),
    port: toInt(args.port, 3000),
    endpoint: String(args.endpoint || "/chat"),
    intervalMs,
    totalRequests,
    approvalSafe: Boolean(args["approval-safe"]),
    workflowHeavy: Boolean(args["workflow-heavy"]),
    timeoutMs: toInt(args["timeout-ms"], 30000),
  };
}

function buildPrompt(index, approvalSafe, workflowHeavy) {
  const templatePool = workflowHeavy ? WORKFLOW_HEAVY_TEMPLATES : TASK_TEMPLATES;
  const base = templatePool[index % templatePool.length];
  const tag = approvalSafe ? "[CompressedValidation][Safe]" : "[CompressedValidation]";
  return `${base} ${tag} seq=${index + 1}`;
}

function buildRequestBody(index, config) {
  const prompt = buildPrompt(index, config.approvalSafe, config.workflowHeavy);
  if (config.endpoint === "/vnext/dispatch") {
    return {
      raw_input: prompt,
      metadata: { validation_window: "30m_compressed" },
    };
  }
  return {
    message: `/coder: ${prompt}`,
    payload: { destructive: false, validation_window: "30m_compressed" },
  };
}

function sendJsonRequest(config, body) {
  return new Promise((resolve) => {
    const payload = JSON.stringify(body);
    const req = http.request(
      {
        hostname: config.hostname,
        port: config.port,
        path: config.endpoint,
        method: "POST",
        timeout: config.timeoutMs,
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => {
          data += chunk;
        });
        res.on("end", () => {
          try {
            resolve({
              ok: res.statusCode >= 200 && res.statusCode < 300,
              statusCode: res.statusCode,
              body: data ? JSON.parse(data) : null,
            });
          } catch {
            resolve({
              ok: false,
              statusCode: res.statusCode,
              body: { raw: data },
            });
          }
        });
      }
    );

    req.on("timeout", () => {
      req.destroy(new Error("request timeout"));
    });

    req.on("error", (err) => {
      resolve({
        ok: false,
        statusCode: 0,
        body: { error: err.message || String(err) },
      });
    });

    req.write(payload);
    req.end();
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const config = buildConfig(parseArgs(process.argv.slice(2)));
  console.log("[inject_live_traffic] starting compressed validation run");
  console.log(`[inject_live_traffic] target=${config.hostname}:${config.port}${config.endpoint}`);
  console.log(`[inject_live_traffic] requests=${config.totalRequests} interval_ms=${config.intervalMs}`);

  let okCount = 0;
  let failCount = 0;
  const startedAt = Date.now();

  for (let i = 0; i < config.totalRequests; i += 1) {
    const body = buildRequestBody(i, config);
    const result = await sendJsonRequest(config, body);
    const runId =
      result.body?.run_id ||
      result.body?.workflow_run_id ||
      result.body?.task_id ||
      "n/a";

    if (result.ok && result.body?.ok !== false) {
      okCount += 1;
      console.log(`[${i + 1}/${config.totalRequests}] ok status=${result.statusCode} ref=${runId}`);
    } else {
      failCount += 1;
      console.log(`[${i + 1}/${config.totalRequests}] fail status=${result.statusCode} ref=${runId}`);
    }

    if (i < config.totalRequests - 1) {
      await sleep(config.intervalMs);
    }
  }

  const elapsedMs = Date.now() - startedAt;
  console.log("[inject_live_traffic] completed");
  console.log(`[inject_live_traffic] ok=${okCount} fail=${failCount} elapsed_ms=${elapsedMs}`);
}

main().catch((err) => {
  console.error("[inject_live_traffic] fatal:", err.message || String(err));
  process.exit(1);
});
