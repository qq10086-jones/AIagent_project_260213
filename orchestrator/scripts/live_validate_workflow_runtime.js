import fs from "fs";
import path from "path";
import {
  resolveCanaryInputPath,
  resolveOrchestratorArtifactPath,
} from "./_paths.js";

function arg(name, fallback = "") {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return String(process.argv[idx + 1]);
  return fallback;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function getJson(url, init = {}) {
  const res = await fetch(url, init);
  const text = await res.text();
  let json;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = { raw: text };
  }
  return { ok: res.ok, status: res.status, json };
}

async function checkHealth(baseUrl) {
  const res = await fetch(`${baseUrl}/health`);
  const text = await res.text();
  assert(res.ok && String(text).trim() === "ok", `health failed: ${res.status} ${text}`);
}

async function pollWorkflow(baseUrl, workflowRunId, timeoutMs = 120000, intervalMs = 4000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const state = await getJson(`${baseUrl}/workflow-runs/${encodeURIComponent(workflowRunId)}`);
    if (state.ok) {
      const status = String(state.json?.run?.status || "");
      if (status === "succeeded" || status === "failed") {
        return state.json;
      }
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting workflow ${workflowRunId}`);
}

async function main() {
  const baseUrl = String(arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000")).replace(/\/+$/, "");
  const inputPath = resolveCanaryInputPath(arg("input", "crm_mini.json"));
  // Full coding_team_v0 live runs can exceed 5 minutes when all delegate steps execute end-to-end.
  const timeoutMs = Math.max(60000, Number(arg("timeout-ms", "420000")));

  const report = {
    generated_at: new Date().toISOString(),
    base_url: baseUrl,
    input_path: inputPath,
    timeout_ms: timeoutMs,
    overall: "pass",
  };

  try {
    await checkHealth(baseUrl);
    const payload = loadJson(inputPath);
    const startRes = await getJson(`${baseUrl}/workflow-runs/start`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    assert(startRes.ok, `workflow start failed: ${startRes.status}`);
    report.start = startRes.json;
    const workflowRunId = String(startRes.json?.workflow_run_id || "");
    const runId = String(startRes.json?.run_id || "");
    assert(workflowRunId, "workflow_run_id missing");
    assert(runId, "run_id missing");

    report.terminal = await pollWorkflow(baseUrl, workflowRunId, timeoutMs, 4000);
    assert(String(report.terminal?.run?.status || "") === "succeeded", `workflow ended with status=${String(report.terminal?.run?.status || "")}`);
    report.timeline = (await getJson(`${baseUrl}/runs/${encodeURIComponent(runId)}/timeline`)).json || null;
    report.artifacts = (await getJson(`${baseUrl}/runs/${encodeURIComponent(runId)}/artifacts`)).json || null;
    assert(Array.isArray(report.timeline?.tasks), "timeline tasks missing");
    assert(report.timeline.tasks.length > 0, "timeline tasks empty");
  } catch (err) {
    report.overall = "fail";
    report.error = err.message || String(err);
  }

  const outDir = resolveOrchestratorArtifactPath("canary", "live_workflow_runtime");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "live_workflow_runtime_report.json");
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");

  console.log(`# Live Workflow Runtime Validation`);
  console.log(`- report: ${outPath.replace(/\\/g, "/")}`);
  console.log(`- overall: ${report.overall}`);
  if (report.error) console.log(`- error: ${report.error}`);
  if (report.overall !== "pass") process.exit(1);
}

main().catch((err) => {
  console.error(`[live-workflow-validate] failed: ${err.message || String(err)}`);
  process.exit(1);
});
