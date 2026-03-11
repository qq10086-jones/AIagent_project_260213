import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";
import {
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

async function pollRun(baseUrl, runId, predicate, timeoutMs = 30000, intervalMs = 1000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const runRes = await getJson(`${baseUrl}/runs/${encodeURIComponent(runId)}/status`);
    if (runRes.ok && predicate(runRes.json || {})) {
      return runRes.json;
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting run ${runId}`);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const APPROVAL_TRIGGER_MESSAGE = "/coder: Fix repo bug by running rm -rf on old build files and build the service again";

async function checkHealth(baseUrl) {
  const res = await fetch(`${baseUrl}/health`);
  const text = await res.text();
  assert(res.ok && String(text).trim() === "ok", `health failed: ${res.status} ${text}`);
  return { ok: true, body: text };
}

async function validateDirectChatBypass(baseUrl) {
  const chatRes = await getJson(`${baseUrl}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message: "你好，帮我总结一下今天的项目进展" }),
  });
  assert(chatRes.ok, `direct chat request failed: ${chatRes.status}`);
  const runId = String(chatRes.json?.run_id || "");
  assert(runId, "direct chat run_id missing");
  assert(chatRes.json?.mode === "direct_reply", `expected direct_reply mode, got ${chatRes.json?.mode}`);

  const runState = await pollRun(baseUrl, runId, (payload) => Array.isArray(payload?.tasks), 10000, 500);
  assert(Array.isArray(runState.tasks), "direct chat run tasks missing");
  assert(runState.tasks.length === 0, `direct chat created unexpected tasks: ${runState.tasks.length}`);
  const counts = runState.counts || {};
  const total = Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0);
  assert(total === 0, `direct chat created unexpected task counts: ${JSON.stringify(counts)}`);

  return {
    run_id: runId,
    mode: chatRes.json?.mode || "",
    run_status: runState.run?.status || "",
    task_count: runState.tasks.length,
  };
}

async function validateApprovalReject(baseUrl, approvalToken) {
  const chatRes = await getJson(`${baseUrl}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message: APPROVAL_TRIGGER_MESSAGE,
      payload: { destructive: true },
    }),
  });
  assert(chatRes.ok, `approval reject request failed: ${chatRes.status}`);
  assert(chatRes.json?.mode === "workflow", `expected workflow mode, got ${chatRes.json?.mode}`);
  assert(chatRes.json?.waiting_approval === true, "expected waiting_approval=true");

  const runId = String(chatRes.json?.run_id || "");
  const taskId = String(chatRes.json?.first_step?.task_id || "");
  assert(runId && taskId, "approval reject run/task id missing");

  const pending = await getJson(`${baseUrl}/approvals/pending?limit=50`);
  assert(pending.ok, `pending approvals failed: ${pending.status}`);
  assert(Array.isArray(pending.json?.tasks), "pending approvals tasks missing");
  assert(pending.json.tasks.some((item) => String(item.task_id || "") === taskId), "pending approvals missing task");

  const rejectRes = await getJson(`${baseUrl}/tasks/${encodeURIComponent(taskId)}/reject`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "X-Approval-Token": approvalToken,
    },
    body: JSON.stringify({ reason: "live validation reject path" }),
  });
  assert(rejectRes.ok, `reject failed: ${rejectRes.status}`);

  const runState = await pollRun(
    baseUrl,
    runId,
    (payload) => payload?.run?.status === "failed" && Array.isArray(payload?.tasks) && payload.tasks.some((task) => String(task.task_id || "") === taskId && String(task.status || "") === "failed"),
    15000,
    1000,
  );
  const rejectedTask = runState.tasks.find((item) => String(item.task_id || "") === taskId) || {};
  assert(String(rejectedTask.error_code || "") === "APPROVAL_REJECTED", `expected APPROVAL_REJECTED, got ${rejectedTask.error_code}`);

  return {
    run_id: runId,
    task_id: taskId,
    run_status: runState.run?.status || "",
    task_status: rejectedTask.status || "",
    error_code: rejectedTask.error_code || "",
  };
}

async function validateApprovalApprove(baseUrl, approvalToken) {
  const chatRes = await getJson(`${baseUrl}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message: APPROVAL_TRIGGER_MESSAGE,
      payload: { destructive: true },
    }),
  });
  assert(chatRes.ok, `approval approve request failed: ${chatRes.status}`);
  assert(chatRes.json?.mode === "workflow", `expected workflow mode, got ${chatRes.json?.mode}`);
  assert(chatRes.json?.waiting_approval === true, "expected waiting_approval=true");

  const runId = String(chatRes.json?.run_id || "");
  const taskId = String(chatRes.json?.first_step?.task_id || "");
  assert(runId && taskId, "approval approve run/task id missing");

  const approveRes = await getJson(`${baseUrl}/tasks/${encodeURIComponent(taskId)}/approve`, {
    method: "POST",
    headers: {
      "X-Approval-Token": approvalToken,
    },
  });
  assert(approveRes.ok, `approve failed: ${approveRes.status}`);

  const runState = await pollRun(
    baseUrl,
    runId,
    (payload) => Array.isArray(payload?.tasks) && payload.tasks.some((task) => {
      const currentTaskId = String(task.task_id || "");
      const currentStatus = String(task.status || "");
      return currentTaskId === taskId && (currentStatus === "running" || currentStatus === "succeeded" || currentStatus === "failed");
    }),
    120000,
    1000,
  );
  const approvedTask = runState.tasks.find((item) => String(item.task_id || "") === taskId) || {};
  const approvedStatus = String(approvedTask.status || "");
  assert(approvedStatus !== "waiting_approval", "task remained waiting_approval after approve");
  assert(["running", "succeeded", "failed"].includes(approvedStatus), `unexpected task status after approve: ${approvedStatus}`);

  return {
    run_id: runId,
    task_id: taskId,
    run_status: runState.run?.status || "",
    task_status: approvedTask.status || "",
  };
}

export async function main(options = {}) {
  const baseUrl = String(options.baseUrl || arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000")).replace(/\/+$/, "");
  const approvalToken = String(options.approvalToken || arg("approval-token", process.env.APPROVAL_TOKEN || "dev-approval-token"));
  // Approval-approve path can take a few minutes when the live coding worker runs a real delegate task.
  const timeoutMs = Math.max(60000, Number(options.timeoutMs || arg("timeout-ms", "240000")));

  const report = {
    generated_at: new Date().toISOString(),
    base_url: baseUrl,
    timeout_ms: timeoutMs,
    checks: {},
    overall: "pass",
  };

  try {
    report.checks.health = await checkHealth(baseUrl);
    report.checks.direct_chat_bypass = await validateDirectChatBypass(baseUrl);
    report.checks.approval_reject = await validateApprovalReject(baseUrl, approvalToken);
    report.checks.approval_approve = await validateApprovalApprove(baseUrl, approvalToken);
  } catch (err) {
    report.overall = "fail";
    report.error = err.message || String(err);
  }

  const outDir = resolveOrchestratorArtifactPath("canary", "live_vnext_runtime");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "live_vnext_runtime_report.json");
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");

  console.log(`# Live vNext Runtime Validation`);
  console.log(`- report: ${outPath.replace(/\\/g, "/")}`);
  console.log(`- overall: ${report.overall}`);
  if (report.error) console.log(`- error: ${report.error}`);
  if (report.overall !== "pass") {
    throw new Error(report.error || "live vNext runtime validation failed");
  }
  return {
    reportPath: outPath.replace(/\\/g, "/"),
    report,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(`[live-validate] failed: ${err.message || String(err)}`);
    process.exit(1);
  });
}
