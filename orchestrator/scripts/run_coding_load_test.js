#!/usr/bin/env node

import fs from "fs";
import path from "path";
import {
  getDefaultWorkspaceRoot,
  resolveCanaryInputPath,
  resolveOrchestratorArtifactPath,
} from "./_paths.js";

function arg(name, fallback = "") {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return String(process.argv[idx + 1]);
  return fallback;
}

function hasFlag(name) {
  return process.argv.includes(`--${name}`);
}

function parseBool(value, fallback = false) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) return fallback;
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off"].includes(raw)) return false;
  return fallback;
}

function toInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : fallback;
}

function percentile(sortedValues, pct) {
  if (!sortedValues.length) return 0;
  const idx = Math.ceil((pct / 100) * sortedValues.length) - 1;
  return sortedValues[Math.max(0, Math.min(sortedValues.length - 1, idx))];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function writeText(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, value, "utf8");
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function parseJsonSafe(value, fallback = {}) {
  try {
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

async function getJson(url, init = {}) {
  const res = await fetch(url, init);
  const text = await res.text();
  let json = null;
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
  if (!res.ok || String(text).trim() !== "ok") {
    throw new Error(`health failed: ${res.status} ${text}`);
  }
}

function buildGoalVariant(baseGoal, index, className, stamp) {
  const suffixes = {
    short: "Keep scope minimal and reviewable.",
    medium: "Include release evidence and deterministic verification outputs.",
    long: "Exercise the full PM/Arch/BE/FE/QA/Release path with artifact completeness.",
  };
  return [
    String(baseGoal || "").trim(),
    `[CodingLoadTest ${stamp} #${index + 1}]`,
    `class=${className}.`,
    suffixes[className] || suffixes.medium,
  ].filter(Boolean).join(" ");
}

function buildClassSchedule({ warmupCount, runCount }) {
  const out = [];
  for (let i = 0; i < warmupCount; i += 1) out.push("warmup");
  const baseline = Math.max(0, runCount);
  const shortCount = Math.round(baseline * 0.4);
  const mediumCount = Math.round(baseline * 0.35);
  const longCount = Math.max(0, baseline - shortCount - mediumCount);
  for (let i = 0; i < shortCount; i += 1) out.push("short");
  for (let i = 0; i < mediumCount; i += 1) out.push("medium");
  for (let i = 0; i < longCount; i += 1) out.push("long");
  return out.map((item) => (item === "warmup" ? "short" : item));
}

function resolveInputPath(inputArg) {
  if (!inputArg) return resolveCanaryInputPath("crm_mini_stable_cloud_lane.json");
  if (path.isAbsolute(inputArg)) return inputArg;
  const canaryPath = resolveCanaryInputPath(inputArg);
  if (fs.existsSync(canaryPath)) return canaryPath;
  return path.resolve(process.cwd(), inputArg);
}

function applyOverrides(payload, { provider, model, executionLane, fastMode }) {
  const next = cloneJson(payload);
  next.input = next.input && typeof next.input === "object" ? next.input : {};
  if (provider) next.input.provider = provider;
  if (model) next.input.model = model;
  if (executionLane) next.input.execution_lane = executionLane;
  if (fastMode !== null) next.input.fast_mode = fastMode;
  return next;
}

function extractStepMetrics(steps) {
  const stepStatusCounts = {};
  const failureCounts = {};
  for (const step of steps) {
    const status = String(step?.status || "unknown");
    const stepId = String(step?.step_id || "unknown");
    stepStatusCounts[status] = (stepStatusCounts[status] || 0) + 1;
    if (status === "failed") {
      failureCounts[stepId] = (failureCounts[stepId] || 0) + 1;
    }
  }
  return { stepStatusCounts, failureCounts };
}

function readGoNoGo(workspaceRoot, runId) {
  const goPath = path.resolve(workspaceRoot, "artifacts", "release", String(runId || ""), "qa", "go_no_go_result.json");
  if (!fs.existsSync(goPath)) {
    return { exists: false, verdict: "", path: goPath };
  }
  const value = readJson(goPath);
  return {
    exists: true,
    verdict: String(value?.verdict || ""),
    failed_checks: Number(value?.failed_checks || 0),
    path: goPath,
  };
}

async function startWorkflow(baseUrl, payload) {
  const startedAt = Date.now();
  const response = await getJson(`${baseUrl}/workflow-runs/start`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  return {
    ok: response.ok,
    status: response.status,
    body: response.json,
    latencyMs: Date.now() - startedAt,
  };
}

async function pollWorkflow(baseUrl, workflowRunId, timeoutMs, intervalMs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const response = await getJson(`${baseUrl}/workflow-runs/${encodeURIComponent(workflowRunId)}`);
    if (response.ok) {
      const status = String(response.json?.run?.status || "");
      if (["succeeded", "failed", "partial_failure"].includes(status)) {
        return response.json;
      }
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting workflow ${workflowRunId}`);
}

async function runSingleLoadItem({
  baseUrl,
  payload,
  timeoutMs,
  pollMs,
  workspaceRoot,
  index,
  className,
}) {
  const startedAt = Date.now();
  const dispatch = await startWorkflow(baseUrl, payload);
  if (!dispatch.ok) {
    return {
      index,
      class: className,
      ok: false,
      dispatch_ok: false,
      dispatch_status: dispatch.status,
      dispatch_latency_ms: dispatch.latencyMs,
      run_id: String(dispatch.body?.run_id || ""),
      workflow_run_id: String(dispatch.body?.workflow_run_id || ""),
      workflow_status: "dispatch_failed",
      error_code: String(dispatch.body?.error_code || ""),
      error_message: String(dispatch.body?.error || dispatch.body?.raw || `HTTP ${dispatch.status}`),
      total_duration_ms: Date.now() - startedAt,
      go_no_go_verdict: "",
      go_no_go_exists: false,
      steps: [],
      step_status_counts: {},
    };
  }

  const workflowRunId = String(dispatch.body?.workflow_run_id || "");
  const runId = String(dispatch.body?.run_id || "");
  if (!workflowRunId) {
    return {
      index,
      class: className,
      ok: false,
      dispatch_ok: true,
      dispatch_status: dispatch.status,
      dispatch_latency_ms: dispatch.latencyMs,
      run_id: runId,
      workflow_run_id: "",
      workflow_status: "invalid_response",
      error_code: "MISSING_WORKFLOW_RUN_ID",
      error_message: "workflow_run_id missing from /workflow-runs/start response",
      total_duration_ms: Date.now() - startedAt,
      go_no_go_verdict: "",
      go_no_go_exists: false,
      steps: [],
      step_status_counts: {},
    };
  }

  try {
    const terminal = await pollWorkflow(baseUrl, workflowRunId, timeoutMs, pollMs);
    const steps = Array.isArray(terminal?.steps) ? terminal.steps : [];
    const run = terminal?.run && typeof terminal.run === "object" ? terminal.run : {};
    const goNoGo = readGoNoGo(workspaceRoot, runId || run.run_id || "");
    const { stepStatusCounts } = extractStepMetrics(steps);
    const failedStep = steps.find((step) => String(step?.status || "") === "failed") || null;
    const failedOutput = parseJsonSafe(failedStep?.result_json || "{}", {});
    return {
      index,
      class: className,
      ok: String(run.status || "") === "succeeded" && (!goNoGo.exists || goNoGo.verdict === "GO"),
      dispatch_ok: true,
      dispatch_status: dispatch.status,
      dispatch_latency_ms: dispatch.latencyMs,
      run_id: String(runId || run.run_id || ""),
      workflow_run_id: workflowRunId,
      workflow_status: String(run.status || ""),
      error_code: String(run.error_code || failedStep?.error_code || failedOutput?.diagnostics?.error_code || ""),
      error_message: String(run.error_message || ""),
      total_duration_ms: Date.now() - startedAt,
      go_no_go_verdict: String(goNoGo.verdict || ""),
      go_no_go_exists: Boolean(goNoGo.exists),
      go_no_go_failed_checks: Number(goNoGo.failed_checks || 0),
      steps: steps.map((step) => ({
        step_id: String(step?.step_id || ""),
        status: String(step?.status || ""),
        task_id: String(step?.task_id || ""),
        error_code: String(step?.error_code || ""),
      })),
      step_status_counts: stepStatusCounts,
    };
  } catch (err) {
    return {
      index,
      class: className,
      ok: false,
      dispatch_ok: true,
      dispatch_status: dispatch.status,
      dispatch_latency_ms: dispatch.latencyMs,
      run_id: runId,
      workflow_run_id: workflowRunId,
      workflow_status: "timeout",
      error_code: "LOAD_TEST_TIMEOUT",
      error_message: err.message || String(err),
      total_duration_ms: Date.now() - startedAt,
      go_no_go_verdict: "",
      go_no_go_exists: false,
      steps: [],
      step_status_counts: {},
    };
  }
}

async function runWithConcurrency(items, concurrency, workerFn, staggerMs) {
  const results = new Array(items.length);
  let cursor = 0;
  async function worker(workerIndex) {
    while (cursor < items.length) {
      const current = cursor;
      cursor += 1;
      if (staggerMs > 0 && current > 0) {
        await sleep(staggerMs * workerIndex);
      }
      results[current] = await workerFn(items[current], current);
    }
  }
  const runners = [];
  for (let i = 0; i < Math.max(1, concurrency); i += 1) {
    runners.push(worker(i));
  }
  await Promise.all(runners);
  return results;
}

function buildHistogram(items, key) {
  const out = {};
  for (const item of items) {
    const value = String(item?.[key] || "unknown");
    out[value] = (out[value] || 0) + 1;
  }
  return out;
}

function makeReportMd(report) {
  const lines = [
    "# Coding Load Test Report",
    "",
    `- generated_at: ${report.generated_at}`,
    `- input_file: ${report.config.input_file}`,
    `- run_count: ${report.summary.total_runs}`,
    `- concurrency: ${report.config.concurrency}`,
    `- success_count: ${report.summary.success_count}`,
    `- failure_count: ${report.summary.failure_count}`,
    `- workflow_success_count: ${report.summary.workflow_success_count}`,
    `- go_no_go_pass_count: ${report.summary.go_no_go_pass_count}`,
    `- dispatch_p50_ms: ${report.latency.dispatch_p50_ms}`,
    `- dispatch_p95_ms: ${report.latency.dispatch_p95_ms}`,
    `- total_p50_ms: ${report.latency.total_p50_ms}`,
    `- total_p95_ms: ${report.latency.total_p95_ms}`,
    `- verdict: ${report.verdict}`,
    "",
    "## Workflow Status Histogram",
  ];
  for (const [status, count] of Object.entries(report.summary.workflow_status_counts)) {
    lines.push(`- ${status}: ${count}`);
  }
  lines.push("", "## Step Failure Histogram");
  const stepFailureEntries = Object.entries(report.summary.failed_step_counts);
  if (stepFailureEntries.length === 0) {
    lines.push("- none");
  } else {
    for (const [stepId, count] of stepFailureEntries) {
      lines.push(`- ${stepId}: ${count}`);
    }
  }
  lines.push("", "## Runs");
  for (const item of report.results) {
    lines.push(`- #${item.index} class=${item.class} workflow=${item.workflow_run_id || "n/a"} status=${item.workflow_status} dispatch_ms=${item.dispatch_latency_ms} total_ms=${item.total_duration_ms} gonogo=${item.go_no_go_verdict || "n/a"} error=${item.error_code || "none"}`);
  }
  return `${lines.join("\n")}\n`;
}

function printUsage() {
  console.log("Usage: node scripts/run_coding_load_test.js [options]");
  console.log("");
  console.log("Options:");
  console.log("  --base-url <url>           Orchestrator base URL. Default: http://localhost:3000");
  console.log("  --input <file>             Canary input JSON name or absolute path.");
  console.log("  --runs <n>                 Measured run count. Default: 12");
  console.log("  --warmup <n>               Warmup run count. Default: 2");
  console.log("  --concurrency <n>          Concurrent workflow submissions. Default: 3");
  console.log("  --stagger-ms <ms>          Per-worker launch stagger. Default: 250");
  console.log("  --poll-ms <ms>             Workflow poll interval. Default: 5000");
  console.log("  --timeout-sec <sec>        Per-workflow timeout. Default: 1800");
  console.log("  --provider <name>          Override input.provider.");
  console.log("  --model <name>             Override input.model.");
  console.log("  --execution-lane <name>    Override input.execution_lane.");
  console.log("  --fast-mode <bool>         Override input.fast_mode.");
  console.log("  --strict <bool>            Exit non-zero when failures exist. Default: true");
}

async function main() {
  if (hasFlag("help")) {
    printUsage();
    return;
  }

  const baseUrl = String(arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000")).replace(/\/+$/, "");
  const inputFile = resolveInputPath(arg("input", "crm_mini_stable_cloud_lane.json"));
  const runCount = Math.max(1, toInt(arg("runs", "12"), 12));
  const warmupCount = toInt(arg("warmup", "2"), 2);
  const concurrency = Math.max(1, toInt(arg("concurrency", "3"), 3));
  const staggerMs = toInt(arg("stagger-ms", "250"), 250);
  const pollMs = toInt(arg("poll-ms", "5000"), 5000);
  const timeoutMs = Math.max(120000, toInt(arg("timeout-sec", "1800"), 1800) * 1000);
  const provider = String(arg("provider", "")).trim();
  const model = String(arg("model", "")).trim();
  const executionLane = String(arg("execution-lane", "")).trim();
  const fastModeArg = String(arg("fast-mode", "")).trim();
  const fastMode = fastModeArg ? parseBool(fastModeArg, false) : null;
  const strict = parseBool(arg("strict", "true"), true);
  const workspaceRoot = path.resolve(arg("workspace-root", getDefaultWorkspaceRoot()));

  if (!fs.existsSync(inputFile)) {
    throw new Error(`input file not found: ${inputFile}`);
  }

  const basePayload = applyOverrides(readJson(inputFile), {
    provider,
    model,
    executionLane,
    fastMode,
  });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const classes = buildClassSchedule({ warmupCount, runCount });
  const allPayloads = classes.map((className, index) => {
    const next = cloneJson(basePayload);
    next.input = next.input && typeof next.input === "object" ? next.input : {};
    next.input.goal = buildGoalVariant(next.input.goal, index, className, stamp);
    next.input.load_test_meta = {
      tag: "coding_load_test",
      stamp,
      index: index + 1,
      class: className,
    };
    return { className, payload: next };
  });

  console.log(`[coding-load-test] base_url=${baseUrl}`);
  console.log(`[coding-load-test] input=${inputFile}`);
  console.log(`[coding-load-test] runs=${runCount} warmup=${warmupCount} concurrency=${concurrency}`);

  await checkHealth(baseUrl);

  const startedAt = Date.now();
  const results = await runWithConcurrency(
    allPayloads,
    concurrency,
    async (item, index) => {
      console.log(`[coding-load-test] start #${index + 1} class=${item.className}`);
      const result = await runSingleLoadItem({
        baseUrl,
        payload: item.payload,
        timeoutMs,
        pollMs,
        workspaceRoot,
        index: index + 1,
        className: item.className,
      });
      console.log(`[coding-load-test] done  #${index + 1} class=${item.className} status=${result.workflow_status} error=${result.error_code || "none"}`);
      return result;
    },
    staggerMs,
  );
  const finishedAt = Date.now();

  const measuredResults = results.slice(warmupCount);
  const dispatchDurations = measuredResults.map((item) => Number(item.dispatch_latency_ms || 0)).sort((a, b) => a - b);
  const totalDurations = measuredResults.map((item) => Number(item.total_duration_ms || 0)).sort((a, b) => a - b);
  const workflowStatusCounts = buildHistogram(measuredResults, "workflow_status");
  const failedStepCounts = {};
  for (const item of measuredResults) {
    for (const step of item.steps || []) {
      if (String(step?.status || "") === "failed") {
        const stepId = String(step?.step_id || "unknown");
        failedStepCounts[stepId] = (failedStepCounts[stepId] || 0) + 1;
      }
    }
  }

  const summary = {
    total_runs: measuredResults.length,
    success_count: measuredResults.filter((item) => item.ok).length,
    failure_count: measuredResults.filter((item) => !item.ok).length,
    workflow_success_count: measuredResults.filter((item) => item.workflow_status === "succeeded").length,
    go_no_go_pass_count: measuredResults.filter((item) => item.go_no_go_verdict === "GO").length,
    workflow_status_counts: workflowStatusCounts,
    failed_step_counts: failedStepCounts,
  };

  const report = {
    generated_at: new Date().toISOString(),
    config: {
      base_url: baseUrl,
      input_file: inputFile,
      workspace_root: workspaceRoot,
      provider: provider || String(basePayload.input?.provider || ""),
      model: model || String(basePayload.input?.model || ""),
      execution_lane: executionLane || String(basePayload.input?.execution_lane || ""),
      fast_mode: fastMode === null ? basePayload.input?.fast_mode : fastMode,
      run_count: runCount,
      warmup_count: warmupCount,
      concurrency,
      stagger_ms: staggerMs,
      poll_ms: pollMs,
      timeout_ms: timeoutMs,
      strict,
    },
    execution_window: {
      started_at: new Date(startedAt).toISOString(),
      finished_at: new Date(finishedAt).toISOString(),
      elapsed_ms: finishedAt - startedAt,
    },
    latency: {
      dispatch_p50_ms: percentile(dispatchDurations, 50),
      dispatch_p95_ms: percentile(dispatchDurations, 95),
      total_p50_ms: percentile(totalDurations, 50),
      total_p95_ms: percentile(totalDurations, 95),
    },
    summary,
    verdict: summary.failure_count === 0 ? "PASS" : "FAIL",
    results,
  };

  const outDir = resolveOrchestratorArtifactPath("validation", "coding_load_test", stamp);
  const jsonPath = path.join(outDir, "coding_load_test_report.json");
  const mdPath = path.join(outDir, "coding_load_test_report.md");
  writeJson(jsonPath, report);
  writeText(mdPath, makeReportMd(report));

  console.log("# Coding Load Test");
  console.log(`- json: ${jsonPath.replace(/\\/g, "/")}`);
  console.log(`- md: ${mdPath.replace(/\\/g, "/")}`);
  console.log(`- success_count: ${summary.success_count}`);
  console.log(`- failure_count: ${summary.failure_count}`);
  console.log(`- verdict: ${report.verdict}`);

  if (strict && report.verdict !== "PASS") {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(`[coding-load-test] failed: ${err.message || String(err)}`);
  process.exit(1);
});
