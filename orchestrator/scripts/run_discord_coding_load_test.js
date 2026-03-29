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

function toOptionalNumber(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
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

function resolveSuitePath(inputArg) {
  if (!inputArg) return resolveCanaryInputPath("discord_coding_light_suite_v1.json");
  if (path.isAbsolute(inputArg)) return inputArg;
  const canaryPath = resolveCanaryInputPath(inputArg);
  if (fs.existsSync(canaryPath)) return canaryPath;
  return path.resolve(process.cwd(), inputArg);
}

function buildWeightedPool(scenarios) {
  const pool = [];
  for (const item of scenarios) {
    const weight = Math.max(1, Number(item.weight || 1));
    for (let i = 0; i < weight; i += 1) {
      pool.push(item);
    }
  }
  return pool;
}

function pickScenario(weightedPool, index) {
  if (!weightedPool.length) throw new Error("scenario pool is empty");
  return weightedPool[index % weightedPool.length];
}

function buildDiscordEvent({ scenario, suiteDefaults, index, stamp }) {
  const channelPrefix = String(suiteDefaults.channel_prefix || "load-test-chan");
  const userPrefix = String(suiteDefaults.user_prefix || "load-test-user");
  const threadPrefix = String(suiteDefaults.thread_prefix || "load-test-thread");
  const guildId = String(suiteDefaults.guild_id || "guild-load-test");
  const message = `${String(scenario.message || "").trim()} [DLT ${stamp} #${index + 1} scenario=${scenario.id}]`;
  return {
    content: message,
    channel_id: `${channelPrefix}-${(index % 4) + 1}`,
    thread_id: `${threadPrefix}-${index + 1}`,
    guild_id: guildId,
    author: {
      id: `${userPrefix}-${(index % 8) + 1}`,
      username: `${userPrefix}-${(index % 8) + 1}`,
    },
    attachments: Array.isArray(scenario.attachments) ? scenario.attachments : [],
  };
}

async function pollWorkflow(baseUrl, workflowRunId, timeoutMs, intervalMs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const state = await getJson(`${baseUrl}/workflow-runs/${encodeURIComponent(workflowRunId)}`);
    if (state.ok) {
      const status = String(state.json?.run?.status || "");
      if (["succeeded", "failed", "partial_failure"].includes(status)) {
        return state.json;
      }
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting workflow ${workflowRunId}`);
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

function readFidelityReport(workspaceRoot, runId) {
  const p = path.resolve(workspaceRoot, "artifacts", "release", String(runId || ""), "qa", "product_fidelity_report.json");
  if (!fs.existsSync(p)) return { exists: false, classification: "", perceptual_score: "", should_warn: false };
  const value = readJson(p);
  return {
    exists: true,
    classification: String(value?.classification || ""),
    perceptual_score: String(value?.perceptual_quality?.score || ""),
    should_warn: Boolean(value?.should_warn),
  };
}

function readPreviewValidationReport(workspaceRoot, runId) {
  const p = path.resolve(workspaceRoot, "artifacts", "release", String(runId || ""), "qa", "preview_validation_report.json");
  if (!fs.existsSync(p)) return { exists: false, classification: "", should_warn: false };
  const value = readJson(p);
  return {
    exists: true,
    classification: String(value?.classification || ""),
    should_warn: Boolean(value?.should_warn),
  };
}

async function runSingleDispatch({
  baseUrl,
  discordEvent,
  scenario,
  index,
  timeoutMs,
  pollMs,
  workspaceRoot,
}) {
  const startedAt = Date.now();
  const dispatch = await getJson(`${baseUrl}/vnext/dispatch`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ discord_event: discordEvent }),
  });
  const dispatchLatencyMs = Date.now() - startedAt;
  const mode = String(dispatch.json?.mode || dispatch.json?.response_mode || "");
  const workflowRunId = String(dispatch.json?.workflow_run_id || dispatch.json?.execution?.workflow_run_id || "");
  const runId = String(dispatch.json?.run_id || "");

  if (!dispatch.ok || dispatch.json?.ok === false) {
    return {
      index,
      scenario_id: String(scenario.id || ""),
      class: String(scenario.class || ""),
      discord_message: discordEvent.content,
      dispatch_ok: false,
      dispatch_status: dispatch.status,
      dispatch_mode: mode || "dispatch_failed",
      dispatch_latency_ms: dispatchLatencyMs,
      run_id: runId,
      workflow_run_id: workflowRunId,
      workflow_status: "",
      go_no_go_verdict: "",
      go_no_go_exists: false,
      fidelity_classification: "",
      fidelity_perceptual_score: "",
      fidelity_should_warn: false,
      fidelity_exists: false,
      preview_classification: "",
      preview_should_warn: false,
      preview_validation_exists: false,
      ok: false,
      error_code: String(dispatch.json?.error_code || ""),
      error_message: String(dispatch.json?.error || dispatch.json?.raw || `HTTP ${dispatch.status}`),
    };
  }

  const workflowLikeMode = mode === "workflow" || mode === "progress_update";
  if (!workflowLikeMode || !workflowRunId) {
    return {
      index,
      scenario_id: String(scenario.id || ""),
      class: String(scenario.class || ""),
      discord_message: discordEvent.content,
      dispatch_ok: true,
      dispatch_status: dispatch.status,
      dispatch_mode: mode || "unknown",
      dispatch_latency_ms: dispatchLatencyMs,
      run_id: runId,
      workflow_run_id: workflowRunId,
      workflow_status: "",
      go_no_go_verdict: "",
      go_no_go_exists: false,
      fidelity_classification: "",
      fidelity_perceptual_score: "",
      fidelity_should_warn: false,
      fidelity_exists: false,
      preview_classification: "",
      preview_should_warn: false,
      preview_validation_exists: false,
      ok: false,
      error_code: "UNEXPECTED_DISPATCH_MODE",
      error_message: `expected workflow/progress_update mode but received '${mode || "unknown"}'`,
    };
  }

  try {
    const terminal = await pollWorkflow(baseUrl, workflowRunId, timeoutMs, pollMs);
    const workflowStatus = String(terminal?.run?.status || "");
    const resolvedRunId = runId || String(terminal?.run?.run_id || "");
    const goNoGo = readGoNoGo(workspaceRoot, resolvedRunId);
    const fidelity = readFidelityReport(workspaceRoot, resolvedRunId);
    const previewVal = readPreviewValidationReport(workspaceRoot, resolvedRunId);
    const steps = Array.isArray(terminal?.steps) ? terminal.steps : [];
    const failedStep = steps.find((step) => String(step?.status || "") === "failed") || null;
    return {
      index,
      scenario_id: String(scenario.id || ""),
      class: String(scenario.class || ""),
      discord_message: discordEvent.content,
      dispatch_ok: true,
      dispatch_status: dispatch.status,
      dispatch_mode: mode,
      dispatch_latency_ms: dispatchLatencyMs,
      run_id: resolvedRunId,
      workflow_run_id: workflowRunId,
      workflow_status: workflowStatus,
      go_no_go_verdict: String(goNoGo.verdict || ""),
      go_no_go_exists: Boolean(goNoGo.exists),
      go_no_go_failed_checks: Number(goNoGo.failed_checks || 0),
      fidelity_classification: fidelity.classification,
      fidelity_perceptual_score: fidelity.perceptual_score,
      fidelity_should_warn: fidelity.should_warn,
      fidelity_exists: fidelity.exists,
      preview_classification: previewVal.classification,
      preview_should_warn: previewVal.should_warn,
      preview_validation_exists: previewVal.exists,
      ok: workflowStatus === "succeeded" && (!goNoGo.exists || goNoGo.verdict === "GO"),
      error_code: String(terminal?.run?.error_code || failedStep?.error_code || ""),
      error_message: String(terminal?.run?.error_message || ""),
      total_duration_ms: Date.now() - startedAt,
    };
  } catch (err) {
    return {
      index,
      scenario_id: String(scenario.id || ""),
      class: String(scenario.class || ""),
      discord_message: discordEvent.content,
      dispatch_ok: true,
      dispatch_status: dispatch.status,
      dispatch_mode: mode,
      dispatch_latency_ms: dispatchLatencyMs,
      run_id: runId,
      workflow_run_id: workflowRunId,
      workflow_status: "timeout",
      go_no_go_verdict: "",
      go_no_go_exists: false,
      fidelity_classification: "",
      fidelity_perceptual_score: "",
      fidelity_should_warn: false,
      fidelity_exists: false,
      preview_classification: "",
      preview_should_warn: false,
      preview_validation_exists: false,
      ok: false,
      error_code: "LOAD_TEST_TIMEOUT",
      error_message: err.message || String(err),
      total_duration_ms: Date.now() - startedAt,
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
    "# Discord Coding Load Test Report",
    "",
    `- generated_at: ${report.generated_at}`,
    `- suite_file: ${report.config.suite_file}`,
    `- total_runs: ${report.summary.total_runs}`,
    `- success_count: ${report.summary.success_count}`,
    `- failure_count: ${report.summary.failure_count}`,
    `- workflow_success_count: ${report.summary.workflow_success_count}`,
    `- dispatch_p50_ms: ${report.latency.dispatch_p50_ms}`,
    `- dispatch_p95_ms: ${report.latency.dispatch_p95_ms}`,
    `- total_p50_ms: ${report.latency.total_p50_ms}`,
    `- total_p95_ms: ${report.latency.total_p95_ms}`,
    `- verdict: ${report.verdict}`,
    "",
    "## Dispatch Modes",
  ];
  for (const [mode, count] of Object.entries(report.summary.dispatch_mode_counts)) {
    lines.push(`- ${mode}: ${count}`);
  }
  lines.push("", "## Workflow Statuses");
  for (const [status, count] of Object.entries(report.summary.workflow_status_counts)) {
    lines.push(`- ${status}: ${count}`);
  }
  lines.push("", "## Scenario Counts");
  for (const [scenarioId, count] of Object.entries(report.summary.scenario_counts)) {
    lines.push(`- ${scenarioId}: ${count}`);
  }
  const fwRate = report.rates.fidelity_warn_rate;
  lines.push("", "## Product Fidelity Quality");
  lines.push(`- fidelity_report_rate: ${(report.rates.fidelity_report_rate * 100).toFixed(1)}% (${report.summary.fidelity_report_count}/${report.summary.total_runs} runs generated report)`);
  lines.push(`- fidelity_warn_rate: ${fwRate !== null ? `${(fwRate * 100).toFixed(1)}% (${report.summary.fidelity_warn_count}/${report.summary.fidelity_report_count} with report)` : "n/a (no fidelity reports)"}`);
  lines.push(`- preview_mismatch_count: ${report.summary.preview_mismatch_count}`);
  lines.push("", "### Fidelity Classifications");
  for (const [cls, count] of Object.entries(report.summary.fidelity_classification_counts)) {
    lines.push(`- ${cls}: ${count}`);
  }
  lines.push("", "### Perceptual Quality Scores");
  for (const [score, count] of Object.entries(report.summary.perceptual_score_counts)) {
    lines.push(`- ${score}: ${count}`);
  }
  lines.push("", "## Runs");
  for (const item of report.results) {
    const fidelityTag = item.fidelity_exists ? `fidelity=${item.fidelity_classification}(${item.fidelity_perceptual_score})` : "fidelity=n/a";
    lines.push(`- #${item.index} scenario=${item.scenario_id} class=${item.class} mode=${item.dispatch_mode} workflow_status=${item.workflow_status || "n/a"} dispatch_ms=${item.dispatch_latency_ms} total_ms=${item.total_duration_ms || 0} ${fidelityTag} error=${item.error_code || "none"}`);
  }
  return `${lines.join("\n")}\n`;
}

function printUsage() {
  console.log("Usage: node scripts/run_discord_coding_load_test.js [options]");
  console.log("");
  console.log("Options:");
  console.log("  --base-url <url>         Orchestrator base URL. Default: http://localhost:3000");
  console.log("  --suite <file>           Suite JSON name or absolute path.");
  console.log("  --runs <n>               Measured run count. Default: 12");
  console.log("  --warmup <n>             Warmup run count. Default: 3");
  console.log("  --concurrency <n>        Concurrent Discord dispatches. Default: 3");
  console.log("  --stagger-ms <ms>        Per-worker launch stagger. Default: 300");
  console.log("  --poll-ms <ms>           Workflow poll interval. Default: 5000");
  console.log("  --timeout-sec <sec>      Per-workflow timeout. Default: 1800");
  console.log("  --min-workflow-success-rate <0-1>  Gate threshold for workflow succeeded rate.");
  console.log("  --min-go-rate <0-1>      Gate threshold for GO/No-GO pass rate.");
  console.log("  --max-dispatch-p95-ms <ms> Gate threshold for dispatch p95 latency.");
  console.log("  --max-total-p95-ms <ms>  Gate threshold for end-to-end p95 latency.");
  console.log("  --max-fidelity-warn-rate <0-1>  Gate threshold for fidelity warn rate (runs with fidelity report).");
  console.log("  --strict <bool>          Exit non-zero when failures exist. Default: true");
}

async function main() {
  if (hasFlag("help")) {
    printUsage();
    return;
  }

  const baseUrl = String(arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000")).replace(/\/+$/, "");
  const suiteFile = resolveSuitePath(arg("suite", "discord_coding_light_suite_v1.json"));
  const runCount = Math.max(1, toInt(arg("runs", "12"), 12));
  const warmupCount = toInt(arg("warmup", "3"), 3);
  const concurrency = Math.max(1, toInt(arg("concurrency", "3"), 3));
  const staggerMs = toInt(arg("stagger-ms", "300"), 300);
  const pollMs = toInt(arg("poll-ms", "5000"), 5000);
  const timeoutMs = Math.max(120000, toInt(arg("timeout-sec", "1800"), 1800) * 1000);
  const minWorkflowSuccessRate = toOptionalNumber(arg("min-workflow-success-rate", ""));
  const minGoRate = toOptionalNumber(arg("min-go-rate", ""));
  const maxDispatchP95Ms = toOptionalNumber(arg("max-dispatch-p95-ms", ""));
  const maxTotalP95Ms = toOptionalNumber(arg("max-total-p95-ms", ""));
  const strict = parseBool(arg("strict", "true"), true);
  const workspaceRoot = path.resolve(arg("workspace-root", getDefaultWorkspaceRoot()));

  if (!fs.existsSync(suiteFile)) {
    throw new Error(`suite file not found: ${suiteFile}`);
  }

  const suite = readJson(suiteFile);
  const scenarios = Array.isArray(suite.scenarios) ? suite.scenarios : [];
  if (scenarios.length === 0) {
    throw new Error("suite.scenarios is empty");
  }
  const weightedPool = buildWeightedPool(scenarios);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const totalRuns = warmupCount + runCount;
  const jobs = Array.from({ length: totalRuns }, (_, index) => {
    const scenario = pickScenario(weightedPool, index);
    return {
      scenario,
      discordEvent: buildDiscordEvent({
        scenario,
        suiteDefaults: suite.defaults || {},
        index,
        stamp,
      }),
    };
  });

  console.log(`[discord-coding-load-test] base_url=${baseUrl}`);
  console.log(`[discord-coding-load-test] suite=${suiteFile}`);
  console.log(`[discord-coding-load-test] runs=${runCount} warmup=${warmupCount} concurrency=${concurrency}`);

  await checkHealth(baseUrl);

  const startedAt = Date.now();
  const results = await runWithConcurrency(
    jobs,
    concurrency,
    async (job, index) => {
      console.log(`[discord-coding-load-test] start #${index + 1} scenario=${job.scenario.id}`);
      const result = await runSingleDispatch({
        baseUrl,
        discordEvent: job.discordEvent,
        scenario: job.scenario,
        index: index + 1,
        timeoutMs,
        pollMs,
        workspaceRoot,
      });
      const fTag = result.fidelity_exists ? `fidelity=${result.fidelity_classification}(${result.fidelity_perceptual_score})` : "fidelity=n/a";
      console.log(`[discord-coding-load-test] done  #${index + 1} scenario=${job.scenario.id} mode=${result.dispatch_mode} workflow=${result.workflow_status || "n/a"} ${fTag} error=${result.error_code || "none"}`);
      return result;
    },
    staggerMs,
  );
  const finishedAt = Date.now();

  const measuredResults = results.slice(warmupCount);
  const dispatchDurations = measuredResults.map((item) => Number(item.dispatch_latency_ms || 0)).sort((a, b) => a - b);
  const totalDurations = measuredResults.map((item) => Number(item.total_duration_ms || 0)).sort((a, b) => a - b);
  const summary = {
    total_runs: measuredResults.length,
    success_count: measuredResults.filter((item) => item.ok).length,
    failure_count: measuredResults.filter((item) => !item.ok).length,
    workflow_success_count: measuredResults.filter((item) => item.workflow_status === "succeeded").length,
    go_pass_count: measuredResults.filter((item) => item.go_no_go_verdict === "GO").length,
    fidelity_report_count: measuredResults.filter((item) => item.fidelity_exists).length,
    fidelity_warn_count: measuredResults.filter((item) => item.fidelity_should_warn).length,
    preview_mismatch_count: measuredResults.filter((item) => item.preview_classification === "preview_mismatch").length,
    dispatch_mode_counts: buildHistogram(measuredResults, "dispatch_mode"),
    workflow_status_counts: buildHistogram(measuredResults, "workflow_status"),
    fidelity_classification_counts: buildHistogram(
      measuredResults.filter((item) => item.fidelity_exists),
      "fidelity_classification"
    ),
    perceptual_score_counts: buildHistogram(
      measuredResults.filter((item) => item.fidelity_exists),
      "fidelity_perceptual_score"
    ),
    scenario_counts: buildHistogram(measuredResults, "scenario_id"),
    class_counts: buildHistogram(measuredResults, "class"),
  };
  const workflowSuccessRate = summary.total_runs > 0 ? summary.workflow_success_count / summary.total_runs : 0;
  const goRate = summary.total_runs > 0 ? summary.go_pass_count / summary.total_runs : 0;
  const fidelityReportRate = summary.total_runs > 0 ? summary.fidelity_report_count / summary.total_runs : 0;
  const fidelityWarnRate = summary.fidelity_report_count > 0 ? summary.fidelity_warn_count / summary.fidelity_report_count : null;
  const latency = {
    dispatch_p50_ms: percentile(dispatchDurations, 50),
    dispatch_p95_ms: percentile(dispatchDurations, 95),
    total_p50_ms: percentile(totalDurations, 50),
    total_p95_ms: percentile(totalDurations, 95),
  };
  const maxFidelityWarnRate = toOptionalNumber(arg("max-fidelity-warn-rate", ""));
  const gateFailures = [];
  if (Number.isFinite(minWorkflowSuccessRate) && workflowSuccessRate < minWorkflowSuccessRate) {
    gateFailures.push(`workflow_success_rate ${workflowSuccessRate.toFixed(3)} < ${minWorkflowSuccessRate.toFixed(3)}`);
  }
  if (Number.isFinite(minGoRate) && goRate < minGoRate) {
    gateFailures.push(`go_rate ${goRate.toFixed(3)} < ${minGoRate.toFixed(3)}`);
  }
  if (Number.isFinite(maxDispatchP95Ms) && latency.dispatch_p95_ms > maxDispatchP95Ms) {
    gateFailures.push(`dispatch_p95_ms ${latency.dispatch_p95_ms} > ${maxDispatchP95Ms}`);
  }
  if (Number.isFinite(maxTotalP95Ms) && latency.total_p95_ms > maxTotalP95Ms) {
    gateFailures.push(`total_p95_ms ${latency.total_p95_ms} > ${maxTotalP95Ms}`);
  }
  if (Number.isFinite(maxFidelityWarnRate) && fidelityWarnRate !== null && fidelityWarnRate > maxFidelityWarnRate) {
    gateFailures.push(`fidelity_warn_rate ${fidelityWarnRate.toFixed(3)} > ${maxFidelityWarnRate.toFixed(3)}`);
  }

  const report = {
    generated_at: new Date().toISOString(),
    config: {
      base_url: baseUrl,
      suite_file: suiteFile,
      suite_id: String(suite.suite_id || ""),
      workspace_root: workspaceRoot,
      run_count: runCount,
      warmup_count: warmupCount,
      concurrency,
      stagger_ms: staggerMs,
      poll_ms: pollMs,
      timeout_ms: timeoutMs,
      min_workflow_success_rate: Number.isFinite(minWorkflowSuccessRate) ? minWorkflowSuccessRate : null,
      min_go_rate: Number.isFinite(minGoRate) ? minGoRate : null,
      max_dispatch_p95_ms: Number.isFinite(maxDispatchP95Ms) ? maxDispatchP95Ms : null,
      max_total_p95_ms: Number.isFinite(maxTotalP95Ms) ? maxTotalP95Ms : null,
      max_fidelity_warn_rate: Number.isFinite(maxFidelityWarnRate) ? maxFidelityWarnRate : null,
      strict,
    },
    execution_window: {
      started_at: new Date(startedAt).toISOString(),
      finished_at: new Date(finishedAt).toISOString(),
      elapsed_ms: finishedAt - startedAt,
    },
    latency,
    summary,
    rates: {
      workflow_success_rate: workflowSuccessRate,
      go_rate: goRate,
      fidelity_report_rate: fidelityReportRate,
      fidelity_warn_rate: fidelityWarnRate,
    },
    gate_failures: gateFailures,
    verdict: summary.failure_count === 0 && gateFailures.length === 0 ? "PASS" : "FAIL",
    results,
  };

  const outDir = resolveOrchestratorArtifactPath("validation", "discord_coding_load_test", stamp);
  const jsonPath = path.join(outDir, "discord_coding_load_test_report.json");
  const mdPath = path.join(outDir, "discord_coding_load_test_report.md");
  writeJson(jsonPath, report);
  writeText(mdPath, makeReportMd(report));

  console.log("# Discord Coding Load Test");
  console.log(`- json: ${jsonPath.replace(/\\/g, "/")}`);
  console.log(`- md: ${mdPath.replace(/\\/g, "/")}`);
  console.log(`- success_count: ${summary.success_count}`);
  console.log(`- failure_count: ${summary.failure_count}`);
  console.log(`- fidelity_report_rate: ${(fidelityReportRate * 100).toFixed(1)}%`);
  console.log(`- fidelity_warn_rate: ${fidelityWarnRate !== null ? `${(fidelityWarnRate * 100).toFixed(1)}%` : "n/a"}`);
  console.log(`- verdict: ${report.verdict}`);

  if (strict && report.verdict !== "PASS") {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(`[discord-coding-load-test] failed: ${err.message || String(err)}`);
  process.exit(1);
});
