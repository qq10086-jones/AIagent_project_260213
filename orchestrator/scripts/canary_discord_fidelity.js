#!/usr/bin/env node
/**
 * canary_discord_fidelity.js
 *
 * Focused Discord-to-fidelity quality canary.
 *
 * Runs a small set of Discord scenarios sequentially through the live
 * orchestrator and captures product fidelity classification, perceptual
 * quality, timing, and GO/No-Go results for each run.
 *
 * Unlike the load test (which measures throughput), this canary is
 * quality-focused: it verifies that product_fidelity_report.json is
 * generated for every run and reports the classifier's actual output
 * against expected classification ranges.
 *
 * Usage:
 *   node scripts/canary_discord_fidelity.js [options]
 *
 * Options:
 *   --base-url <url>        Orchestrator base URL. Default: http://localhost:3000
 *   --suite <file>          Suite JSON name or absolute path. Default: discord_fidelity_suite_v1.json
 *   --timeout-sec <sec>     Per-workflow timeout. Default: 1800
 *   --poll-ms <ms>          Workflow poll interval. Default: 5000
 *   --fail-on-warn          Exit non-zero when any fidelity warning is triggered.
 *   --fail-on-unexpected    Exit non-zero when classification is outside expected_classifications.
 *   --strict <bool>         Exit non-zero on any dispatch or workflow failure. Default: true
 */

import fs from "fs";
import path from "path";
import {
  getDefaultWorkspaceRoot,
  resolveCanaryInputPath,
  resolveOrchestratorArtifactPath,
} from "./_paths.js";

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

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

function resolveSuitePath(inputArg) {
  if (!inputArg) return resolveCanaryInputPath("discord_fidelity_suite_v1.json");
  if (path.isAbsolute(inputArg)) return inputArg;
  const canaryPath = resolveCanaryInputPath(inputArg);
  if (fs.existsSync(canaryPath)) return canaryPath;
  return path.resolve(process.cwd(), inputArg);
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

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
    throw new Error(`health check failed: ${res.status} ${text}`);
  }
}

// ---------------------------------------------------------------------------
// Workflow polling
// ---------------------------------------------------------------------------

async function pollWorkflow(baseUrl, workflowRunId, timeoutMs, pollMs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const state = await getJson(`${baseUrl}/workflow-runs/${encodeURIComponent(workflowRunId)}`);
    if (state.ok) {
      const status = String(state.json?.run?.status || "");
      if (["succeeded", "failed", "partial_failure"].includes(status)) {
        return state.json;
      }
    }
    await sleep(pollMs);
  }
  throw new Error(`timeout waiting for workflow ${workflowRunId} after ${timeoutMs}ms`);
}

// ---------------------------------------------------------------------------
// Artifact readers
// ---------------------------------------------------------------------------

function readArtifact(workspaceRoot, runId, relPath) {
  const p = path.resolve(workspaceRoot, "artifacts", "release", String(runId || ""), relPath);
  if (!fs.existsSync(p)) return null;
  try {
    return readJson(p);
  } catch {
    return null;
  }
}

function readFidelityReport(workspaceRoot, runId) {
  const value = readArtifact(workspaceRoot, runId, "qa/product_fidelity_report.json");
  if (!value) return { exists: false, classification: "", perceptual_score: "", should_warn: false, reasoning_count: 0 };
  return {
    exists: true,
    classification: String(value?.classification || ""),
    perceptual_score: String(value?.perceptual_quality?.score || ""),
    should_warn: Boolean(value?.should_warn),
    reasoning_count: Array.isArray(value?.reasoning) ? value.reasoning.length : 0,
    domain_pack: value?.domain_pack?.domain || null,
  };
}

function readPreviewValidationReport(workspaceRoot, runId) {
  const value = readArtifact(workspaceRoot, runId, "qa/preview_validation_report.json");
  if (!value) return { exists: false, classification: "", preview_source: "" };
  return {
    exists: true,
    classification: String(value?.classification || ""),
    preview_source: String(value?.preview_source || ""),
    should_warn: Boolean(value?.should_warn),
  };
}

function readGoNoGo(workspaceRoot, runId) {
  const value = readArtifact(workspaceRoot, runId, "qa/go_no_go_result.json");
  if (!value) return { exists: false, verdict: "", failed_checks: 0 };
  return {
    exists: true,
    verdict: String(value?.verdict || ""),
    failed_checks: Number(value?.failed_checks || 0),
  };
}

// ---------------------------------------------------------------------------
// Per-step timing extraction
// ---------------------------------------------------------------------------

function extractStepTimings(steps) {
  if (!Array.isArray(steps)) return [];
  return steps.map((s) => ({
    step_id: String(s?.step_id || ""),
    status: String(s?.status || ""),
  }));
}

// ---------------------------------------------------------------------------
// Single scenario run
// ---------------------------------------------------------------------------

async function runScenario({ baseUrl, scenario, suiteDefaults, index, timeoutMs, pollMs, workspaceRoot, stamp }) {
  const channelPrefix = String(suiteDefaults.channel_prefix || "fidelity-test");
  const userPrefix = String(suiteDefaults.user_prefix || "fidelity-user");
  const threadPrefix = String(suiteDefaults.thread_prefix || "fidelity-thread");
  const guildId = String(suiteDefaults.guild_id || "guild-fidelity");
  const message = `${String(scenario.message || "").trim()} [FDC ${stamp} #${index + 1}]`;
  const discordEvent = {
    content: message,
    channel_id: `${channelPrefix}-1`,
    thread_id: `${threadPrefix}-${index + 1}`,
    guild_id: guildId,
    author: { id: `${userPrefix}-1`, username: `${userPrefix}-1` },
    attachments: [],
  };

  const startedAt = Date.now();
  console.log(`\n[fidelity-canary] #${index + 1} scenario=${scenario.id} risk=${scenario.risk_profile || "?"}`);
  console.log(`[fidelity-canary]   message: ${message.slice(0, 80)}...`);

  // Dispatch
  const dispatch = await getJson(`${baseUrl}/vnext/dispatch`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ discord_event: discordEvent }),
  });
  const dispatchLatencyMs = Date.now() - startedAt;

  if (!dispatch.ok || dispatch.json?.ok === false) {
    return {
      index,
      scenario_id: scenario.id,
      risk_profile: scenario.risk_profile || "",
      expected_classifications: Array.isArray(scenario.expected_classifications) ? scenario.expected_classifications : [],
      dispatch_ok: false,
      dispatch_latency_ms: dispatchLatencyMs,
      workflow_status: "",
      workflow_run_id: "",
      run_id: "",
      fidelity: { exists: false, classification: "", perceptual_score: "", should_warn: false, reasoning_count: 0, domain_pack: null },
      preview: { exists: false, classification: "", preview_source: "" },
      go_no_go: { exists: false, verdict: "", failed_checks: 0 },
      steps: [],
      classification_expected: false,
      total_duration_ms: Date.now() - startedAt,
      ok: false,
      error: `dispatch failed: HTTP ${dispatch.status}`,
    };
  }

  const mode = String(dispatch.json?.mode || dispatch.json?.response_mode || "");
  const workflowRunId = String(dispatch.json?.workflow_run_id || dispatch.json?.execution?.workflow_run_id || "");
  const runId = String(dispatch.json?.run_id || "");

  if (!workflowRunId) {
    return {
      index,
      scenario_id: scenario.id,
      risk_profile: scenario.risk_profile || "",
      expected_classifications: Array.isArray(scenario.expected_classifications) ? scenario.expected_classifications : [],
      dispatch_ok: true,
      dispatch_latency_ms: dispatchLatencyMs,
      workflow_status: "",
      workflow_run_id: "",
      run_id: runId,
      fidelity: { exists: false, classification: "", perceptual_score: "", should_warn: false, reasoning_count: 0, domain_pack: null },
      preview: { exists: false, classification: "", preview_source: "" },
      go_no_go: { exists: false, verdict: "", failed_checks: 0 },
      steps: [],
      classification_expected: false,
      total_duration_ms: Date.now() - startedAt,
      ok: false,
      error: `no workflow_run_id in dispatch response (mode=${mode})`,
    };
  }

  console.log(`[fidelity-canary]   dispatched workflow_run_id=${workflowRunId} dispatch_ms=${dispatchLatencyMs}`);

  // Poll to completion
  let terminal = null;
  let pollError = null;
  try {
    terminal = await pollWorkflow(baseUrl, workflowRunId, timeoutMs, pollMs);
  } catch (err) {
    pollError = err.message || String(err);
  }

  const totalDurationMs = Date.now() - startedAt;
  const workflowStatus = String(terminal?.run?.status || (pollError ? "timeout" : "unknown"));
  const resolvedRunId = runId || String(terminal?.run?.run_id || "");

  // Read quality artifacts
  const fidelity = readFidelityReport(workspaceRoot, resolvedRunId);
  const preview = readPreviewValidationReport(workspaceRoot, resolvedRunId);
  const goNoGo = readGoNoGo(workspaceRoot, resolvedRunId);
  const steps = extractStepTimings(terminal?.steps);

  // Validate classification against expected range
  const expectedClassifications = Array.isArray(scenario.expected_classifications) ? scenario.expected_classifications : [];
  const classificationExpected = !fidelity.exists || expectedClassifications.length === 0
    ? null
    : expectedClassifications.includes(fidelity.classification);

  const ok = !pollError && ["succeeded", "partial_failure"].includes(workflowStatus);

  console.log(`[fidelity-canary]   workflow=${workflowStatus} total_ms=${totalDurationMs}`);
  console.log(`[fidelity-canary]   fidelity: exists=${fidelity.exists} class=${fidelity.classification || "n/a"} perceptual=${fidelity.perceptual_score || "n/a"} warn=${fidelity.should_warn}`);
  if (fidelity.exists && classificationExpected === false) {
    console.warn(`[fidelity-canary]   UNEXPECTED classification=${fidelity.classification} expected=[${expectedClassifications.join(",")}]`);
  }
  if (!fidelity.exists && workflowStatus === "succeeded") {
    console.warn(`[fidelity-canary]   MISSING fidelity report despite workflow succeeded`);
  }

  return {
    index,
    scenario_id: scenario.id,
    risk_profile: scenario.risk_profile || "",
    expected_classifications: expectedClassifications,
    dispatch_ok: true,
    dispatch_latency_ms: dispatchLatencyMs,
    workflow_status: workflowStatus,
    workflow_run_id: workflowRunId,
    run_id: resolvedRunId,
    fidelity,
    preview,
    go_no_go: goNoGo,
    steps,
    classification_expected: classificationExpected,
    total_duration_ms: totalDurationMs,
    ok,
    error: pollError || null,
  };
}

// ---------------------------------------------------------------------------
// Report builder
// ---------------------------------------------------------------------------

function makeReportMd(report) {
  const lines = [
    "# Discord Fidelity Canary Report",
    "",
    `- generated_at: ${report.generated_at}`,
    `- suite_file: ${report.config.suite_file}`,
    `- base_url: ${report.config.base_url}`,
    `- scenarios: ${report.results.length}`,
    `- fidelity_report_rate: ${report.summary.fidelity_report_rate_pct}%`,
    `- fidelity_warn_rate: ${report.summary.fidelity_warn_rate_pct !== null ? `${report.summary.fidelity_warn_rate_pct}%` : "n/a"}`,
    `- verdict: ${report.verdict}`,
    "",
    "## Results Per Scenario",
    "",
  ];

  for (const r of report.results) {
    lines.push(`### #${r.index + 1} ${r.scenario_id} (${r.risk_profile})`);
    lines.push(`- workflow: ${r.workflow_status} | total_ms: ${r.total_duration_ms} | dispatch_ms: ${r.dispatch_latency_ms}`);
    lines.push(`- fidelity_exists: ${r.fidelity.exists}`);
    if (r.fidelity.exists) {
      lines.push(`- fidelity_classification: ${r.fidelity.classification}`);
      lines.push(`- perceptual_quality: ${r.fidelity.perceptual_score}`);
      lines.push(`- fidelity_should_warn: ${r.fidelity.should_warn}`);
      lines.push(`- reasoning_criteria: ${r.fidelity.reasoning_count}`);
      if (r.fidelity.domain_pack) lines.push(`- domain_pack: ${r.fidelity.domain_pack}`);
    }
    lines.push(`- preview_classification: ${r.preview.classification || "n/a"}`);
    lines.push(`- go_no_go_verdict: ${r.go_no_go.verdict || "n/a"}`);
    if (r.expected_classifications.length > 0) {
      const match = r.classification_expected === true ? "PASS" : r.classification_expected === false ? "FAIL" : "n/a";
      lines.push(`- classification_check: ${match} (expected=[${r.expected_classifications.join(",")}] actual=${r.fidelity.classification || "none"})`);
    }
    if (r.error) lines.push(`- error: ${r.error}`);
    lines.push("");
  }

  lines.push("## Fidelity Classification Distribution");
  for (const [cls, count] of Object.entries(report.summary.fidelity_classification_counts)) {
    lines.push(`- ${cls}: ${count}`);
  }
  lines.push("");
  lines.push("## Perceptual Quality Distribution");
  for (const [score, count] of Object.entries(report.summary.perceptual_score_counts)) {
    lines.push(`- ${score}: ${count}`);
  }
  lines.push("");
  lines.push("## Timing");
  lines.push(`- min_total_ms: ${report.timing.min_total_ms}`);
  lines.push(`- max_total_ms: ${report.timing.max_total_ms}`);
  lines.push(`- avg_total_ms: ${report.timing.avg_total_ms}`);

  return `${lines.join("\n")}\n`;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  if (hasFlag("help")) {
    console.log("Usage: node scripts/canary_discord_fidelity.js [options]");
    console.log("  --base-url <url>        Orchestrator base URL. Default: http://localhost:3000");
    console.log("  --suite <file>          Suite JSON name or absolute path.");
    console.log("  --timeout-sec <sec>     Per-workflow timeout. Default: 1800");
    console.log("  --poll-ms <ms>          Workflow poll interval. Default: 5000");
    console.log("  --fail-on-warn          Exit non-zero if any fidelity warning triggered.");
    console.log("  --fail-on-unexpected    Exit non-zero if classification outside expected range.");
    console.log("  --strict <bool>         Exit non-zero on dispatch/workflow failure. Default: true");
    return;
  }

  const baseUrl = String(arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000")).replace(/\/+$/, "");
  const suiteFile = resolveSuitePath(arg("suite", "discord_fidelity_suite_v1.json"));
  const timeoutMs = Math.max(120000, toInt(arg("timeout-sec", "1800"), 1800) * 1000);
  const pollMs = toInt(arg("poll-ms", "5000"), 5000);
  const failOnWarn = hasFlag("fail-on-warn");
  const failOnUnexpected = hasFlag("fail-on-unexpected");
  const strict = parseBool(arg("strict", "true"), true);
  const workspaceRoot = path.resolve(arg("workspace-root", getDefaultWorkspaceRoot()));

  if (!fs.existsSync(suiteFile)) throw new Error(`suite file not found: ${suiteFile}`);
  const suite = readJson(suiteFile);
  const scenarios = Array.isArray(suite.scenarios) ? suite.scenarios : [];
  if (scenarios.length === 0) throw new Error("suite.scenarios is empty");

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");

  console.log(`[fidelity-canary] base_url=${baseUrl}`);
  console.log(`[fidelity-canary] suite=${suiteFile} scenarios=${scenarios.length}`);
  await checkHealth(baseUrl);

  const canaryStartedAt = Date.now();
  const results = [];
  for (let i = 0; i < scenarios.length; i++) {
    const result = await runScenario({
      baseUrl,
      scenario: scenarios[i],
      suiteDefaults: suite.defaults || {},
      index: i,
      timeoutMs,
      pollMs,
      workspaceRoot,
      stamp,
    });
    results.push(result);
  }
  const canaryFinishedAt = Date.now();

  // Aggregate
  const withFidelity = results.filter((r) => r.fidelity.exists);
  const classificationCounts = {};
  const perceptualCounts = {};
  for (const r of withFidelity) {
    const cls = r.fidelity.classification || "unknown";
    classificationCounts[cls] = (classificationCounts[cls] || 0) + 1;
    const ps = r.fidelity.perceptual_score || "unknown";
    perceptualCounts[ps] = (perceptualCounts[ps] || 0) + 1;
  }
  const totalDurations = results.map((r) => r.total_duration_ms).filter((n) => n > 0);
  const avgTotalMs = totalDurations.length > 0 ? Math.round(totalDurations.reduce((a, b) => a + b, 0) / totalDurations.length) : 0;

  const fidelityReportCount = withFidelity.length;
  const fidelityWarnCount = results.filter((r) => r.fidelity.should_warn).length;
  const fidelityReportRatePct = results.length > 0 ? Math.round((fidelityReportCount / results.length) * 1000) / 10 : 0;
  const fidelityWarnRatePct = fidelityReportCount > 0 ? Math.round((fidelityWarnCount / fidelityReportCount) * 1000) / 10 : null;

  // Verdict determination
  const unexpectedClassifications = results.filter((r) => r.classification_expected === false);
  const dispatchFailures = results.filter((r) => !r.dispatch_ok);
  const workflowFailures = results.filter((r) => r.dispatch_ok && r.ok === false);
  const missingFidelityOnSuccess = results.filter((r) => r.workflow_status === "succeeded" && !r.fidelity.exists);

  const verdictReasons = [];
  if (strict && dispatchFailures.length > 0) {
    verdictReasons.push(`${dispatchFailures.length} dispatch failure(s)`);
  }
  if (strict && workflowFailures.length > 0) {
    verdictReasons.push(`${workflowFailures.length} workflow failure(s)`);
  }
  if (missingFidelityOnSuccess.length > 0) {
    verdictReasons.push(`${missingFidelityOnSuccess.length} run(s) succeeded but fidelity report missing`);
  }
  if (failOnWarn && fidelityWarnCount > 0) {
    verdictReasons.push(`${fidelityWarnCount} fidelity warning(s) (--fail-on-warn active)`);
  }
  if (failOnUnexpected && unexpectedClassifications.length > 0) {
    verdictReasons.push(`${unexpectedClassifications.length} unexpected classification(s) (--fail-on-unexpected active)`);
  }

  const verdict = verdictReasons.length === 0 ? "PASS" : "FAIL";
  const summary = {
    total_scenarios: results.length,
    workflow_success_count: results.filter((r) => r.workflow_status === "succeeded").length,
    fidelity_report_count: fidelityReportCount,
    fidelity_warn_count: fidelityWarnCount,
    fidelity_report_rate_pct: fidelityReportRatePct,
    fidelity_warn_rate_pct: fidelityWarnRatePct,
    missing_fidelity_on_success: missingFidelityOnSuccess.length,
    unexpected_classification_count: unexpectedClassifications.length,
    fidelity_classification_counts: classificationCounts,
    perceptual_score_counts: perceptualCounts,
  };

  const report = {
    generated_at: new Date().toISOString(),
    config: { base_url: baseUrl, suite_file: suiteFile, timeout_ms: timeoutMs, poll_ms: pollMs, fail_on_warn: failOnWarn, fail_on_unexpected: failOnUnexpected },
    execution_window: {
      started_at: new Date(canaryStartedAt).toISOString(),
      finished_at: new Date(canaryFinishedAt).toISOString(),
      elapsed_ms: canaryFinishedAt - canaryStartedAt,
    },
    summary,
    timing: {
      min_total_ms: totalDurations.length > 0 ? Math.min(...totalDurations) : 0,
      max_total_ms: totalDurations.length > 0 ? Math.max(...totalDurations) : 0,
      avg_total_ms: avgTotalMs,
    },
    verdict,
    verdict_reasons: verdictReasons,
    results,
  };

  const outDir = resolveOrchestratorArtifactPath("validation", "discord_fidelity_canary", stamp);
  const jsonPath = path.join(outDir, "discord_fidelity_canary_report.json");
  const mdPath = path.join(outDir, "discord_fidelity_canary_report.md");
  writeJson(jsonPath, report);
  writeText(mdPath, makeReportMd(report));

  console.log("\n[fidelity-canary] === SUMMARY ===");
  console.log(`- scenarios: ${summary.total_scenarios}`);
  console.log(`- workflow_success: ${summary.workflow_success_count}/${summary.total_scenarios}`);
  console.log(`- fidelity_report_rate: ${summary.fidelity_report_rate_pct}%`);
  console.log(`- fidelity_warn_rate: ${summary.fidelity_warn_rate_pct !== null ? `${summary.fidelity_warn_rate_pct}%` : "n/a"}`);
  console.log(`- missing_fidelity_on_success: ${summary.missing_fidelity_on_success}`);
  if (unexpectedClassifications.length > 0) {
    console.warn(`- unexpected_classifications: ${unexpectedClassifications.map((r) => `${r.scenario_id}=${r.fidelity.classification}`).join(", ")}`);
  }
  console.log(`- verdict: ${verdict}`);
  if (verdictReasons.length > 0) {
    for (const reason of verdictReasons) console.warn(`  - ${reason}`);
  }
  console.log(`- json: ${jsonPath.replace(/\\/g, "/")}`);
  console.log(`- md: ${mdPath.replace(/\\/g, "/")}`);

  if (verdict !== "PASS") process.exitCode = 1;
}

main().catch((err) => {
  console.error(`[fidelity-canary] fatal: ${err.message || String(err)}`);
  process.exit(1);
});
