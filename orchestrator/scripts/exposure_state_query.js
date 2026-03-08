#!/usr/bin/env node
/**
 * WS-25-01  Exposure State Diagnostic Tool
 *
 * Single-command tool for on-call engineers to query the current rollout state.
 * Must return accurate results within 30 seconds of invocation.
 *
 * Usage:
 *   node scripts/exposure_state_query.js [--last N]   (default N=100)
 *
 * Output:
 *   - Rollout master state
 *   - Circuit-breaker state
 *   - Eligibility policy summary
 *   - Effective exposure decision distribution for last N runs
 *   - Timestamp of last policy change
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const ROLLOUT_PATH = path.join(ROOT, "configs/production_parallel_rollout.json");
const ELIGIBILITY_PATH = path.join(ROOT, "configs/parallel_exposure_policy.json");
const STAGING_REPLAY_DIR = path.join(ROOT, "artifacts/m6_staging_replay");

function loadJson(p) {
  try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch { return null; }
}

function parseArgs(argv) {
  let last = 100;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--last" && argv[i + 1]) last = parseInt(argv[++i], 10);
  }
  return { last };
}

function formatBool(v) { return v === true ? "YES" : v === false ? "NO" : "UNKNOWN"; }

function collectRecentRunDecisions(last) {
  if (!fs.existsSync(STAGING_REPLAY_DIR)) return [];
  const runs = fs.readdirSync(STAGING_REPLAY_DIR)
    .filter((d) => d.startsWith("replay-"))
    .sort()
    .reverse();

  const decisions = [];
  for (const runDir of runs) {
    if (decisions.length >= last) break;
    const fullDir = path.join(STAGING_REPLAY_DIR, runDir);
    const files = fs.readdirSync(fullDir).filter((f) => f.endsWith(".json") && !f.startsWith("_"));
    for (const file of files) {
      if (decisions.length >= last) break;
      const data = loadJson(path.join(fullDir, file));
      if (data?.gate_decision?.effective_exposure_decision) {
        decisions.push({
          replay_id: data.replay_id,
          run_dir: runDir,
          decision: data.gate_decision.effective_exposure_decision,
          source: data.gate_decision.effective_exposure_decision_source,
          deny_reason: data.gate_decision.deny_reason_code || null,
          timestamp: data.timestamp,
        });
      }
    }
  }
  return decisions;
}

function summarizeDecisions(decisions) {
  const total = decisions.length;
  const parallel = decisions.filter((d) => d.decision === "gated_parallel_allowed").length;
  const sequential = total - parallel;
  const bySource = {};
  for (const d of decisions) {
    const key = d.deny_reason || d.source || "unknown";
    bySource[key] = (bySource[key] || 0) + 1;
  }
  return { total, parallel, sequential, bySource };
}

// ── Main ──────────────────────────────────────────────────────────────────────

const args = parseArgs(process.argv.slice(2));
const rollout = loadJson(ROLLOUT_PATH);
const eligibility = loadJson(ELIGIBILITY_PATH);
const decisions = collectRecentRunDecisions(args.last);
const summary = summarizeDecisions(decisions);

console.log("═══════════════════════════════════════════════════════");
console.log("  OpenClaw Nexus — M6 Exposure State Query");
console.log(`  ${new Date().toISOString()}`);
console.log("═══════════════════════════════════════════════════════");

// ── Rollout master state ──────────────────────────────────────────────────────
console.log("\n── Rollout Master State ────────────────────────────────");
if (!rollout) {
  console.log("  [MISSING] production_parallel_rollout.json not found");
  console.log("  Effective state: SEQUENTIAL (deny-by-default)");
} else {
  const masterState = rollout.master_enabled === true
    ? (rollout.force_sequential === true ? "FORCE-SEQUENTIAL" : "ENABLED")
    : "DISABLED";
  console.log(`  master_enabled:      ${formatBool(rollout.master_enabled)}`);
  console.log(`  force_sequential:    ${formatBool(rollout.force_sequential)}`);
  console.log(`  effective_state:     ${masterState}`);
  console.log(`  last_policy_change:  ${rollout.last_policy_change || "unknown"}`);
}

// ── Circuit-breaker state ─────────────────────────────────────────────────────
console.log("\n── Circuit-Breaker State ───────────────────────────────");
if (!rollout?.circuit_breaker) {
  console.log("  [NOT CONFIGURED]");
} else {
  const cb = rollout.circuit_breaker;
  const cbState = cb.activated === true ? "ACTIVATED (force-sequential)" : "NORMAL";
  console.log(`  state:                    ${cbState}`);
  console.log(`  rolling_window_size:      ${cb.rolling_window_size ?? "N/A"}`);
  console.log(`  partial_failure_threshold: ${cb.partial_failure_rate_threshold_pct ?? "N/A"}%`);
  console.log(`  alert_destination:        ${cb.alert_destination ?? "N/A"}`);
  if (cb.activated_at) console.log(`  activated_at:             ${cb.activated_at}`);
  if (cb.activation_reason) console.log(`  activation_reason:        ${cb.activation_reason}`);
}

// ── Eligibility policy summary ────────────────────────────────────────────────
console.log("\n── Eligibility Policy Summary ──────────────────────────");
if (!eligibility) {
  console.log("  [MISSING] parallel_exposure_policy.json not found");
} else {
  console.log(`  allowed_workflow_types:        ${(eligibility.allowed_workflow_types || []).join(", ") || "(none)"}`);
  console.log(`  allowed_project_types:         ${(eligibility.allowed_project_types || []).join(", ") || "(none)"}`);
  console.log(`  fe_safe_eligible_input_classes: ${(eligibility.fe_safe_eligible_input_classes || []).join(", ") || "(none)"}`);
  console.log("  active_deny_conditions:");
  for (const [key, val] of Object.entries(eligibility.deny_conditions || {})) {
    if (val.enabled !== false) console.log(`    - ${key}`);
  }
}

// ── Effective decision distribution (last N runs) ─────────────────────────────
console.log(`\n── Decision Distribution (last ${args.last} runs, found ${summary.total}) ──`);
if (summary.total === 0) {
  console.log("  No staging run artifacts found.");
} else {
  const parPct = ((summary.parallel / summary.total) * 100).toFixed(1);
  const seqPct = ((summary.sequential / summary.total) * 100).toFixed(1);
  console.log(`  gated_parallel_allowed: ${summary.parallel}  (${parPct}%)`);
  console.log(`  forced_sequential:      ${summary.sequential}  (${seqPct}%)`);
  console.log("  by_source:");
  for (const [src, count] of Object.entries(summary.bySource)) {
    console.log(`    ${src}: ${count}`);
  }
}

// ── Overall assessment ────────────────────────────────────────────────────────
console.log("\n── Overall Exposure Assessment ─────────────────────────");
const cbActivated = rollout?.circuit_breaker?.activated === true;
const masterEnabled = rollout?.master_enabled === true;
const forceSeq = rollout?.force_sequential === true;

if (!rollout) {
  console.log("  ASSESSMENT: SEQUENTIAL (config missing — deny-by-default)");
} else if (cbActivated) {
  console.log("  ASSESSMENT: SEQUENTIAL — circuit-breaker ACTIVATED");
  console.log("  ACTION REQUIRED: investigate trigger cause before resetting");
} else if (forceSeq) {
  console.log("  ASSESSMENT: SEQUENTIAL — force_sequential override active");
} else if (!masterEnabled) {
  console.log("  ASSESSMENT: SEQUENTIAL — rollout master disabled");
} else {
  console.log("  ASSESSMENT: GATED-PARALLEL ELIGIBLE — master enabled, eligibility policy applies");
}
console.log("═══════════════════════════════════════════════════════");
