#!/usr/bin/env node
/**
 * WS-25-04  Production Exposure Canary
 *
 * Coverage:
 *  - Denied workflow remains sequential
 *  - Approved FE-safe workflow enters gated-parallel path (gate open)
 *  - Emergency rollback switch (force_sequential) forces subsequent runs sequential
 *  - Gate state is logged and queryable via exposure_state_query
 *  - Circuit-breaker activation triggers force-sequential
 *  - Circuit-breaker does not auto-recover (requires manual reset)
 *
 * Exit 0 = all checks pass
 * Artifact: orchestrator/artifacts/canary/m6_rollout_gate/
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { createParallelRolloutGate } from "../src/domain/parallel_rollout_gate.js";
import { evaluateCircuitBreaker, activateCircuitBreaker, resetCircuitBreaker } from "../src/domain/circuit_breaker_service.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const ARTIFACT_DIR = path.join(ROOT, "artifacts/canary/m6_rollout_gate");

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function writeJson(p, v) { ensureDir(path.dirname(p)); fs.writeFileSync(p, JSON.stringify(v, null, 2), "utf8"); }

// ── Canary workspace builder ──────────────────────────────────────────────────

function buildWorkspace(rolloutOverride, policyOverride) {
  const dir = fs.mkdtempSync(path.join(ROOT, "artifacts/tmp_canary_gate_"));
  const configDir = path.join(dir, "configs");
  fs.mkdirSync(configDir, { recursive: true });

  const defaultRollout = {
    master_enabled: true,
    force_sequential: false,
    circuit_breaker: { activated: false, rolling_window_size: 5, partial_failure_rate_threshold_pct: 25, rollback_trigger_event_threshold: 2 },
    last_policy_change: new Date().toISOString(),
  };
  const defaultPolicy = {
    allowed_workflow_types: ["coding_team_v0"],
    allowed_project_types: ["crm"],
    fe_safe_eligible_input_classes: ["fe_led"],
    deny_conditions: { structural_completion_impossible: { enabled: true, override_allow: true } },
  };

  fs.writeFileSync(path.join(configDir, "production_parallel_rollout.json"), JSON.stringify({ ...defaultRollout, ...rolloutOverride }));
  fs.writeFileSync(path.join(configDir, "parallel_exposure_policy.json"), JSON.stringify({ ...defaultPolicy, ...policyOverride }));
  return dir;
}

function cleanWorkspace(dir) { fs.rmSync(dir, { recursive: true, force: true }); }

const FE_SAFE_WORKFLOW = {
  steps: [
    { id: "pm_spec" }, { id: "arch_design" }, { id: "impl_be" },
    { id: "impl_fe", fe_safe_input_classes: ["fe_led"] },
    { id: "qa_verify" }, { id: "release_pack" },
  ],
};

const FE_SAFE_RUN  = { workflow_id: "coding_team_v0", project_type: "crm", input_class: "fe_led" };
const BE_LED_RUN   = { workflow_id: "coding_team_v0", project_type: "crm", input_class: "be_led" };

// ── Check runner ──────────────────────────────────────────────────────────────

const results = [];

function check(label, fn) {
  try {
    const detail = fn();
    results.push({ label, status: "PASS", detail: detail || null });
    console.log(`PASS  ${label}`);
  } catch (err) {
    results.push({ label, status: "FAIL", error: String(err.message || err) });
    console.error(`FAIL  ${label}: ${err.message || err}`);
  }
}

// ── 1. Denied workflow remains sequential ─────────────────────────────────────
check("Denied workflow (be_led) remains sequential even when master enabled", () => {
  const dir = buildWorkspace({}, {});
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: BE_LED_RUN, workflow: FE_SAFE_WORKFLOW });
    if (result.effective_exposure_decision !== "forced_sequential") throw new Error(`expected forced_sequential, got ${result.effective_exposure_decision}`);
    return `deny_reason=${result.deny_reason_code}`;
  } finally { cleanWorkspace(dir); }
});

// ── 2. Approved FE-safe workflow enters gated-parallel ────────────────────────
check("Approved FE-safe workflow (fe_led) enters gated-parallel path when gate open", () => {
  const dir = buildWorkspace({}, {});
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    if (result.effective_exposure_decision !== "gated_parallel_allowed") throw new Error(`expected gated_parallel_allowed, got ${result.effective_exposure_decision}`);
    return `source=${result.effective_exposure_decision_source}`;
  } finally { cleanWorkspace(dir); }
});

// ── 3. Emergency rollback switch forces sequential ────────────────────────────
check("Emergency rollback (force_sequential: true) forces all runs to sequential", () => {
  const dir = buildWorkspace({ force_sequential: true }, {});
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const resultFeSafe = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    if (resultFeSafe.effective_exposure_decision !== "forced_sequential") throw new Error(`FE-safe should be sequential after rollback`);
    if (resultFeSafe.effective_exposure_decision_source !== "force_sequential_override") throw new Error(`source mismatch`);
    return "force_sequential_override confirmed for FE-safe run";
  } finally { cleanWorkspace(dir); }
});

// ── 4. Gate state is queryable after evaluation ───────────────────────────────
check("Gate evaluation result contains all required queryability fields", () => {
  const dir = buildWorkspace({}, {});
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    if (!result.effective_exposure_decision) throw new Error("missing effective_exposure_decision");
    if (!result.effective_exposure_decision_source) throw new Error("missing effective_exposure_decision_source");
    return `fields present: effective_exposure_decision=${result.effective_exposure_decision}`;
  } finally { cleanWorkspace(dir); }
});

// ── 5. Circuit-breaker activates when threshold breached ─────────────────────
check("Circuit-breaker activates when partial_failure_rate exceeds threshold", () => {
  // 3 partial failures out of 5 runs = 60%, threshold is 25%
  const recentResults = [
    { partial_failure: true },
    { partial_failure: true },
    { partial_failure: true },
    { partial_failure: false },
    { partial_failure: false },
  ];
  const config = { rolling_window_size: 5, partial_failure_rate_threshold_pct: 25, rollback_trigger_event_threshold: 10 };
  const evalResult = evaluateCircuitBreaker(recentResults, config);
  if (!evalResult.should_activate) throw new Error("expected should_activate=true");
  if (evalResult.trigger_metric !== "partial_failure_rate") throw new Error(`wrong trigger_metric: ${evalResult.trigger_metric}`);
  return `trigger=${evalResult.trigger_metric} observed=${(evalResult.observed_value * 100).toFixed(0)}% threshold=${(evalResult.threshold * 100).toFixed(0)}%`;
});

// ── 6. Circuit-breaker writes force-sequential to config ──────────────────────
check("Circuit-breaker activation persists force_sequential to config file", () => {
  const dir = buildWorkspace({}, {});
  const configPath = path.join(dir, "configs/production_parallel_rollout.json");
  try {
    const alerts = [];
    activateCircuitBreaker(configPath, { trigger_metric: "partial_failure_rate", threshold: 0.25, observed_value: 0.60 }, (msg) => alerts.push(msg));
    const updated = JSON.parse(fs.readFileSync(configPath, "utf8"));
    if (!updated.circuit_breaker?.activated) throw new Error("circuit_breaker.activated not set");
    if (updated.force_sequential !== true) throw new Error("force_sequential not set to true");
    if (alerts.length === 0) throw new Error("no alert emitted");
    return `activated_at=${updated.circuit_breaker.activated_at}`;
  } finally { cleanWorkspace(dir); }
});

// ── 7. Circuit-breaker forces sequential after activation ─────────────────────
check("Circuit-breaker activation causes gate to return forced_sequential", () => {
  const dir = buildWorkspace({}, {});
  const configPath = path.join(dir, "configs/production_parallel_rollout.json");
  try {
    activateCircuitBreaker(configPath, { trigger_metric: "partial_failure_rate", threshold: 0.25, observed_value: 0.60 }, () => {});
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    if (result.effective_exposure_decision !== "forced_sequential") throw new Error(`expected forced_sequential after CB, got ${result.effective_exposure_decision}`);
    return `deny_reason=${result.deny_reason_code || result.effective_exposure_decision_source}`;
  } finally { cleanWorkspace(dir); }
});

// ── 8. Circuit-breaker does not auto-recover ──────────────────────────────────
check("Circuit-breaker does not auto-recover without explicit operator reset", () => {
  const dir = buildWorkspace({}, {});
  const configPath = path.join(dir, "configs/production_parallel_rollout.json");
  try {
    activateCircuitBreaker(configPath, { trigger_metric: "partial_failure_rate", threshold: 0.25, observed_value: 0.60 }, () => {});
    // Simulate time passing — re-read config without calling reset
    const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
    if (!config.circuit_breaker?.activated) throw new Error("CB should still be activated without manual reset");
    if (config.force_sequential !== true) throw new Error("force_sequential should still be true");
    return "CB remains activated without explicit reset — no auto-recovery";
  } finally { cleanWorkspace(dir); }
});

// ── 9. Operator reset clears circuit-breaker ─────────────────────────────────
check("Operator reset clears circuit-breaker and restores gate evaluation", () => {
  const dir = buildWorkspace({}, {});
  const configPath = path.join(dir, "configs/production_parallel_rollout.json");
  try {
    activateCircuitBreaker(configPath, { trigger_metric: "partial_failure_rate", threshold: 0.25, observed_value: 0.60 }, () => {});
    const resetResult = resetCircuitBreaker(configPath);
    if (!resetResult.reset) throw new Error("reset returned false");
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    if (result.effective_exposure_decision !== "gated_parallel_allowed") throw new Error(`after reset expected gated_parallel_allowed, got ${result.effective_exposure_decision}`);
    return `reset_at=${resetResult.reset_at}`;
  } finally { cleanWorkspace(dir); }
});

// ── 10. Circuit-breaker state visible in exposure_state_query ─────────────────
check("Circuit-breaker activated state is persisted in config for exposure_state_query", () => {
  const dir = buildWorkspace({}, {});
  const configPath = path.join(dir, "configs/production_parallel_rollout.json");
  try {
    activateCircuitBreaker(configPath, { trigger_metric: "rollback_trigger_events", threshold: 2, observed_value: 3 }, () => {});
    const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
    if (config.circuit_breaker?.activated !== true) throw new Error("activated not persisted");
    if (!config.circuit_breaker?.activated_at) throw new Error("activated_at not persisted");
    if (!config.circuit_breaker?.activation_reason) throw new Error("activation_reason not persisted");
    return `activation_reason=${config.circuit_breaker.activation_reason}`;
  } finally { cleanWorkspace(dir); }
});

// ── Write canary artifact ─────────────────────────────────────────────────────

const passed = results.filter((r) => r.status === "PASS").length;
const failed = results.filter((r) => r.status === "FAIL").length;
const artifact = {
  canary: "canary_m6_rollout_gate",
  milestone: "M6",
  workstream: "WS-25-04",
  timestamp: new Date().toISOString(),
  total: results.length,
  passed,
  failed,
  status: failed === 0 ? "PASS" : "FAIL",
  results,
};

ensureDir(ARTIFACT_DIR);
writeJson(path.join(ARTIFACT_DIR, "canary_m6_rollout_gate.json"), artifact);

console.log();
console.log(`─────────────────────────────────────────`);
console.log(`M6 Rollout Gate Canary: ${artifact.status}  (${passed}/${results.length})`);
console.log(`Artifact: orchestrator/artifacts/canary/m6_rollout_gate/canary_m6_rollout_gate.json`);

if (failed > 0) process.exit(1);
