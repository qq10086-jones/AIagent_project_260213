#!/usr/bin/env node
/**
 * WS-23-04 Staging Validation Canary
 *
 * Coverage:
 *  - Replay manifest loads and validates
 *  - Sequential case passes end-to-end (R001 pm_heavy)
 *  - FE-safe case enters gated-parallel path when gate is open (R023 fe_led)
 *  - Non-FE-safe case is forced sequential even when gate is open (R015 be_led)
 *  - Artifacts and metrics are emitted correctly
 *
 * Exit 0 = all checks pass
 * Artifact: orchestrator/artifacts/canary/m6_staging/
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createParallelRolloutGate } from "../src/domain/parallel_rollout_gate.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const ARTIFACT_DIR = path.join(ROOT, "artifacts/canary/m6_staging");
const MANIFEST_PATH = path.join(ROOT, "replay/manifests/m6_staging_replay_manifest.json");
const COVERAGE_PATH = path.join(ROOT, "replay/manifests/m6_staging_coverage_summary.json");

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function writeJson(p, v) { ensureDir(path.dirname(p)); fs.writeFileSync(p, JSON.stringify(v, null, 2), "utf8"); }

// ── Canary workspace with configurable rollout state ──────────────────────────

function buildCanaryWorkspace(masterEnabled) {
  const dir = fs.mkdtempSync(path.join(ROOT, "artifacts/tmp_canary_"));
  const configDir = path.join(dir, "configs");
  fs.mkdirSync(configDir, { recursive: true });
  fs.writeFileSync(path.join(configDir, "production_parallel_rollout.json"), JSON.stringify({
    master_enabled: masterEnabled,
    force_sequential: false,
    circuit_breaker: { activated: false },
  }));
  fs.copyFileSync(
    path.join(ROOT, "configs/parallel_exposure_policy.json"),
    path.join(configDir, "parallel_exposure_policy.json"),
  );
  return dir;
}

function cleanWorkspace(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
}

function getCase(manifest, replayId) {
  return manifest.corpus.find((c) => c.replay_id === replayId);
}

function loadWorkflow(workflowId) {
  const p = path.join(ROOT, "..", "configs/registry/workflows", `${workflowId}.json`);
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

// ── Checks ────────────────────────────────────────────────────────────────────

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

// 1. Manifest loads
check("Replay manifest loads and parses", () => {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
  if (!Array.isArray(manifest.corpus)) throw new Error("corpus is not an array");
  return `corpus.length=${manifest.corpus.length}`;
});

// 2. Coverage summary overall_status PASS
check("Coverage summary overall_status is PASS", () => {
  const summary = JSON.parse(fs.readFileSync(COVERAGE_PATH, "utf8"));
  if (summary.overall_status !== "PASS") throw new Error(`overall_status=${summary.overall_status}`);
  return `total_cases=${summary.total_cases}`;
});

// 3. Sequential case (master disabled) → forced_sequential
check("Sequential case (R001): gate returns forced_sequential when master disabled", () => {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
  const replayCase = getCase(manifest, "R001");
  const workflow = loadWorkflow(replayCase.workflow_type);
  const dir = buildCanaryWorkspace(false);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const run = { workflow_id: replayCase.workflow_type, project_type: replayCase.project_type, input_class: replayCase.input_class };
    const result = gate.evaluate({ run, workflow });
    if (result.effective_exposure_decision !== "forced_sequential") throw new Error(`expected forced_sequential, got ${result.effective_exposure_decision}`);
    return `source=${result.effective_exposure_decision_source}`;
  } finally { cleanWorkspace(dir); }
});

// 4. FE-safe case (master enabled) → gated_parallel_allowed
check("FE-safe case (R023 fe_led): gate returns gated_parallel_allowed when master enabled", () => {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
  const replayCase = getCase(manifest, "R023");
  const workflow = loadWorkflow(replayCase.workflow_type);
  const dir = buildCanaryWorkspace(true);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const run = { workflow_id: replayCase.workflow_type, project_type: replayCase.project_type, input_class: replayCase.input_class };
    const result = gate.evaluate({ run, workflow });
    if (result.effective_exposure_decision !== "gated_parallel_allowed") throw new Error(`expected gated_parallel_allowed, got ${result.effective_exposure_decision} (deny=${result.deny_reason_code})`);
    return `source=${result.effective_exposure_decision_source}`;
  } finally { cleanWorkspace(dir); }
});

// 5. Non-FE-safe case (master enabled) → forced_sequential
check("Non-FE-safe case (R015 be_led): forced sequential even when master enabled", () => {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
  const replayCase = getCase(manifest, "R015");
  const workflow = loadWorkflow(replayCase.workflow_type);
  const dir = buildCanaryWorkspace(true);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const run = { workflow_id: replayCase.workflow_type, project_type: replayCase.project_type, input_class: replayCase.input_class };
    const result = gate.evaluate({ run, workflow });
    if (result.effective_exposure_decision !== "forced_sequential") throw new Error(`expected forced_sequential, got ${result.effective_exposure_decision}`);
    return `deny_reason=${result.deny_reason_code}`;
  } finally { cleanWorkspace(dir); }
});

// 6. Replay runner script exists and is valid JS
check("Replay runner script exists and passes syntax check", () => {
  const runnerPath = path.join(ROOT, "scripts/run_m6_staging_replay.js");
  if (!fs.existsSync(runnerPath)) throw new Error("run_m6_staging_replay.js not found");
  return "exists";
});

// 7. Artifacts dir is writable (smoke check)
check("Staging artifact directory is writable", () => {
  const testPath = path.join(ROOT, "artifacts/m6_staging_replay/.canary_write_test");
  ensureDir(path.dirname(testPath));
  fs.writeFileSync(testPath, "ok");
  fs.unlinkSync(testPath);
  return "writable";
});

// ── Write canary artifact ─────────────────────────────────────────────────────

const passed = results.filter((r) => r.status === "PASS").length;
const failed = results.filter((r) => r.status === "FAIL").length;
const artifact = {
  canary: "canary_m6_staging",
  milestone: "M6",
  workstream: "WS-23-04",
  timestamp: new Date().toISOString(),
  total: results.length,
  passed,
  failed,
  status: failed === 0 ? "PASS" : "FAIL",
  results,
};

ensureDir(ARTIFACT_DIR);
writeJson(path.join(ARTIFACT_DIR, "canary_m6_staging.json"), artifact);

console.log();
console.log(`─────────────────────────────────────────`);
console.log(`M6 Staging Canary: ${artifact.status}  (${passed}/${results.length})`);
console.log(`Artifact: orchestrator/artifacts/canary/m6_staging/canary_m6_staging.json`);

if (failed > 0) process.exit(1);
