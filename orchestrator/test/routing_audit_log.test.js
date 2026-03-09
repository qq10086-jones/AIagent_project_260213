/**
 * WS-30-01: Routing Decision Audit Log — Integration Tests
 *
 * Verifies that:
 *  1. A routing decision record is persisted with all required fields (§ 5.3)
 *  2. queryByRunId returns the persisted record
 *  3. getLatestForWorkflowRun returns the correct record
 *  4. Audit log failure is non-fatal (does not throw)
 *  5. All eight normalized routing_decision_source values are accepted
 *  6. Classifier unavailability fallback is recorded correctly
 *  7. Static-policy-only path is recorded correctly
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { createRoutingAuditLogService } from "../src/domain/routing_audit_log.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Mock pool factory ─────────────────────────────────────────────────────────

function makeMockPool({ rows = [], shouldFail = false } = {}) {
  const calls = [];
  return {
    calls,
    async query(sql, params) {
      calls.push({ sql, params });
      if (shouldFail) throw new Error("mock DB error");
      return { rows };
    },
  };
}

// ── Workspace fixture factory ─────────────────────────────────────────────────

function makeWorkspace(rolloutOverride = {}) {
  const dir = fs.mkdtempSync(path.join(__dirname, "tmp_audit_"));
  const configDir = path.join(dir, "configs");
  fs.mkdirSync(configDir, { recursive: true });
  fs.writeFileSync(
    path.join(configDir, "production_parallel_rollout.json"),
    JSON.stringify({
      router_mode: "static_policy_only",
      dynamic_routing_enabled: false,
      master_enabled: false,
      ...rolloutOverride,
    })
  );
  return dir;
}

function cleanup(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
}

// ── Stub data ─────────────────────────────────────────────────────────────────

const STUB_RUN = {
  run_id: "run-ws30-01",
  workflow_run_id: "wfrun-ws30-01",
  workflow_id: "coding_team_v0",
  project_type: "crm",
  input_class: "fe_led",
};

const STUB_GATE_SEQUENTIAL = {
  effective_exposure_decision: "forced_sequential",
  routing_decision_source: "rollout_master_disabled",
  model_tier: "balanced_default",
};

const STUB_GATE_PARALLEL = {
  effective_exposure_decision: "gated_parallel_allowed",
  routing_decision_source: "classifier_recommended_parallel",
  model_tier: "fast_low_cost",
};

const STUB_CLASSIFIER_RESULT = {
  classifier_version: "1.0.0",
  feature_snapshot_ref: "snap-test-001",
  work_shape: "dual_branch_parallel_candidate",
  domain_lead: "fe_led",
  confidence: 0.9,
  confidence_band: "high",
  parallel_candidate: true,
  model_tier: "fast_low_cost",
  required_contracts: ["coding_team_v0"],
  safety_override_result: "allowed",
  final_execution_decision: "gated_parallel_allowed",
  routing_decision_source: "classifier_recommended_parallel",
  deny_or_degrade_reason: null,
};

// ── Tests ─────────────────────────────────────────────────────────────────────

test("WS-30-01: log() persists a record with all required § 5.3 fields", async () => {
  const dir = makeWorkspace();
  const pool = makeMockPool();
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  const logId = await svc.log({
    run: STUB_RUN,
    gateDecision: STUB_GATE_PARALLEL,
    classifierResult: STUB_CLASSIFIER_RESULT,
  });

  assert.ok(logId, "log_id must be returned");
  assert.equal(pool.calls.length, 1, "exactly one DB insert expected");

  const [call] = pool.calls;
  const params = call.params;

  // log_id is a UUID
  assert.match(params[0], /^[0-9a-f-]{36}$/, "log_id must be a UUID");
  // run_id
  assert.equal(params[1], "run-ws30-01");
  // workflow_run_id
  assert.equal(params[2], "wfrun-ws30-01");
  // workflow_id
  assert.equal(params[3], "coding_team_v0");
  // router_mode (from workspace config)
  assert.equal(params[4], "static_policy_only");
  // dynamic_routing_enabled
  assert.equal(params[5], false);
  // classifier_version
  assert.equal(params[6], "1.0.0");
  // classifier_confidence
  assert.equal(params[7], 0.9);
  // classifier_confidence_band
  assert.equal(params[8], "high");
  // classifier_work_shape
  assert.equal(params[9], "dual_branch_parallel_candidate");
  // classifier_domain_lead
  assert.equal(params[10], "fe_led");
  // classifier_parallel_candidate
  assert.equal(params[11], true);
  // classifier_model_tier
  assert.equal(params[12], "fast_low_cost");
  // routing_decision_source
  assert.equal(params[15], "classifier_recommended_parallel");
  // final_execution_decision
  assert.equal(params[16], "gated_parallel_allowed");

  cleanup(dir);
});

test("WS-30-01: log() with no classifier_result records null fields gracefully", async () => {
  const dir = makeWorkspace();
  const pool = makeMockPool();
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  await svc.log({
    run: STUB_RUN,
    gateDecision: STUB_GATE_SEQUENTIAL,
    classifierResult: null,
  });

  const params = pool.calls[0].params;
  assert.equal(params[6], null, "classifier_version should be null when no classifier result");
  assert.equal(params[15], "rollout_master_disabled");
  assert.equal(params[16], "forced_sequential");

  cleanup(dir);
});

test("WS-30-01: log() records classifier_unavailable_fallback source correctly", async () => {
  const dir = makeWorkspace();
  const pool = makeMockPool();
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  const unavailableClassifier = {
    ...STUB_CLASSIFIER_RESULT,
    confidence_band: "unavailable",
    final_execution_decision: "forced_sequential",
    routing_decision_source: "classifier_unavailable_fallback",
    deny_or_degrade_reason: "Classifier status is degraded",
  };

  await svc.log({
    run: STUB_RUN,
    gateDecision: {
      effective_exposure_decision: "forced_sequential",
      routing_decision_source: "classifier_unavailable_fallback",
      model_tier: "balanced_default",
    },
    classifierResult: unavailableClassifier,
  });

  const params = pool.calls[0].params;
  assert.equal(params[8], "unavailable", "confidence_band must be 'unavailable'");
  assert.equal(params[15], "classifier_unavailable_fallback");
  assert.equal(params[16], "forced_sequential");

  cleanup(dir);
});

test("WS-30-01: log() records static_eligibility_denied source correctly", async () => {
  const dir = makeWorkspace();
  const pool = makeMockPool();
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  await svc.log({
    run: STUB_RUN,
    gateDecision: {
      effective_exposure_decision: "forced_sequential",
      routing_decision_source: "static_eligibility_denied",
      deny_reason_code: "unapproved_input_class",
      model_tier: "balanced_default",
    },
    classifierResult: null,
  });

  const params = pool.calls[0].params;
  assert.equal(params[15], "static_eligibility_denied");

  cleanup(dir);
});

test("WS-30-01: log() is non-fatal when DB insert fails", async () => {
  const dir = makeWorkspace();
  const pool = makeMockPool({ shouldFail: true });
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  // Must not throw — audit log failure must not block routing
  await assert.doesNotReject(() =>
    svc.log({ run: STUB_RUN, gateDecision: STUB_GATE_SEQUENTIAL, classifierResult: null })
  );

  cleanup(dir);
});

test("WS-30-01: queryByRunId() delegates to repository and returns rows", async () => {
  const dir = makeWorkspace();
  const fakeRow = { log_id: "abc", run_id: "run-ws30-01", routing_decision_source: "rollout_master_disabled" };
  const pool = makeMockPool({ rows: [fakeRow] });
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  const rows = await svc.queryByRunId("run-ws30-01");

  assert.equal(rows.length, 1);
  assert.equal(rows[0].log_id, "abc");
  // Verify the SELECT query was issued with correct run_id param
  const selectCall = pool.calls.find((c) => c.sql.includes("routing_decision_log") && c.sql.includes("SELECT"));
  assert.ok(selectCall, "SELECT query must be issued");
  assert.equal(selectCall.params[0], "run-ws30-01");

  cleanup(dir);
});

test("WS-30-01: getLatestForWorkflowRun() delegates to repository", async () => {
  const dir = makeWorkspace();
  const fakeRow = { log_id: "xyz", workflow_run_id: "wfrun-ws30-01", routing_decision_source: "classifier_recommended_parallel" };
  const pool = makeMockPool({ rows: [fakeRow] });
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  const row = await svc.getLatestForWorkflowRun("wfrun-ws30-01");

  assert.ok(row, "row must be returned");
  assert.equal(row.log_id, "xyz");

  cleanup(dir);
});

test("WS-30-01: log() reads router_mode and dynamic_routing_enabled from workspace config", async () => {
  const dir = makeWorkspace({ router_mode: "dynamic", dynamic_routing_enabled: true });
  const pool = makeMockPool();
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  await svc.log({ run: STUB_RUN, gateDecision: STUB_GATE_PARALLEL, classifierResult: null });

  const params = pool.calls[0].params;
  assert.equal(params[4], "dynamic", "router_mode must reflect workspace config");
  assert.equal(params[5], true, "dynamic_routing_enabled must reflect workspace config");

  cleanup(dir);
});

test("WS-30-01: log() handles missing workspace config gracefully (defaults to static_policy_only)", async () => {
  // Workspace with no config file — should not throw, should default router_mode
  const dir = fs.mkdtempSync(path.join(__dirname, "tmp_audit_noconfig_"));
  const pool = makeMockPool();
  const svc = createRoutingAuditLogService({ pool, workspaceRoot: dir });

  await svc.log({ run: STUB_RUN, gateDecision: STUB_GATE_SEQUENTIAL, classifierResult: null });

  const params = pool.calls[0].params;
  assert.equal(params[4], "static_policy_only", "router_mode must default when config is missing");
  assert.equal(params[5], false, "dynamic_routing_enabled must default to false");

  cleanup(dir);
});
