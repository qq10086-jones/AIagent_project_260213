/**
 * WS-30-03: Routing Evaluation Report — Integration Tests
 *
 * Verifies that:
 *  1. generateReport() returns all six WS-30-03 dimensions
 *  2. routing_precision is computed correctly
 *  3. high_risk_misroute_rate is computed correctly
 *  4. low_confidence_fallback_ratio is computed correctly
 *  5. forced_sequential_ratio is computed correctly
 *  6. latency_stats populated when waterfallTraceService is provided
 *  7. latency_stats is null without waterfallTraceService
 *  8. counterfactual_comparison is null when no manifest path is given
 *  9. counterfactual_comparison is built from manifest when path is given
 * 10. static decision reads from policy file (not hardcoded), fe_led only → 2 parallel
 * 11. empty dataset returns null ratios (no division by zero)
 * 12. generateReport() rejects when DB query fails
 * 13. breakdown tables are included in report output
 * 14. factory throws when workspaceRoot is missing
 */

import test from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

import { createRoutingEvaluationReportService } from "../src/domain/routing_evaluation_report.js";

// ── Mock pool factory ─────────────────────────────────────────────────────────

function makeMockPool({ shouldFail = false, rows = [] } = {}) {
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

// ── Mock waterfall trace service ──────────────────────────────────────────────

function makeMockWaterfallService(p50Map = {}) {
  return {
    async queryP50P95(stage) {
      const p50 = p50Map[stage] ?? 0;
      return { stage, p50_ms: p50, p95_ms: p50 * 1.5, sample_size: 10 };
    },
  };
}

// ── Aggregate stat row builder ────────────────────────────────────────────────

function makeStatRow({
  routing_decision_source = "dynamic_routing_disabled",
  final_execution_decision = "forced_sequential",
  classifier_confidence_band = null,
  classifier_work_shape = null,
  classifier_model_tier = "balanced_default",
  count = 1,
} = {}) {
  return {
    routing_decision_source,
    final_execution_decision,
    classifier_confidence_band,
    classifier_work_shape,
    classifier_model_tier,
    count,
  };
}

// ── Workspace + manifest fixture builder ──────────────────────────────────────

/**
 * Creates a temp workspace with the standard eligibility policy (fe_led only).
 * Returns workspaceRoot.
 */
function makeWorkspace(dir, policyOverride = null) {
  const configsDir = join(dir, "configs");
  mkdirSync(configsDir, { recursive: true });
  const policy = policyOverride ?? {
    allowed_workflow_types:        ["coding_team_v0"],
    allowed_project_types:         ["crm"],
    fe_safe_eligible_input_classes: ["fe_led"],
  };
  writeFileSync(join(configsDir, "parallel_exposure_policy.json"), JSON.stringify(policy), "utf8");
  return dir;
}

function writeManifest(dir, corpus) {
  const path = join(dir, "manifest.json");
  writeFileSync(
    path,
    JSON.stringify({ manifest_id: "test-manifest-v1", corpus }),
    "utf8"
  );
  return path;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test("WS-30-03: generateReport() returns all six required dimensions", async () => {
  const dir  = join(tmpdir(), `ws30-03-dims-${Date.now()}`);
  const root = makeWorkspace(dir);
  const pool = makeMockPool({ rows: [] });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.ok("routing_precision"             in report, "routing_precision");
  assert.ok("high_risk_misroute_rate"       in report, "high_risk_misroute_rate");
  assert.ok("low_confidence_fallback_ratio" in report, "low_confidence_fallback_ratio");
  assert.ok("forced_sequential_ratio"       in report, "forced_sequential_ratio");
  assert.ok("latency_stats"                in report, "latency_stats");
  assert.ok("incident_delta"               in report, "incident_delta");
  assert.ok("generated_at"                in report, "generated_at");
  assert.ok("counterfactual_comparison"    in report, "counterfactual_comparison");
});

test("WS-30-03: routing_precision is gated_parallel_allowed / total", async () => {
  const dir  = join(tmpdir(), `ws30-03-prec-${Date.now()}`);
  const root = makeWorkspace(dir);
  const rows = [
    makeStatRow({ final_execution_decision: "gated_parallel_allowed", count: 30 }),
    makeStatRow({ final_execution_decision: "forced_sequential",       count: 70 }),
  ];
  const pool = makeMockPool({ rows });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.equal(report.total_runs, 100);
  assert.equal(report.routing_precision, 0.3);
});

test("WS-30-03: high_risk_misroute_rate counts high_risk_release_sensitive + gated_parallel_allowed", async () => {
  const dir  = join(tmpdir(), `ws30-03-misroute-${Date.now()}`);
  const root = makeWorkspace(dir);
  const rows = [
    makeStatRow({ classifier_work_shape: "high_risk_release_sensitive", final_execution_decision: "gated_parallel_allowed", count: 2 }),
    makeStatRow({ classifier_work_shape: "dual_branch_parallel_candidate", final_execution_decision: "gated_parallel_allowed", count: 18 }),
    makeStatRow({ final_execution_decision: "forced_sequential", count: 80 }),
  ];
  const pool = makeMockPool({ rows });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.equal(report.high_risk_misroute_count, 2);
  assert.equal(report.high_risk_misroute_rate, 0.02);
});

test("WS-30-03: low_confidence_fallback_ratio counts band='low' rows", async () => {
  const dir  = join(tmpdir(), `ws30-03-lowconf-${Date.now()}`);
  const root = makeWorkspace(dir);
  const rows = [
    makeStatRow({ classifier_confidence_band: "low",  count: 15 }),
    makeStatRow({ classifier_confidence_band: "high", count: 85 }),
  ];
  const pool = makeMockPool({ rows });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.equal(report.low_confidence_fallback_count, 15);
  assert.equal(report.low_confidence_fallback_ratio, 0.15);
});

test("WS-30-03: forced_sequential_ratio computed correctly", async () => {
  const dir  = join(tmpdir(), `ws30-03-fseq-${Date.now()}`);
  const root = makeWorkspace(dir);
  const rows = [
    makeStatRow({ final_execution_decision: "gated_parallel_allowed", count: 40 }),
    makeStatRow({ final_execution_decision: "forced_sequential",       count: 60 }),
  ];
  const pool = makeMockPool({ rows });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.equal(report.forced_sequential_count, 60);
  assert.equal(report.forced_sequential_ratio, 0.6);
});

test("WS-30-03: latency_stats populated when waterfallTraceService provided", async () => {
  const dir  = join(tmpdir(), `ws30-03-lat-${Date.now()}`);
  const root = makeWorkspace(dir);
  const pool  = makeMockPool({ rows: [] });
  const wfSvc = makeMockWaterfallService({ intake: 5, routing: 45, policy_evaluation: 8, execution_dispatch: 200 });
  const svc   = createRoutingEvaluationReportService({ pool, workspaceRoot: root, waterfallTraceService: wfSvc });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.ok(report.latency_stats !== null,           "latency_stats must be set");
  assert.equal(report.latency_stats.routing_overhead_p50_ms, 58); // 5+45+8
  assert.equal(report.latency_stats.intake.p50_ms, 5);
  assert.equal(report.latency_stats.routing.p50_ms, 45);
});

test("WS-30-03: latency_stats is null when no waterfallTraceService provided", async () => {
  const dir  = join(tmpdir(), `ws30-03-nolat-${Date.now()}`);
  const root = makeWorkspace(dir);
  const pool = makeMockPool({ rows: [] });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.equal(report.latency_stats, null);
});

test("WS-30-03: counterfactual_comparison is null when no replayManifestPath given", async () => {
  const dir  = join(tmpdir(), `ws30-03-nocc-${Date.now()}`);
  const root = makeWorkspace(dir);
  const pool = makeMockPool({ rows: [] });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.equal(report.counterfactual_comparison, null);
});

test("WS-30-03: counterfactual_comparison builds from manifest when path provided", async () => {
  const dir = join(tmpdir(), `ws30-03-cc-${Date.now()}`);
  const root = makeWorkspace(dir);
  const corpus = [
    { replay_id: "R001", input_class: "pm_heavy",    workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: false },
    { replay_id: "R002", input_class: "fe_led",       workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: true  },
    { replay_id: "R003", input_class: "be_fe_simple", workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: true  },
  ];
  const manifestPath = writeManifest(dir, corpus);

  const pool = makeMockPool({ rows: [] });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({ replayManifestPath: manifestPath });
  rmSync(dir, { recursive: true, force: true });

  const cc = report.counterfactual_comparison;
  assert.ok(cc !== null,           "counterfactual_comparison must be populated");
  assert.equal(cc.total_cases, 3, "total_cases");
  assert.ok(Array.isArray(cc.cases), "cases must be array");
});

test("WS-30-03: static decision reads from policy file — fe_led only admits 2, dynamic admits 3", async () => {
  const dir  = join(tmpdir(), `ws30-03-static-${Date.now()}`);
  const root = makeWorkspace(dir); // policy: fe_safe_eligible_input_classes = ["fe_led"]
  const corpus = [
    { replay_id: "R001", input_class: "fe_led",       workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: true  },
    { replay_id: "R002", input_class: "pm_heavy",     workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: false },
    { replay_id: "R003", input_class: "fe_led",       workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: true  },
    { replay_id: "R004", input_class: "be_fe_simple", workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: true  },
  ];
  const manifestPath = writeManifest(dir, corpus);

  const pool = makeMockPool({ rows: [] });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({ replayManifestPath: manifestPath });
  rmSync(dir, { recursive: true, force: true });

  const cc = report.counterfactual_comparison;
  // Static: only R001, R003 (fe_led) = 2 parallel
  // Dynamic: R001, R003, R004 (fe_parallel_eligible_expected=true) = 3 parallel
  assert.equal(cc.static_gated_parallel,  2, "static_gated_parallel = 2 (fe_led only)");
  assert.equal(cc.dynamic_gated_parallel, 3, "dynamic_gated_parallel = 3");
  assert.equal(cc.dynamic_uplift,         1, "dynamic_uplift = 1");
});

test("WS-30-03: static decision respects policy change — adding be_fe_simple to eligible classes", async () => {
  const dir  = join(tmpdir(), `ws30-03-policy-change-${Date.now()}`);
  // Expanded policy: both fe_led AND be_fe_simple eligible
  const root = makeWorkspace(dir, {
    allowed_workflow_types:        ["coding_team_v0"],
    allowed_project_types:         ["crm"],
    fe_safe_eligible_input_classes: ["fe_led", "be_fe_simple"],
  });
  const corpus = [
    { replay_id: "R001", input_class: "fe_led",       workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: true  },
    { replay_id: "R002", input_class: "be_fe_simple", workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: true  },
    { replay_id: "R003", input_class: "pm_heavy",     workflow_type: "coding_team_v0", project_type: "crm", fe_parallel_eligible_expected: false },
  ];
  const manifestPath = writeManifest(dir, corpus);

  const pool = makeMockPool({ rows: [] });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({ replayManifestPath: manifestPath });
  rmSync(dir, { recursive: true, force: true });

  const cc = report.counterfactual_comparison;
  // With expanded policy, static now admits both R001 and R002 = 2
  assert.equal(cc.static_gated_parallel,  2, "expanded policy: static admits both fe_led and be_fe_simple");
  assert.equal(cc.dynamic_gated_parallel, 2, "dynamic also admits same 2");
});

test("WS-30-03: empty dataset returns null ratios (no division by zero)", async () => {
  const dir  = join(tmpdir(), `ws30-03-empty-${Date.now()}`);
  const root = makeWorkspace(dir);
  const pool = makeMockPool({ rows: [] });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.equal(report.total_runs,                    0);
  assert.equal(report.routing_precision,             null);
  assert.equal(report.high_risk_misroute_rate,       null);
  assert.equal(report.low_confidence_fallback_ratio, null);
  assert.equal(report.forced_sequential_ratio,       null);
});

test("WS-30-03: generateReport() rejects when DB query fails", async () => {
  const dir  = join(tmpdir(), `ws30-03-fail-${Date.now()}`);
  const root = makeWorkspace(dir);
  const pool = makeMockPool({ shouldFail: true });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  await assert.rejects(() => svc.generateReport({}), /mock DB error/);
  rmSync(dir, { recursive: true, force: true });
});

test("WS-30-03: breakdown tables are included in report output", async () => {
  const dir  = join(tmpdir(), `ws30-03-breakdown-${Date.now()}`);
  const root = makeWorkspace(dir);
  const rows = [
    makeStatRow({ routing_decision_source: "dynamic_routing_disabled",      classifier_work_shape: "single_branch_safe", count: 10 }),
    makeStatRow({ routing_decision_source: "classifier_recommended_parallel", classifier_work_shape: "dual_branch_parallel_candidate", final_execution_decision: "gated_parallel_allowed", count: 5 }),
  ];
  const pool = makeMockPool({ rows });
  const svc  = createRoutingEvaluationReportService({ pool, workspaceRoot: root });

  const report = await svc.generateReport({});
  rmSync(dir, { recursive: true, force: true });

  assert.equal(typeof report.by_decision_source, "object", "by_decision_source");
  assert.equal(typeof report.by_work_shape,       "object", "by_work_shape");
  assert.equal(typeof report.by_confidence_band,  "object", "by_confidence_band");
  assert.equal(typeof report.by_model_tier,       "object", "by_model_tier");
  assert.equal(report.by_decision_source["dynamic_routing_disabled"], 10);
  assert.equal(report.by_work_shape["single_branch_safe"], 10);
});

test("WS-30-03: factory throws when workspaceRoot is missing", () => {
  const pool = makeMockPool({ rows: [] });
  assert.throws(
    () => createRoutingEvaluationReportService({ pool }),
    /workspaceRoot is required/
  );
});
