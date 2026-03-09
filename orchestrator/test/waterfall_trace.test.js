/**
 * WS-30-02: Waterfall Trace & Latency Attribution — Integration Tests
 *
 * Verifies that:
 *  1. recordStage() persists a stage with correct fields
 *  2. recordStage() is non-fatal when DB insert fails
 *  3. buildWaterfallReport() assembles explicit + derived stages
 *  4. buildWaterfallReport() computes routing_overhead_ms and routing_overhead_pct
 *  5. queryP50P95() returns correct percentile values
 *  6. All four explicit stages (intake, routing, policy_evaluation, execution_dispatch) are recorded
 *  7. buildWaterfallReport() handles missing workflow_run_id gracefully
 *  8. percentile computation is correct across edge cases
 */

import test from "node:test";
import assert from "node:assert/strict";

import { createWaterfallTraceService } from "../src/domain/waterfall_trace_service.js";

// ── Mock pool factory ─────────────────────────────────────────────────────────

/**
 * Builds a mock pool that records all queries and returns configurable rows.
 * Supports per-query-pattern row overrides via `rowsByPattern`.
 */
function makeMockPool({ shouldFail = false, rowsByPattern = {} } = {}) {
  const calls = [];
  return {
    calls,
    async query(sql, params) {
      calls.push({ sql, params });
      if (shouldFail) throw new Error("mock DB error");
      // Pattern-matched row override
      for (const [pattern, rows] of Object.entries(rowsByPattern)) {
        if (sql.includes(pattern)) return { rows };
      }
      return { rows: [] };
    },
  };
}

// ── Stub data ─────────────────────────────────────────────────────────────────

const T0 = new Date("2026-03-09T10:00:00.000Z");
const T1 = new Date("2026-03-09T10:00:00.005Z"); // 5 ms later
const T2 = new Date("2026-03-09T10:00:00.050Z"); // +45 ms (routing)
const T3 = new Date("2026-03-09T10:00:00.058Z"); // +8 ms (policy)
const T4 = new Date("2026-03-09T10:00:00.258Z"); // +200 ms (dispatch)

// ── Tests ─────────────────────────────────────────────────────────────────────

test("WS-30-02: recordStage() persists a stage record with correct fields", async () => {
  const pool = makeMockPool();
  const svc = createWaterfallTraceService({ pool });

  await svc.recordStage("run-waterfall-01", "routing", T1, T2);

  assert.equal(pool.calls.length, 1);
  const { sql, params } = pool.calls[0];
  assert.ok(sql.includes("waterfall_stage_log"), "must INSERT into waterfall_stage_log");
  assert.equal(params[0], "run-waterfall-01", "run_id");
  assert.equal(params[2], "routing", "stage");
  assert.deepEqual(params[3], T1, "started_at");
  assert.deepEqual(params[4], T2, "ended_at");
  assert.equal(params[5], 45, "duration_ms should be 45 ms");
});

test("WS-30-02: recordStage() is non-fatal when DB insert fails", async () => {
  const pool = makeMockPool({ shouldFail: true });
  const svc = createWaterfallTraceService({ pool });

  await assert.doesNotReject(() => svc.recordStage("run-123", "intake", T0, T1));
});

test("WS-30-02: recordStage() records all four explicit stages correctly", async () => {
  const pool = makeMockPool();
  const svc = createWaterfallTraceService({ pool });

  const stages = ["intake", "routing", "policy_evaluation", "execution_dispatch"];
  for (const stage of stages) {
    await svc.recordStage("run-4stages", stage, T0, T1);
  }

  assert.equal(pool.calls.length, 4);
  const recordedStages = pool.calls.map((c) => c.params[2]);
  assert.deepEqual(recordedStages, stages);
});

test("WS-30-02: buildWaterfallReport() assembles explicit stages in chronological order", async () => {
  const explicitRows = [
    { run_id: "run-r01", workflow_run_id: "wfrun-r01", stage: "routing", started_at: T1, ended_at: T2, duration_ms: 45, source: "explicit" },
    { run_id: "run-r01", workflow_run_id: "wfrun-r01", stage: "intake", started_at: T0, ended_at: T1, duration_ms: 5, source: "explicit" },
    { run_id: "run-r01", workflow_run_id: "wfrun-r01", stage: "policy_evaluation", started_at: T2, ended_at: T3, duration_ms: 8, source: "explicit" },
    { run_id: "run-r01", workflow_run_id: "wfrun-r01", stage: "execution_dispatch", started_at: T3, ended_at: T4, duration_ms: 200, source: "explicit" },
  ];
  const pool = makeMockPool({
    rowsByPattern: {
      "waterfall_stage_log": explicitRows,
      "workflow_steps": [],
    },
  });
  const svc = createWaterfallTraceService({ pool });

  const report = await svc.buildWaterfallReport("run-r01");

  assert.equal(report.run_id, "run-r01");
  assert.equal(report.workflow_run_id, "wfrun-r01");
  assert.ok(Array.isArray(report.stages), "stages must be an array");
  assert.equal(report.stages.length, 4, "all 4 explicit stages must appear");

  // Chronological order (intake first)
  assert.equal(report.stages[0].stage, "intake");
});

test("WS-30-02: buildWaterfallReport() computes routing_overhead_ms correctly", async () => {
  // intake=5ms, routing=45ms, policy_evaluation=8ms → routing overhead = 58ms
  const explicitRows = [
    { stage: "intake", started_at: T0, ended_at: T1, duration_ms: 5, workflow_run_id: null },
    { stage: "routing", started_at: T1, ended_at: T2, duration_ms: 45, workflow_run_id: null },
    { stage: "policy_evaluation", started_at: T2, ended_at: T3, duration_ms: 8, workflow_run_id: null },
    { stage: "execution_dispatch", started_at: T3, ended_at: T4, duration_ms: 200, workflow_run_id: null },
  ];
  const pool = makeMockPool({
    rowsByPattern: { "waterfall_stage_log": explicitRows, "workflow_steps": [] },
  });
  const svc = createWaterfallTraceService({ pool });

  const report = await svc.buildWaterfallReport("run-overhead");

  // routing overhead = intake + routing + policy_evaluation = 5 + 45 + 8 = 58ms
  assert.equal(report.routing_overhead_ms, 58);
  assert.ok(typeof report.routing_overhead_pct === "number", "routing_overhead_pct must be a number");
  assert.ok(report.routing_overhead_pct > 0, "routing overhead pct must be positive");
  assert.ok(report.total_duration_ms > 0, "total_duration_ms must be positive");
});

test("WS-30-02: buildWaterfallReport() incorporates derived workflow_steps stages", async () => {
  const explicitRows = [
    { stage: "intake", started_at: T0, ended_at: T1, duration_ms: 5, workflow_run_id: "wfrun-derived" },
  ];
  const stepRows = [
    { step_id: "impl_be", started_at: new Date("2026-03-09T10:01:00Z"), ended_at: new Date("2026-03-09T10:01:45Z"), status: "succeeded" },
    { step_id: "impl_fe", started_at: new Date("2026-03-09T10:01:00Z"), ended_at: new Date("2026-03-09T10:01:38Z"), status: "succeeded" },
    { step_id: "qa_verify", started_at: new Date("2026-03-09T10:01:45Z"), ended_at: new Date("2026-03-09T10:02:00Z"), status: "succeeded" },
    { step_id: "release_pack", started_at: new Date("2026-03-09T10:02:00Z"), ended_at: new Date("2026-03-09T10:02:05Z"), status: "succeeded" },
  ];
  const pool = makeMockPool({
    rowsByPattern: {
      "waterfall_stage_log": explicitRows,
      "workflow_steps": stepRows,
    },
  });
  const svc = createWaterfallTraceService({ pool });

  const report = await svc.buildWaterfallReport("run-derived");

  const stageNames = report.stages.map((s) => s.stage);
  assert.ok(stageNames.includes("branch_completion_be"), "must include branch_completion_be");
  assert.ok(stageNames.includes("branch_completion_fe"), "must include branch_completion_fe");
  assert.ok(stageNames.includes("qa_admission"), "must include qa_admission");
  assert.ok(stageNames.includes("release_pack_readiness"), "must include release_pack_readiness");

  // Verify source labeling
  const beBranch = report.stages.find((s) => s.stage === "branch_completion_be");
  assert.equal(beBranch.source, "workflow_steps");
});

test("WS-30-02: buildWaterfallReport() handles no stages gracefully", async () => {
  const pool = makeMockPool({ rowsByPattern: { "waterfall_stage_log": [], "workflow_steps": [] } });
  const svc = createWaterfallTraceService({ pool });

  const report = await svc.buildWaterfallReport("run-empty");

  assert.equal(report.stages.length, 0);
  assert.equal(report.routing_overhead_ms, 0);
  assert.equal(report.total_duration_ms, 0);
  assert.equal(report.routing_overhead_pct, null);
});

test("WS-30-02: queryP50P95() returns correct percentile values", async () => {
  // 10 duration values: 10,20,30,40,50,60,70,80,90,100
  const durations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
  const pool = makeMockPool({
    rowsByPattern: { "waterfall_stage_log": durations.map((d) => ({ duration_ms: d })) },
  });
  const svc = createWaterfallTraceService({ pool });

  const result = await svc.queryP50P95("routing");

  assert.equal(result.stage, "routing");
  assert.equal(result.sample_size, 10);
  // p50 = 5th value (index 4) = 50ms
  assert.equal(result.p50_ms, 50);
  // p95 = 10th value (index 9) = 100ms
  assert.equal(result.p95_ms, 100);
});

test("WS-30-02: queryP50P95() handles empty dataset (returns 0)", async () => {
  const pool = makeMockPool({ rowsByPattern: { "waterfall_stage_log": [] } });
  const svc = createWaterfallTraceService({ pool });

  const result = await svc.queryP50P95("routing");

  assert.equal(result.p50_ms, 0);
  assert.equal(result.p95_ms, 0);
  assert.equal(result.sample_size, 0);
});

test("WS-30-02: buildWaterfallReport() includes generated_at timestamp in output", async () => {
  const pool = makeMockPool({ rowsByPattern: { "waterfall_stage_log": [], "workflow_steps": [] } });
  const svc = createWaterfallTraceService({ pool });

  const report = await svc.buildWaterfallReport("run-timestamp");

  assert.ok(report.generated_at, "generated_at must be present");
  assert.ok(!isNaN(new Date(report.generated_at).getTime()), "generated_at must be a valid ISO string");
});
