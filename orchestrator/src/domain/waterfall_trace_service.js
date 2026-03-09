/**
 * waterfall_trace_service.js  (Layer 3 — Domain)
 *
 * WS-30-02: Waterfall Trace & Latency Attribution Service
 *
 * Records per-stage latency for every workflow run and exposes a report API
 * that produces P50/P95-queryable waterfall breakdowns.
 *
 * Explicit stages (recorded by this service at call sites):
 *   intake              – request received by dispatch
 *   routing             – brain_router.routeTaskRequest() duration
 *   policy_evaluation   – parallel_rollout_gate.evaluate() duration
 *   execution_dispatch  – workflowEngine.startWorkflowRun() duration
 *
 * Derived stages (read from workflow_steps at report time):
 *   branch_completion_be  – impl_be started_at → ended_at
 *   branch_completion_fe  – impl_fe started_at → ended_at
 *   qa_admission          – qa_verify started_at → ended_at
 *   release_pack_readiness – release_pack started_at → ended_at
 *
 * Design authority: OpenClaw_Nexus_Design_Document_v4.md § 7
 * Acceptance criteria: WS-30-02 in OpenClaw_Nexus_Engineering_Task_List_M7_v2.md
 */

import {
  insertWaterfallStage,
  listStagesByRunId,
  listWorkflowStepsForWaterfall,
  listStageDurationsForPercentiles,
  getWorkflowRunIdForRun,
} from "../data/waterfall_trace_repository.js";

// Mapping from workflow step_id to waterfall stage name
const STEP_TO_STAGE = {
  impl_be: "branch_completion_be",
  impl_fe: "branch_completion_fe",
  qa_verify: "qa_admission",
  release_pack: "release_pack_readiness",
};

/** @param {Date} start @param {Date} end @returns {number} */
function durationMs(start, end) {
  if (!start || !end) return 0;
  return Math.max(0, new Date(end).getTime() - new Date(start).getTime());
}

/**
 * Compute a percentile value from a sorted numeric array.
 * @param {number[]} sorted  ascending-sorted array
 * @param {number}   pct     percentile (0–100)
 * @returns {number}
 */
function percentile(sorted, pct) {
  if (!sorted.length) return 0;
  const idx = Math.ceil((pct / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(sorted.length - 1, idx))];
}

/**
 * @param {{ pool: import('pg').Pool }} opts
 */
export function createWaterfallTraceService({ pool }) {
  /**
   * Record a single completed stage (fire-and-forget from call sites).
   *
   * @param {string}  run_id
   * @param {string}  stage       one of the explicit stage names
   * @param {Date}    startedAt
   * @param {Date}    endedAt
   * @param {object}  [metadata]
   * @returns {Promise<void>}
   */
  async function recordStage(run_id, stage, startedAt, endedAt, metadata = {}) {
    const duration = durationMs(startedAt, endedAt);
    try {
      await insertWaterfallStage(pool, {
        run_id,
        workflow_run_id: metadata.workflow_run_id ?? null,
        stage,
        started_at: startedAt,
        ended_at: endedAt,
        duration_ms: duration,
        metadata,
      });
    } catch (err) {
      // Non-fatal: trace failure must never block the routing or dispatch path
      console.warn(`[waterfall_trace] recordStage(${stage}) failed:`, err.message);
    }
  }

  /**
   * Build a full waterfall report for a given run_id.
   *
   * Combines explicit stage records with derived stages from workflow_steps.
   * Computes routing_overhead_ms and routing_overhead_pct as standalone metrics
   * (satisfies WS-30-02 "routing overhead percentage is reported").
   *
   * @param {string} run_id
   * @returns {Promise<object>}  machine-readable waterfall report
   */
  async function buildWaterfallReport(run_id) {
    const explicitRows = await listStagesByRunId(pool, run_id);

    // Resolve workflow_run_id to fetch derived stages from workflow_steps
    const workflowRunId =
      explicitRows.find((r) => r.workflow_run_id)?.workflow_run_id ??
      (await getWorkflowRunIdForRun(pool, run_id));

    const derivedSteps = workflowRunId
      ? await listWorkflowStepsForWaterfall(pool, workflowRunId)
      : [];

    // Map explicit stage rows → normalized stage objects
    const explicitStages = explicitRows.map((r) => ({
      stage: r.stage,
      started_at: r.started_at,
      ended_at: r.ended_at,
      duration_ms: r.duration_ms ?? 0,
      source: "explicit",
    }));

    // Map workflow step rows → normalized stage objects
    const derivedStages = derivedSteps
      .filter((s) => STEP_TO_STAGE[s.step_id])
      .map((s) => ({
        stage: STEP_TO_STAGE[s.step_id],
        started_at: s.started_at,
        ended_at: s.ended_at,
        duration_ms: s.started_at && s.ended_at ? durationMs(s.started_at, s.ended_at) : null,
        source: "workflow_steps",
      }));

    const allStages = [...explicitStages, ...derivedStages].sort(
      (a, b) => new Date(a.started_at ?? 0) - new Date(b.started_at ?? 0)
    );

    // Routing overhead = intake + routing + policy_evaluation combined
    const ROUTING_STAGES = new Set(["intake", "routing", "policy_evaluation"]);
    const routingOverheadMs = allStages
      .filter((s) => ROUTING_STAGES.has(s.stage))
      .reduce((sum, s) => sum + (s.duration_ms ?? 0), 0);

    // Total duration: from earliest started_at to latest ended_at across all stages
    const starts = allStages.map((s) => new Date(s.started_at ?? 0).getTime()).filter(Boolean);
    const ends = allStages.map((s) => new Date(s.ended_at ?? 0).getTime()).filter(Boolean);
    const totalDurationMs =
      starts.length && ends.length
        ? Math.max(...ends) - Math.min(...starts)
        : 0;

    const routingOverheadPct =
      totalDurationMs > 0
        ? Math.round((routingOverheadMs / totalDurationMs) * 10000) / 100
        : null;

    return {
      run_id,
      workflow_run_id: workflowRunId ?? null,
      stages: allStages,
      routing_overhead_ms: routingOverheadMs,
      routing_overhead_pct: routingOverheadPct,
      total_duration_ms: totalDurationMs,
      generated_at: new Date().toISOString(),
    };
  }

  /**
   * Compute P50/P95 latency for a given stage across recent runs.
   * Satisfies WS-30-02 "P50/P95 latency attribution is queryable".
   *
   * @param {string} stage
   * @param {number} [limit=200]  number of recent runs to sample
   * @returns {Promise<{ stage:string, p50_ms:number, p95_ms:number, sample_size:number }>}
   */
  async function queryP50P95(stage, limit = 200) {
    const durations = await listStageDurationsForPercentiles(pool, stage, limit);
    return {
      stage,
      p50_ms: percentile(durations, 50),
      p95_ms: percentile(durations, 95),
      sample_size: durations.length,
    };
  }

  return { recordStage, buildWaterfallReport, queryP50P95 };
}
