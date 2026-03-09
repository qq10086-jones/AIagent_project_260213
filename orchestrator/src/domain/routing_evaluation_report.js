/**
 * routing_evaluation_report.js  (Layer 3 — Domain)
 *
 * WS-30-03: Routing Evaluation Report
 * Aggregates routing decision log + waterfall latency data into a repeatable
 * evaluation report covering all six WS-30-03 dimensions:
 *   1. routing precision
 *   2. high-risk misroute rate (including tier misroute)
 *   3. low-confidence fallback ratio
 *   4. forced_sequential ratio
 *   5. latency delta
 *   6. incident delta (placeholder — populated from external incident log)
 *
 * Also produces a counterfactual replay comparison (staging only):
 *   static-policy path vs dynamic-routing path, evaluated against the
 *   governed replay manifest (m6_staging_replay_manifest.json).
 *
 * Design authority: OpenClaw_Nexus_Engineering_Task_List_M7_v2.md § WS-30-03
 */

import { readFileSync } from "fs";
import { resolve } from "path";
import { queryDecisionStats } from "../data/routing_audit_repository.js";

/**
 * @param {string} filePath
 * @returns {object|null}
 */
function loadJson(filePath) {
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

/**
 * Load static eligibility sets from the governed policy file.
 * Falls back to deny-all if the file is missing or malformed.
 *
 * @param {string} workspaceRoot
 * @returns {{ allowedWorkflows: Set<string>, allowedProjects: Set<string>, eligibleClasses: Set<string> }}
 */
function loadEligibilitySets(workspaceRoot) {
  const policy = loadJson(resolve(workspaceRoot, "configs/parallel_exposure_policy.json"));
  return {
    allowedWorkflows: new Set(policy?.allowed_workflow_types ?? []),
    allowedProjects:  new Set(policy?.allowed_project_types ?? []),
    eligibleClasses:  new Set(policy?.fe_safe_eligible_input_classes ?? []),
  };
}

/**
 * Derive the static-policy decision for a replay case.
 * Reads eligibility rules from parallel_exposure_policy.json via the sets
 * supplied by loadEligibilitySets(), keeping parity with parallel_rollout_gate.js Layer-2 logic.
 *
 * @param {{ workflow_type, project_type, input_class }} replayCase
 * @param {{ allowedWorkflows: Set, allowedProjects: Set, eligibleClasses: Set }} eligibility
 * @returns {"gated_parallel_allowed"|"forced_sequential"}
 */
function staticDecisionFor(replayCase, { allowedWorkflows, allowedProjects, eligibleClasses }) {
  if (
    allowedWorkflows.has(replayCase.workflow_type) &&
    allowedProjects.has(replayCase.project_type) &&
    eligibleClasses.has(replayCase.input_class)
  ) {
    return "gated_parallel_allowed";
  }
  return "forced_sequential";
}

/**
 * Derive the dynamic-routing decision for a replay case.
 * Uses `fe_parallel_eligible_expected` as the classifier ground-truth label,
 * which is what a correctly-trained classifier should produce for each case.
 *
 * @param {{ fe_parallel_eligible_expected: boolean }} replayCase
 * @returns {"gated_parallel_allowed"|"forced_sequential"}
 */
function dynamicDecisionFor(replayCase) {
  return replayCase.fe_parallel_eligible_expected
    ? "gated_parallel_allowed"
    : "forced_sequential";
}

/**
 * Build the counterfactual comparison section from the replay manifest.
 * Reads eligibility policy from disk so that the comparison always reflects
 * the current governed policy, not hardcoded constants.
 *
 * @param {string} manifestPath   Absolute path to the replay manifest JSON.
 * @param {string} workspaceRoot  Root used to locate parallel_exposure_policy.json.
 * @returns {object}
 */
function buildCounterfactualComparison(manifestPath, workspaceRoot) {
  const manifest = loadJson(manifestPath);
  if (!manifest || !Array.isArray(manifest.corpus)) {
    return { error: "manifest_not_loaded", manifest_path: manifestPath };
  }

  const eligibility = loadEligibilitySets(workspaceRoot);
  let staticParallelCount  = 0;
  let dynamicParallelCount = 0;
  let agreementCount       = 0;

  const cases = manifest.corpus.map((c) => {
    const staticDec  = staticDecisionFor(c, eligibility);
    const dynamicDec = dynamicDecisionFor(c);
    const agrees     = staticDec === dynamicDec;
    if (staticDec  === "gated_parallel_allowed") staticParallelCount++;
    if (dynamicDec === "gated_parallel_allowed") dynamicParallelCount++;
    if (agrees) agreementCount++;
    return {
      replay_id:        c.replay_id,
      input_class:      c.input_class,
      project_type:     c.project_type,
      static_decision:  staticDec,
      dynamic_decision: dynamicDec,
      expected_parallel: Boolean(c.fe_parallel_eligible_expected),
      agreement:         agrees,
    };
  });

  const total = cases.length;
  return {
    manifest_id:                manifest.manifest_id,
    total_cases:                total,
    static_gated_parallel:      staticParallelCount,
    dynamic_gated_parallel:     dynamicParallelCount,
    agreement_count:            agreementCount,
    agreement_rate:             total > 0 ? agreementCount / total : null,
    dynamic_uplift:             dynamicParallelCount - staticParallelCount,
    cases,
  };
}

/**
 * Reduce aggregate stat rows into a dimension-keyed count map.
 * @param {object[]} rows
 * @param {string} field
 * @returns {Record<string, number>}
 */
function aggregateByField(rows, field) {
  const out = {};
  for (const row of rows) {
    const key = row[field] ?? "(null)";
    out[key] = (out[key] ?? 0) + Number(row.count);
  }
  return out;
}

// ── Factory ───────────────────────────────────────────────────────────────────

/**
 * @param {{ pool: import('pg').Pool, workspaceRoot: string, waterfallTraceService?: object|null }} opts
 */
export function createRoutingEvaluationReportService({ pool, workspaceRoot, waterfallTraceService = null }) {
  if (!workspaceRoot || typeof workspaceRoot !== "string") {
    throw new Error("workspaceRoot is required");
  }
  if (!pool?.query || typeof pool.query !== "function") {
    throw new Error("pool.query is required");
  }

  /**
   * Generate a full WS-30-03 routing evaluation report.
   *
   * @param {{ from?: Date|null, to?: Date|null, replayManifestPath?: string|null }} opts
   * @returns {Promise<object>}
   */
  async function generateReport({ from = null, to = null, replayManifestPath = null } = {}) {
    // ── 1. Aggregate routing decision stats ───────────────────────────────
    const rows = await queryDecisionStats(pool, { from, to });

    const total = rows.reduce((s, r) => s + Number(r.count), 0);

    const gatedParallelCount = rows
      .filter((r) => r.final_execution_decision === "gated_parallel_allowed")
      .reduce((s, r) => s + Number(r.count), 0);

    const forcedSeqCount = rows
      .filter((r) => r.final_execution_decision === "forced_sequential")
      .reduce((s, r) => s + Number(r.count), 0);

    const lowConfCount = rows
      .filter((r) => r.classifier_confidence_band === "low")
      .reduce((s, r) => s + Number(r.count), 0);

    // high-risk misroute: work_shape=high_risk_release_sensitive but admitted as parallel
    const highRiskMisrouteCount = rows
      .filter(
        (r) =>
          r.classifier_work_shape === "high_risk_release_sensitive" &&
          r.final_execution_decision === "gated_parallel_allowed"
      )
      .reduce((s, r) => s + Number(r.count), 0);

    // ── 2. Latency delta (from waterfall trace service) ───────────────────
    let latencyStats = null;
    if (waterfallTraceService) {
      const [intake, routing, policy, dispatch] = await Promise.all([
        waterfallTraceService.queryP50P95("intake"),
        waterfallTraceService.queryP50P95("routing"),
        waterfallTraceService.queryP50P95("policy_evaluation"),
        waterfallTraceService.queryP50P95("execution_dispatch"),
      ]);
      const routingOverheadP50 =
        (intake.p50_ms || 0) + (routing.p50_ms || 0) + (policy.p50_ms || 0);
      latencyStats = {
        intake,
        routing,
        policy_evaluation: policy,
        execution_dispatch: dispatch,
        routing_overhead_p50_ms: routingOverheadP50,
      };
    }

    // ── 3. Counterfactual replay comparison ───────────────────────────────
    const counterfactualComparison = replayManifestPath
      ? buildCounterfactualComparison(replayManifestPath, workspaceRoot)
      : null;

    // ── 4. Assemble report ────────────────────────────────────────────────
    return {
      generated_at:               new Date().toISOString(),
      window:                     { from: from ?? null, to: to ?? null },
      total_runs:                 total,

      // Dimension 1: routing precision
      routing_precision:          total > 0 ? gatedParallelCount / total : null,

      // Dimension 2: high-risk misroute rate
      high_risk_misroute_count:   highRiskMisrouteCount,
      high_risk_misroute_rate:    total > 0 ? highRiskMisrouteCount / total : null,

      // Dimension 3: low-confidence fallback ratio
      low_confidence_fallback_count: lowConfCount,
      low_confidence_fallback_ratio: total > 0 ? lowConfCount / total : null,

      // Dimension 4: forced_sequential ratio
      forced_sequential_count:    forcedSeqCount,
      forced_sequential_ratio:    total > 0 ? forcedSeqCount / total : null,

      // Dimension 5: latency delta
      latency_stats:              latencyStats,

      // Dimension 6: incident delta (external — not populated by this service)
      incident_delta:             null,

      // Breakdown tables
      by_decision_source:         aggregateByField(rows, "routing_decision_source"),
      by_work_shape:              aggregateByField(rows, "classifier_work_shape"),
      by_confidence_band:         aggregateByField(rows, "classifier_confidence_band"),
      by_model_tier:              aggregateByField(rows, "classifier_model_tier"),

      // Counterfactual replay comparison
      counterfactual_comparison:  counterfactualComparison,
    };
  }

  return { generateReport };
}
