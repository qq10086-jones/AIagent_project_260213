/**
 * routing_audit_log.js  (Layer 3 — Domain)
 *
 * WS-30-01: Routing Decision Audit Log Service
 *
 * Persists the complete routing decision record for every workflow run.
 * Satisfies Design Doc v4.0 § 5.3 queryability requirement and M7 closure
 * evidence criteria (R-15: routing decisions must be reproducible).
 *
 * Design authority: OpenClaw_Nexus_Design_Document_v4.md § 5.3, § 7
 *
 * Integration point: called from workflow_parallelization_runtime.js after
 * every gate evaluation, fire-and-forget (non-blocking to routing path).
 */

import { readFileSync } from "fs";
import { resolve } from "path";
import { v4 as uuidv4 } from "uuid";
import {
  insertRoutingDecision,
  listRoutingDecisionsByRunId,
  getLatestRoutingDecision,
} from "../data/routing_audit_repository.js";

/** @param {string} workspaceRoot */
function loadRouterConfig(workspaceRoot) {
  try {
    const p = resolve(workspaceRoot, "configs/production_parallel_rollout.json");
    return JSON.parse(readFileSync(p, "utf8"));
  } catch {
    return {};
  }
}

/**
 * @param {{ pool: import('pg').Pool, workspaceRoot: string }} opts
 */
export function createRoutingAuditLogService({ pool, workspaceRoot }) {
  /**
   * Assemble and persist a routing decision audit record.
   *
   * Fields sourced from (Design Doc v4.0 § 5.3):
   *   - routerConfig:      router_mode, dynamic_routing_enabled
   *   - classifierResult:  classifier_version, confidence, confidence_band,
   *                        work_shape, domain_lead, parallel_candidate, model_tier,
   *                        deny_or_degrade_reason, feature_snapshot_ref,
   *                        safety_override_result
   *   - gateDecision:      routing_decision_source, final_execution_decision
   *   - run:               run_id, workflow_run_id, workflow_id
   *
   * @param {{ run: object, gateDecision: object, classifierResult: object|null }} params
   * @returns {Promise<string>} log_id of the persisted record
   */
  async function log({ run, gateDecision, classifierResult }) {
    const routerConfig = loadRouterConfig(workspaceRoot);
    const cr = classifierResult || {};
    const logId = uuidv4();

    const record = {
      log_id: logId,
      // Run identity
      run_id: String(run?.run_id || run?.workflow_run_id || ""),
      workflow_run_id: run?.workflow_run_id ?? null,
      workflow_id: run?.workflow_id ?? null,
      // Router config state (§ 5.3)
      router_mode: routerConfig.router_mode ?? "static_policy_only",
      dynamic_routing_enabled: routerConfig.dynamic_routing_enabled ?? false,
      // Classifier outputs (§ 14.3)
      classifier_version: cr.classifier_version ?? null,
      classifier_confidence: cr.confidence ?? null,
      classifier_confidence_band: cr.confidence_band ?? null,
      classifier_work_shape: cr.work_shape ?? null,
      classifier_domain_lead: cr.domain_lead ?? null,
      classifier_parallel_candidate: cr.parallel_candidate ?? null,
      classifier_model_tier: cr.model_tier ?? null,
      classifier_deny_or_degrade_reason: cr.deny_or_degrade_reason ?? null,
      feature_snapshot_ref: cr.feature_snapshot_ref ?? null,
      // Gate decision outputs (normalized sources from § 5.3)
      routing_decision_source:
        gateDecision?.routing_decision_source ?? "rollout_master_disabled",
      final_execution_decision:
        gateDecision?.effective_exposure_decision ?? "forced_sequential",
      safety_override_result: cr.safety_override_result ?? null,
    };

    try {
      await insertRoutingDecision(pool, record);
    } catch (err) {
      // Non-fatal: audit log failure must never block routing decisions (R-15 mitigation)
      console.warn("[routing_audit] insert failed:", err.message);
    }

    return logId;
  }

  /**
   * Retrieve all routing decisions for a run_id.
   * Satisfies WS-30-01 "query API can retrieve decision history per run".
   *
   * @param {string} run_id
   * @returns {Promise<object[]>}
   */
  async function queryByRunId(run_id) {
    return listRoutingDecisionsByRunId(pool, run_id);
  }

  /**
   * Retrieve the latest routing decision for a workflow_run_id.
   * Used for incident review and M7 closure evidence queries.
   *
   * @param {string} workflow_run_id
   * @returns {Promise<object|null>}
   */
  async function getLatestForWorkflowRun(workflow_run_id) {
    return getLatestRoutingDecision(pool, workflow_run_id);
  }

  return { log, queryByRunId, getLatestForWorkflowRun };
}
