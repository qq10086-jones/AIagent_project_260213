import { readFileSync } from "fs";
import { resolve } from "path";

function loadJson(filePath) {
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function result(decision, decision_source, routing_source, deny_reason_code = null, model_tier = "balanced_default") {
  const res = { 
    effective_exposure_decision: decision, 
    effective_exposure_decision_source: decision_source,
    routing_decision_source: routing_source,
    model_tier
  };
  if (deny_reason_code) res.deny_reason_code = deny_reason_code;
  return res;
}

function forced(source, routing_source, deny_reason_code, model_tier) {
  return result("forced_sequential", source, routing_source, deny_reason_code, model_tier);
}

/**
 * WS-29-01 / WS-29-02
 * Three-layer Policy-driven parallel rollout gate.
 * Evaluation order (must not be inverted):
 *   1. production_parallel_rollout.json  (rollout master + circuit-breaker)
 *   2. parallel_exposure_policy.json     (static eligibility whitelist)
 *   3. brain_router classification       (dynamic routing advisory)
 *   4. structural guard                  (independent of policy correctness)
 */
export function createParallelRolloutGate({ workspaceRoot }) {
  const rolloutPath = resolve(workspaceRoot, "configs/production_parallel_rollout.json");
  const eligibilityPath = resolve(workspaceRoot, "configs/parallel_exposure_policy.json");

  function evaluate({ run, workflow, classifier_result }) {
    const runWorkflowId = String(run?.workflow_id || "");
    const runProjectType = String(run?.project_type || "");
    // WS-29: Prioritize classifier_result.domain_lead as the input_class for M7 evaluation
    const runInputClass = String(run?.input_class || classifier_result?.domain_lead || "");

    // ── Layer 1: rollout master switch ────────────────────────────────────
    const rollout = loadJson(rolloutPath);
    if (!rollout || rollout.master_enabled !== true) {
      return forced("rollout_master_disabled", "rollout_master_disabled");
    }
    if (rollout.force_sequential === true) {
      return forced("force_sequential_override", "force_sequential_override");
    }
    if (rollout.circuit_breaker?.activated === true) {
      return forced("force_sequential_override", "force_sequential_override", "circuit_breaker_activated");
    }

    // ── Layer 2: static eligibility policy ────────────────────────────────
    const policy = loadJson(eligibilityPath);
    let staticDenied = false;
    let staticDenyReason = null;
    
    if (!policy) {
      staticDenied = true;
      staticDenyReason = "policy_not_found";
    } else if (!policy.allowed_workflow_types?.includes(runWorkflowId)) {
      staticDenied = true;
      staticDenyReason = "unapproved_workflow_class";
    } else if (!policy.allowed_project_types?.includes(runProjectType)) {
      staticDenied = true;
      staticDenyReason = "unapproved_project_type";
    } else if (!policy.fe_safe_eligible_input_classes?.includes(runInputClass)) {
      staticDenied = true;
      staticDenyReason = "unapproved_input_class";
    }

    if (staticDenied) {
      return forced("static_eligibility_denied", "static_eligibility_denied", staticDenyReason);
    }

    // ── Layer 3: Dynamic routing advisory ─────────────────────────────────
    const dynamicEnabled = rollout.dynamic_routing_enabled === true;
    const isDynamicMode = rollout.router_mode === "dynamic";
    
    if (!dynamicEnabled || !isDynamicMode) {
      return applyStructuralGuard(workflow, runInputClass, "eligibility_policy_allowed", "dynamic_routing_disabled", "balanced_default");
    }

    // Evaluate classifier result
    if (!classifier_result || classifier_result.confidence_band === "unavailable" || rollout.classifier_circuit_breaker?.activated === true) {
      return applyStructuralGuard(workflow, runInputClass, "eligibility_policy_allowed", "classifier_unavailable_fallback", "balanced_default");
    }

    if (classifier_result.confidence_band === "low") {
      return applyStructuralGuard(workflow, runInputClass, "eligibility_policy_allowed", "classifier_low_confidence_fallback", "balanced_default");
    }

    const recommendedTier = classifier_result.model_tier || "balanced_default";

    if (classifier_result.final_execution_decision === "forced_sequential") {
       return forced("eligibility_policy_allowed", "classifier_recommended_sequential", classifier_result.deny_or_degrade_reason, recommendedTier);
    }
    
    // Classifier recommends parallel
    return applyStructuralGuard(workflow, runInputClass, "eligibility_policy_allowed", "classifier_recommended_parallel", recommendedTier);
  }

  function applyStructuralGuard(workflow, runInputClass, legacySource, routingSource, modelTier) {
    if (workflow) {
      const feStep = (workflow.steps || []).find((s) => s.id === "impl_fe");
      if (feStep) {
        const feSafeClasses = feStep.fe_safe_input_classes;
        if (!Array.isArray(feSafeClasses) || !feSafeClasses.includes(runInputClass)) {
          return forced("eligibility_policy_denied", routingSource, "structural_completion_impossible", modelTier);
        }
      }
    }
    
    return result("gated_parallel_allowed", legacySource, routingSource, null, modelTier);
  }

  return { evaluate };
}
