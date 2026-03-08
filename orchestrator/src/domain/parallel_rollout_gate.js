import { readFileSync } from "fs";
import { resolve } from "path";

function loadJson(filePath) {
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function forced(source, deny_reason_code) {
  const result = { effective_exposure_decision: "forced_sequential", effective_exposure_decision_source: source };
  if (deny_reason_code) result.deny_reason_code = deny_reason_code;
  return result;
}

/**
 * WS-24.5-01 / WS-24.5-02
 * Policy-driven parallel rollout gate.
 * Evaluation order (must not be inverted):
 *   1. production_parallel_rollout.json  (rollout master + circuit-breaker)
 *   2. parallel_exposure_policy.json     (eligibility whitelist)
 *   3. structural guard                  (independent of policy correctness)
 */
export function createParallelRolloutGate({ workspaceRoot }) {
  const rolloutPath = resolve(workspaceRoot, "configs/production_parallel_rollout.json");
  const eligibilityPath = resolve(workspaceRoot, "configs/parallel_exposure_policy.json");

  function evaluate({ run, workflow }) {
    const runWorkflowId = String(run?.workflow_id || "");
    const runProjectType = String(run?.project_type || "");
    const runInputClass = String(run?.input_class || "");

    // ── Layer 1: rollout master switch ────────────────────────────────────
    const rollout = loadJson(rolloutPath);
    if (!rollout || rollout.master_enabled !== true) {
      return forced("rollout_master_disabled");
    }
    if (rollout.force_sequential === true) {
      return forced("force_sequential_override");
    }
    if (rollout.circuit_breaker?.activated === true) {
      return forced("force_sequential_override", "circuit_breaker_activated");
    }

    // ── Layer 2: eligibility policy ───────────────────────────────────────
    const policy = loadJson(eligibilityPath);
    if (!policy) {
      return forced("eligibility_policy_denied", "policy_not_found");
    }
    if (!policy.allowed_workflow_types?.includes(runWorkflowId)) {
      return forced("eligibility_policy_denied", "unapproved_workflow_class");
    }
    if (!policy.allowed_project_types?.includes(runProjectType)) {
      return forced("eligibility_policy_denied", "unapproved_project_type");
    }
    if (!policy.fe_safe_eligible_input_classes?.includes(runInputClass)) {
      return forced("eligibility_policy_denied", "unapproved_input_class");
    }

    // ── Layer 3: structural guard (D7 — independent of policy) ───────────
    if (workflow) {
      const feStep = (workflow.steps || []).find((s) => s.id === "impl_fe");
      if (feStep) {
        const feSafeClasses = feStep.fe_safe_input_classes;
        if (!Array.isArray(feSafeClasses) || !feSafeClasses.includes(runInputClass)) {
          return forced("eligibility_policy_denied", "structural_completion_impossible");
        }
      }
    }

    return {
      effective_exposure_decision: "gated_parallel_allowed",
      effective_exposure_decision_source: "eligibility_policy_allowed",
    };
  }

  return { evaluate };
}
