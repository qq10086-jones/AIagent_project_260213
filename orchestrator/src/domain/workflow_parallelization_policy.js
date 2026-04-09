import fs from "fs";
import path from "path";
import { createParallelRolloutGate } from "./parallel_rollout_gate.js";

function cloneWorkflow(workflow) {
  return JSON.parse(JSON.stringify(workflow || {}));
}

function findStepIndex(steps, stepId) {
  return steps.findIndex((step) => String(step?.id || "") === String(stepId || ""));
}

// Only treat the workflow as pre-wired if the BE/FE/QA steps (the ones this
// service manages) already carry explicit depends_on.  Post-processing steps
// like deploy_preview may carry static deps without implying the full DAG is
// declared, and must not short-circuit the rollout gate evaluation.
const MANAGED_STEP_IDS = new Set(["impl_be", "impl_fe", "qa_verify"]);
function hasExplicitDagMetadata(steps = []) {
  return steps.some((step) => MANAGED_STEP_IDS.has(step.id) && Array.isArray(step?.depends_on));
}

/**
 * Phase 3 (P3-2): Check if architect_to_impl.json declares be_fe_independent.
 * When the handoff explicitly states the BE and FE are independent modules,
 * parallel scheduling is safe. Otherwise falls back to sequential.
 * @param {string} workspaceRoot
 * @param {string} runId
 * @returns {{ independent: boolean, source: string }}
 */
function checkBeFeIndependence(workspaceRoot, runId) {
  const artifactRoot = `runtime/artifacts/release/${runId || "unknown-run"}`;
  const handoffPath = path.resolve(workspaceRoot, artifactRoot, "handoff/architect_to_impl.json");
  try {
    if (!fs.existsSync(handoffPath)) {
      return { independent: false, source: "handoff_not_found" };
    }
    const handoff = JSON.parse(fs.readFileSync(handoffPath, "utf8"));
    if (handoff?.be_fe_independent === true) {
      return { independent: true, source: "architect_handoff" };
    }
    // If be_to_fe.json typed_handoff has non-empty contracts → interface deps exist
    const beToFePath = path.resolve(workspaceRoot, artifactRoot, "handoff/be_to_fe.json");
    if (fs.existsSync(beToFePath)) {
      const beToFe = JSON.parse(fs.readFileSync(beToFePath, "utf8"));
      const hasContracts = Array.isArray(beToFe?.api_contracts) && beToFe.api_contracts.length > 0;
      if (hasContracts) {
        return { independent: false, source: "be_to_fe_has_contracts" };
      }
    }
    return { independent: false, source: "not_declared" };
  } catch {
    return { independent: false, source: "handoff_read_error" };
  }
}

export function createWorkflowParallelizationPolicyService({ registry, workspaceRoot }) {
  const rolloutGate = createParallelRolloutGate({ workspaceRoot });

  function evaluateBeFeParallelizationGate(run) {
    const workflow = registry.workflows?.[run.workflow_id];

    if (!workflow || !Array.isArray(workflow.steps)) {
      return { allowed: false, mode: "sequential", reason_code: "WORKFLOW_NOT_FOUND" };
    }
    if (hasExplicitDagMetadata(workflow.steps)) {
      return { allowed: false, mode: "workflow_defined", reason_code: "WORKFLOW_METADATA_EXPLICIT" };
    }

    // WS-29: Extract classifier result from run object (DB load maps input_json to run.input_json)
    let inputObj = run.input;
    if (!inputObj && typeof run.input_json === "string") {
      try {
        inputObj = JSON.parse(run.input_json);
      } catch (e) {
        console.warn("[parallel_policy] failed to parse input_json:", e.message);
      }
    } else if (!inputObj && typeof run.input_json === "object") {
      inputObj = run.input_json;
    }

    const classifierResult = run.classifier_result || inputObj?.task_envelope?.classifier_result || null;
    const gateResult = rolloutGate.evaluate({ run, workflow, classifier_result: classifierResult });
    if (gateResult.effective_exposure_decision === "gated_parallel_allowed") {
      return { allowed: true, mode: "gated_parallel", reason_code: "GATED_PARALLEL_ALLOWED", ...gateResult };
    }
    return {
      allowed: false,
      mode: "sequential",
      reason_code: gateResult.deny_reason_code || gateResult.effective_exposure_decision_source,
      ...gateResult,
    };
  }

  function resolveWorkflowForRun(run) {
    const workflow = registry.workflows?.[run.workflow_id];
    if (!workflow || !Array.isArray(workflow.steps)) {
      return { workflow: null, gateDecision: { allowed: false, mode: "sequential", reason_code: "WORKFLOW_NOT_FOUND" } };
    }
    const cloned = cloneWorkflow(workflow);
    const gateDecision = evaluateBeFeParallelizationGate(run);
    if (!gateDecision.allowed) {
      return { workflow: cloned, gateDecision };
    }

    const steps = cloned.steps || [];
    const archIndex = findStepIndex(steps, "arch_design");
    const beIndex = findStepIndex(steps, "impl_be");
    const feIndex = findStepIndex(steps, "impl_fe");
    const qaIndex = findStepIndex(steps, "qa_verify");
    if ([archIndex, beIndex, feIndex, qaIndex].some((index) => index < 0)) {
      return {
        workflow: cloned,
        gateDecision: { allowed: false, mode: "sequential", reason_code: "WORKFLOW_STEPS_INCOMPLETE" },
      };
    }

    // Phase 3 (P3-2): Check architect handoff for be_fe_independent declaration
    const runId = run.run_id || run.workflow_run_id;
    const independence = checkBeFeIndependence(workspaceRoot, runId);
    const effectiveGateDecision = {
      ...gateDecision,
      be_fe_independence: independence,
    };

    if (!independence.independent) {
      // Architect did not declare independence — fall back to sequential BE→FE
      return {
        workflow: cloned,
        gateDecision: {
          ...effectiveGateDecision,
          allowed: false,
          mode: "sequential_be_fe_dependent",
          reason_code: `BE_FE_DEPENDENT:${independence.source}`,
        },
      };
    }

    steps[beIndex].depends_on = ["arch_design"];
    steps[feIndex].depends_on = ["arch_design"];
    steps[qaIndex].depends_on = ["impl_be", "impl_fe"];
    return { workflow: cloned, gateDecision: effectiveGateDecision };
  }

  return {
    evaluateBeFeParallelizationGate,
    resolveWorkflowForRun,
    checkBeFeIndependence,
  };
}
