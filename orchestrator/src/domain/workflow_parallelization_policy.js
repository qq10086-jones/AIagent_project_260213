import { createParallelRolloutGate } from "./parallel_rollout_gate.js";

function cloneWorkflow(workflow) {
  return JSON.parse(JSON.stringify(workflow || {}));
}

function findStepIndex(steps, stepId) {
  return steps.findIndex((step) => String(step?.id || "") === String(stepId || ""));
}

function hasExplicitDagMetadata(steps = []) {
  return steps.some((step) => Array.isArray(step?.depends_on));
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

    steps[beIndex].depends_on = ["arch_design"];
    steps[feIndex].depends_on = ["arch_design"];
    steps[qaIndex].depends_on = ["impl_be", "impl_fe"];
    return { workflow: cloned, gateDecision };
  }

  return {
    evaluateBeFeParallelizationGate,
    resolveWorkflowForRun,
  };
}
