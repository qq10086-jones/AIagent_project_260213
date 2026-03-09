/**
 * WS-24-04 & WS-29-01 Contract Validation Integration Tests
 * Covers:
 *  - M6 static paths (FE-safe proceeds, rollout master disabled, etc.)
 *  - M7 dynamic routing paths (disabled, unavailable, low confidence, etc.)
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { createParallelRolloutGate } from "../src/domain/parallel_rollout_gate.js";
import { evaluateQaAdmission, evaluateReleaseGating } from "../src/domain/parallel_qa_admission_guard.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function buildWorkspace(rollout, eligibility) {
  const dir = fs.mkdtempSync(path.join(__dirname, "tmp_gate_"));
  const configDir = path.join(dir, "configs");
  fs.mkdirSync(configDir, { recursive: true });
  if (rollout !== null) {
    fs.writeFileSync(path.join(configDir, "production_parallel_rollout.json"), JSON.stringify(rollout));
  }
  if (eligibility !== null) {
    fs.writeFileSync(path.join(configDir, "parallel_exposure_policy.json"), JSON.stringify(eligibility));
  }
  return dir;
}

function cleanup(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
}

const BASE_ROLLOUT = {
  master_enabled: true,
  force_sequential: false,
  dynamic_routing_enabled: false,
  router_mode: "static_policy_only",
  circuit_breaker: { activated: false },
};

const DYNAMIC_ROLLOUT = {
  ...BASE_ROLLOUT,
  dynamic_routing_enabled: true,
  router_mode: "dynamic",
};

const BASE_POLICY = {
  allowed_workflow_types: ["coding_team_v0"],
  allowed_project_types: ["crm"],
  fe_safe_eligible_input_classes: ["fe_led"],
};

const FE_SAFE_WORKFLOW = {
  steps: [
    { id: "pm_spec" },
    { id: "arch_design" },
    { id: "impl_be" },
    { id: "impl_fe", fe_safe_input_classes: ["fe_led"] },
    { id: "qa_verify" },
    { id: "release_pack" },
  ],
};

const FE_SAFE_RUN = {
  workflow_id: "coding_team_v0",
  project_type: "crm",
  input_class: "fe_led",
};

// ── Static Gate tests (M6 baseline) ──────────────────────────────────────────

test("WS-24-04: FE-safe case proceeds under gated parallel path (dynamic routing disabled)", () => {
  const dir = buildWorkspace(BASE_ROLLOUT, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    assert.equal(result.effective_exposure_decision, "gated_parallel_allowed");
    assert.equal(result.effective_exposure_decision_source, "eligibility_policy_allowed");
    assert.equal(result.routing_decision_source, "dynamic_routing_disabled");
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04: non-FE-safe input_class remains sequential (unapproved_input_class)", () => {
  const dir = buildWorkspace(BASE_ROLLOUT, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({
      run: { ...FE_SAFE_RUN, input_class: "be_led" },
      workflow: FE_SAFE_WORKFLOW,
    });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.deny_reason_code, "unapproved_input_class");
    assert.equal(result.routing_decision_source, "static_eligibility_denied");
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04: rollout master disabled -> forced_sequential", () => {
  const dir = buildWorkspace({ ...BASE_ROLLOUT, master_enabled: false }, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.routing_decision_source, "rollout_master_disabled");
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04: force_sequential override -> forced_sequential", () => {
  const dir = buildWorkspace({ ...BASE_ROLLOUT, force_sequential: true }, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.routing_decision_source, "force_sequential_override");
  } finally {
    cleanup(dir);
  }
});

// ── Dynamic Routing Gate tests (M7 WS-29-01) ──────────────────────────────────

test("WS-29-01: classifier unavailable -> falls back to static eligibility", () => {
  const dir = buildWorkspace(DYNAMIC_ROLLOUT, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW, classifier_result: null });
    assert.equal(result.effective_exposure_decision, "gated_parallel_allowed");
    assert.equal(result.routing_decision_source, "classifier_unavailable_fallback");
    assert.equal(result.model_tier, "balanced_default");
  } finally {
    cleanup(dir);
  }
});

test("WS-29-01: classifier circuit breaker active -> falls back to static eligibility", () => {
  const rollout = { ...DYNAMIC_ROLLOUT, classifier_circuit_breaker: { activated: true } };
  const dir = buildWorkspace(rollout, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const classifier_result = { confidence_band: "high", final_execution_decision: "gated_parallel_allowed" };
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW, classifier_result });
    assert.equal(result.effective_exposure_decision, "gated_parallel_allowed");
    assert.equal(result.routing_decision_source, "classifier_unavailable_fallback");
  } finally {
    cleanup(dir);
  }
});

test("WS-29-01: classifier low confidence -> falls back to static eligibility", () => {
  const dir = buildWorkspace(DYNAMIC_ROLLOUT, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const classifier_result = { confidence_band: "low" };
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW, classifier_result });
    assert.equal(result.effective_exposure_decision, "gated_parallel_allowed");
    assert.equal(result.routing_decision_source, "classifier_low_confidence_fallback");
  } finally {
    cleanup(dir);
  }
});

test("WS-29-01: classifier recommends parallel -> gated_parallel_allowed", () => {
  const dir = buildWorkspace(DYNAMIC_ROLLOUT, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const classifier_result = { confidence_band: "high", final_execution_decision: "gated_parallel_allowed", model_tier: "deep_reasoning" };
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW, classifier_result });
    assert.equal(result.effective_exposure_decision, "gated_parallel_allowed");
    assert.equal(result.routing_decision_source, "classifier_recommended_parallel");
    assert.equal(result.model_tier, "deep_reasoning");
  } finally {
    cleanup(dir);
  }
});

test("WS-29-01: classifier recommends sequential -> forced_sequential", () => {
  const dir = buildWorkspace(DYNAMIC_ROLLOUT, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const classifier_result = { confidence_band: "high", final_execution_decision: "forced_sequential", model_tier: "fast_low_cost" };
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW, classifier_result });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.routing_decision_source, "classifier_recommended_sequential");
    assert.equal(result.model_tier, "fast_low_cost");
  } finally {
    cleanup(dir);
  }
});

test("WS-29-01: classifier recommends parallel but structural guard denies", () => {
  const dir = buildWorkspace(DYNAMIC_ROLLOUT, BASE_POLICY);
  const misconfiguredWorkflow = {
    steps: [{ id: "impl_fe" }] // missing fe_safe_input_classes
  };
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const classifier_result = { confidence_band: "high", final_execution_decision: "gated_parallel_allowed", model_tier: "deep_reasoning" };
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: misconfiguredWorkflow, classifier_result });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.routing_decision_source, "classifier_recommended_parallel");
    assert.equal(result.deny_reason_code, "structural_completion_impossible");
  } finally {
    cleanup(dir);
  }
});

// ── QA admission tests (WS-24.5-03) ─────────────────────────────────────────

const PARALLEL_GATE = { effective_exposure_decision: "gated_parallel_allowed" };

test("WS-24-04: QA admission allowed when both branches succeeded", () => {
  const stepRows = [
    { step_id: "impl_be", status: "succeeded" },
    { step_id: "impl_fe", status: "succeeded" },
  ];
  const result = evaluateQaAdmission({ stepRows, gateDecision: PARALLEL_GATE });
  assert.equal(result.allowed, true);
});

test("WS-24-04: QA admission denied when BE branch failed", () => {
  const stepRows = [
    { step_id: "impl_be", status: "failed" },
    { step_id: "impl_fe", status: "succeeded" },
  ];
  const result = evaluateQaAdmission({ stepRows, gateDecision: PARALLEL_GATE });
  assert.equal(result.allowed, false);
  assert.ok(result.denial_reason?.startsWith("be_branch_not_succeeded"));
});

test("WS-24-04: release gating blocked when workflow in partial_failure", () => {
  const result = evaluateReleaseGating({
    workflowRun: { status: "partial_failure" },
    stepRows: [],
    gateDecision: PARALLEL_GATE,
  });
  assert.equal(result.allowed, false);
  assert.equal(result.denial_reason, "workflow_in_partial_failure");
});
