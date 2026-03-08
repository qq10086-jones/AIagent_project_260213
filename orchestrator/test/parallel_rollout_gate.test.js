/**
 * WS-24-04 Contract Validation Integration Tests
 * Covers:
 *  - FE-safe case proceeds under gated parallel path
 *  - Non-FE-safe case remains sequential
 *  - Rollout master disabled → sequential
 *  - Force-sequential override → sequential
 *  - Circuit-breaker activated → sequential
 *  - NEGATIVE TEST: policy declares FE-safe input_class but workflow impl_fe has no
 *    fe_safe_input_classes → structural_completion_impossible
 *  - QA admission: allowed only when both branches succeeded
 *  - Release gating: blocked on partial_failure
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { createParallelRolloutGate } from "../src/domain/parallel_rollout_gate.js";
import { evaluateQaAdmission, evaluateReleaseGating } from "../src/domain/parallel_qa_admission_guard.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Test fixture workspace ────────────────────────────────────────────────────

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
  circuit_breaker: { activated: false },
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

// ── Gate tests ────────────────────────────────────────────────────────────────

test("WS-24-04: FE-safe case proceeds under gated parallel path", () => {
  const dir = buildWorkspace(BASE_ROLLOUT, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    assert.equal(result.effective_exposure_decision, "gated_parallel_allowed");
    assert.equal(result.effective_exposure_decision_source, "eligibility_policy_allowed");
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
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04: rollout master disabled → forced_sequential (rollout_master_disabled)", () => {
  const dir = buildWorkspace({ ...BASE_ROLLOUT, master_enabled: false }, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.effective_exposure_decision_source, "rollout_master_disabled");
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04: missing rollout config → forced_sequential (rollout_master_disabled)", () => {
  const dir = buildWorkspace(null, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.effective_exposure_decision_source, "rollout_master_disabled");
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04: force_sequential override → forced_sequential (force_sequential_override)", () => {
  const dir = buildWorkspace({ ...BASE_ROLLOUT, force_sequential: true }, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.effective_exposure_decision_source, "force_sequential_override");
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04: circuit-breaker activated → forced_sequential", () => {
  const rollout = { ...BASE_ROLLOUT, circuit_breaker: { activated: true } };
  const dir = buildWorkspace(rollout, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: FE_SAFE_WORKFLOW });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.deny_reason_code, "circuit_breaker_activated");
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04: unapproved workflow_type → forced_sequential", () => {
  const dir = buildWorkspace(BASE_ROLLOUT, BASE_POLICY);
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({
      run: { ...FE_SAFE_RUN, workflow_id: "unknown_workflow" },
      workflow: FE_SAFE_WORKFLOW,
    });
    assert.equal(result.effective_exposure_decision, "forced_sequential");
    assert.equal(result.deny_reason_code, "unapproved_workflow_class");
  } finally {
    cleanup(dir);
  }
});

test("WS-24-04 NEGATIVE TEST: policy allows input_class but impl_fe has no fe_safe_input_classes → structural_completion_impossible", () => {
  // This is the negative misconfiguration test required by WS-24-04.
  // The eligibility policy declares "fe_led" as FE-safe, but the workflow's
  // impl_fe step has no fe_safe_input_classes defined, meaning FE structurally
  // requires BE material. The runtime structural guard must catch this and deny.
  const dir = buildWorkspace(BASE_ROLLOUT, BASE_POLICY);
  const misconfiguredWorkflow = {
    steps: [
      { id: "pm_spec" },
      { id: "arch_design" },
      { id: "impl_be" },
      { id: "impl_fe" }, // deliberately missing fe_safe_input_classes
      { id: "qa_verify" },
      { id: "release_pack" },
    ],
  };
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({ run: FE_SAFE_RUN, workflow: misconfiguredWorkflow });
    assert.equal(result.effective_exposure_decision, "forced_sequential",
      "structural guard must deny when fe_safe_input_classes is missing");
    assert.equal(result.deny_reason_code, "structural_completion_impossible",
      "denial reason must be structural_completion_impossible");
  } finally {
    cleanup(dir);
  }
});

// ── QA admission tests (WS-24.5-03) ─────────────────────────────────────────

const PARALLEL_GATE = { effective_exposure_decision: "gated_parallel_allowed" };
const SEQUENTIAL_GATE = { effective_exposure_decision: "forced_sequential" };

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

test("WS-24-04: QA admission denied when FE branch still running", () => {
  const stepRows = [
    { step_id: "impl_be", status: "succeeded" },
    { step_id: "impl_fe", status: "running" },
  ];
  const result = evaluateQaAdmission({ stepRows, gateDecision: PARALLEL_GATE });
  assert.equal(result.allowed, false);
  assert.ok(result.denial_reason?.startsWith("fe_branch_not_succeeded"));
});

test("WS-24-04: QA admission always allowed in sequential mode", () => {
  const stepRows = [{ step_id: "impl_be", status: "failed" }];
  const result = evaluateQaAdmission({ stepRows, gateDecision: SEQUENTIAL_GATE });
  assert.equal(result.allowed, true);
});

// ── Release gating tests (WS-24.5-03) ────────────────────────────────────────

test("WS-24-04: release gating blocked when workflow in partial_failure", () => {
  const result = evaluateReleaseGating({
    workflowRun: { status: "partial_failure" },
    stepRows: [],
    gateDecision: PARALLEL_GATE,
  });
  assert.equal(result.allowed, false);
  assert.equal(result.denial_reason, "workflow_in_partial_failure");
});

test("WS-24-04: release gating blocked when parallel branches not both succeeded", () => {
  const stepRows = [
    { step_id: "impl_be", status: "succeeded" },
    { step_id: "impl_fe", status: "failed" },
    { step_id: "qa_verify", status: "succeeded" },
  ];
  const result = evaluateReleaseGating({
    workflowRun: { status: "running" },
    stepRows,
    gateDecision: PARALLEL_GATE,
  });
  assert.equal(result.allowed, false);
  assert.equal(result.denial_reason, "parallel_branches_not_both_succeeded");
});

test("WS-24-04: release gating allowed when all parallel steps succeeded", () => {
  const stepRows = [
    { step_id: "impl_be", status: "succeeded" },
    { step_id: "impl_fe", status: "succeeded" },
    { step_id: "qa_verify", status: "succeeded" },
  ];
  const result = evaluateReleaseGating({
    workflowRun: { status: "running" },
    stepRows,
    gateDecision: PARALLEL_GATE,
  });
  assert.equal(result.allowed, true);
});
