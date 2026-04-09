import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  normalizeRefinementLineage,
  resolveRefinementSkips,
  hasNonEmptyHandoffDeps,
} from "../src/domain/workflow_refinement_service.js";

const SAMPLE_STEPS = [
  { id: "pm_spec", role: "pm", tool: "coding.delegate" },
  { id: "arch_design", role: "architect", tool: "coding.delegate" },
  { id: "impl_be", role: "backend", tool: "coding.delegate" },
  { id: "impl_fe", role: "frontend", tool: "coding.delegate" },
  { id: "smoke_test", role: "qa", tool: "coding.execute" },
  { id: "qa_verify", role: "qa", tool: "coding.delegate" },
  { id: "release_pack", role: "release", tool: "coding.delegate" },
  { id: "deploy_preview", role: "ops", tool: "ops.deploy_preview" },
];

describe("normalizeRefinementLineage", () => {
  it("returns ok:false for null", () => {
    const result = normalizeRefinementLineage(null);
    assert.equal(result.ok, false);
    assert.equal(result.lineage, null);
  });

  it("returns ok:false for missing parent_task_id", () => {
    const result = normalizeRefinementLineage({ refinement_round: 1 });
    assert.equal(result.ok, false);
  });

  it("returns ok:false for invalid round", () => {
    const result = normalizeRefinementLineage({ parent_task_id: "abc", refinement_round: 0 });
    assert.equal(result.ok, false);
  });

  it("returns ok:false for round > 5", () => {
    const result = normalizeRefinementLineage({ parent_task_id: "abc", refinement_round: 6 });
    assert.equal(result.ok, false);
    assert.ok(result.error.includes("exceeds max"));
  });

  it("normalizes valid lineage", () => {
    const result = normalizeRefinementLineage({
      parent_task_id: "task_123",
      parent_run_id: "run_456",
      refinement_round: 2,
      refinement_instruction: "change API format",
      refinement_task_class: "be_modify",
      inherited_context: { target_paths: ["src/"] },
    });
    assert.equal(result.ok, true);
    assert.equal(result.lineage.parent_task_id, "task_123");
    assert.equal(result.lineage.refinement_round, 2);
    assert.equal(result.lineage.refinement_task_class, "be_modify");
  });
});

describe("resolveRefinementSkips", () => {
  it("skips nothing when no task_class", () => {
    const result = resolveRefinementSkips({ lineage: {}, steps: SAMPLE_STEPS });
    assert.deepEqual(result.skipStepIds, []);
  });

  it("fe_modify: skips pm_spec, arch_design, impl_be, release_pack, deploy_preview", () => {
    const result = resolveRefinementSkips({
      lineage: { refinement_task_class: "fe_modify" },
      steps: SAMPLE_STEPS,
    });
    assert.ok(result.skipStepIds.includes("pm_spec"));
    assert.ok(result.skipStepIds.includes("arch_design"));
    assert.ok(result.skipStepIds.includes("impl_be"));
    assert.ok(result.skipStepIds.includes("release_pack"));
    assert.ok(result.skipStepIds.includes("deploy_preview"));
    assert.ok(!result.skipStepIds.includes("impl_fe"));
    assert.ok(!result.skipStepIds.includes("smoke_test"));
    assert.ok(!result.skipStepIds.includes("qa_verify"));
  });

  it("be_modify: skips impl_fe unless handoff has deps", () => {
    const result = resolveRefinementSkips({
      lineage: { refinement_task_class: "be_modify" },
      steps: SAMPLE_STEPS,
      handoffHasInterfaceDeps: false,
    });
    assert.ok(result.skipStepIds.includes("impl_fe"));

    const withDeps = resolveRefinementSkips({
      lineage: { refinement_task_class: "be_modify" },
      steps: SAMPLE_STEPS,
      handoffHasInterfaceDeps: true,
    });
    assert.ok(!withDeps.skipStepIds.includes("impl_fe"));
  });

  it("bug_fix: skips only pm_spec, arch_design, release_pack, deploy_preview", () => {
    const result = resolveRefinementSkips({
      lineage: { refinement_task_class: "bug_fix" },
      steps: SAMPLE_STEPS,
    });
    assert.ok(result.skipStepIds.includes("pm_spec"));
    assert.ok(result.skipStepIds.includes("arch_design"));
    assert.ok(!result.skipStepIds.includes("impl_be"));
    assert.ok(!result.skipStepIds.includes("impl_fe"));
  });

  it("unknown task_class: skips nothing", () => {
    const result = resolveRefinementSkips({
      lineage: { refinement_task_class: "unknown_type" },
      steps: SAMPLE_STEPS,
    });
    assert.deepEqual(result.skipStepIds, []);
  });
});

describe("hasNonEmptyHandoffDeps", () => {
  it("returns false for null", () => {
    assert.equal(hasNonEmptyHandoffDeps(null), false);
  });

  it("returns false for empty handoff", () => {
    assert.equal(hasNonEmptyHandoffDeps({}), false);
    assert.equal(hasNonEmptyHandoffDeps({ api_contracts: [] }), false);
  });

  it("returns true when api_contracts non-empty", () => {
    assert.equal(hasNonEmptyHandoffDeps({ api_contracts: [{ path: "/api/foo" }] }), true);
  });

  it("returns true when shared_types non-empty", () => {
    assert.equal(hasNonEmptyHandoffDeps({ shared_types: ["User"] }), true);
  });
});
