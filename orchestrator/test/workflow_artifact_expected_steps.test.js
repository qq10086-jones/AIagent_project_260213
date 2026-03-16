import test from "node:test";
import assert from "node:assert/strict";

import { getExpectedWorkflowStepCount } from "../src/domain/workflow_artifact_audit.js";

test("getExpectedWorkflowStepCount returns registry workflow step length", () => {
  const registry = {
    workflows: {
      coding_team_v0: {
        steps: [
          { id: "pm_spec" },
          { id: "arch_design" },
          { id: "impl_be" },
          { id: "impl_fe" },
          { id: "qa_verify" },
          { id: "release_pack" },
          { id: "deploy_preview" },
        ],
      },
    },
  };

  assert.equal(getExpectedWorkflowStepCount(registry, "coding_team_v0"), 7);
});

test("getExpectedWorkflowStepCount returns 0 when workflow is absent", () => {
  assert.equal(getExpectedWorkflowStepCount({}, "missing_workflow"), 0);
  assert.equal(getExpectedWorkflowStepCount({ workflows: {} }, ""), 0);
});
