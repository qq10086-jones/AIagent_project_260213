/**
 * Tests for step_artifact_contract.js
 *
 * getWorkflowStepHandoff is a pure function — tested exhaustively.
 * validateWorkflowStepArtifacts requires live orchestrator modules;
 * covered at integration level via M10 load test / coding_team_e2e canary.
 */
import assert from "node:assert/strict";
import { getWorkflowStepHandoff } from "../step_artifact_contract.js";

function main() {
  // --- pm_spec ---
  const pmHandoff = getWorkflowStepHandoff("pm_spec");
  assert.ok(pmHandoff, "pm_spec should return a handoff");
  assert.equal(pmHandoff.from_step, "pm_spec");
  assert.deepEqual(pmHandoff.required_artifacts, ["handoff/pm_to_architect.json"]);
  assert.equal(pmHandoff.typed_handoff.file, "handoff/pm_to_architect.json");
  assert.ok(pmHandoff.typed_handoff.required_fields.includes("acceptance.criteria"), "pm handoff must require acceptance.criteria");

  // --- arch_design ---
  const archHandoff = getWorkflowStepHandoff("arch_design");
  assert.ok(archHandoff, "arch_design should return a handoff");
  assert.equal(archHandoff.from_step, "arch_design");
  assert.ok(archHandoff.required_artifacts.includes("plan/workplan.json"), "arch handoff must require structured workplan json");
  assert.ok(archHandoff.typed_handoff.required_fields.includes("decisions"), "arch handoff must require decisions array");
  assert.ok(archHandoff.typed_handoff.required_fields.includes("risks"), "arch handoff must require risks array");

  // --- impl_be ---
  const beHandoff = getWorkflowStepHandoff("impl_be");
  assert.ok(beHandoff, "impl_be should return a handoff");
  assert.equal(beHandoff.from_step, "impl_be");
  assert.ok(beHandoff.required_artifacts.includes("impl/be_changes/server.js"), "impl_be must require server.js");
  assert.ok(beHandoff.required_artifacts.includes("impl/be_changes/package.json"), "impl_be must require package.json");
  assert.ok(beHandoff.required_artifacts.includes("handoff/be_to_fe.json"), "impl_be must require be_to_fe.json");
  assert.ok(beHandoff.required_sections.includes("api_contracts"), "impl_be must require api_contracts section");

  // --- impl_fe ---
  const feHandoff = getWorkflowStepHandoff("impl_fe");
  assert.ok(feHandoff, "impl_fe should return a handoff");
  assert.equal(feHandoff.from_step, "impl_fe");
  assert.ok(feHandoff.required_artifacts.includes("impl/fe_changes/public/app.js"), "impl_fe must require public/app.js");
  assert.ok(feHandoff.typed_handoff.required_fields.includes("run_instructions"), "impl_fe handoff must require run_instructions");

  // --- release_pack --- (no handoff required)
  const releaseHandoff = getWorkflowStepHandoff("release_pack");
  assert.equal(releaseHandoff, null, "release_pack should return null (no handoff)");

  // --- unknown step ---
  assert.equal(getWorkflowStepHandoff("qa_verify"), null, "unknown step should return null");
  assert.equal(getWorkflowStepHandoff(""), null, "empty string should return null");
  assert.equal(getWorkflowStepHandoff(undefined), null, "undefined should return null");

  // --- immutability: each call returns a fresh object ---
  const h1 = getWorkflowStepHandoff("impl_be");
  const h2 = getWorkflowStepHandoff("impl_be");
  h1.required_artifacts.push("injected");
  assert.equal(h2.required_artifacts.length, beHandoff.required_artifacts.length,
    "each call to getWorkflowStepHandoff should return a fresh object");

  console.log("step_artifact_contract.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("step_artifact_contract.test.js: failed");
  console.error(err);
  process.exit(1);
}
