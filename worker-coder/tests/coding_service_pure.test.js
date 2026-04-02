/**
 * Unit tests for pure utility functions in coding_service.js:
 *   - payloadToAdapterRequest: input normalization / shape contract
 *   - requiresScopedTargetPaths: boolean gate logic
 *
 * No filesystem, no network; pure JS only.
 */
import assert from "node:assert/strict";
import {
  buildStepContractExpectedArtifacts,
  payloadToAdapterRequest,
  requiresScopedTargetPaths,
} from "../coding_service.js";

function main() {
  assert.equal(requiresScopedTargetPaths("impl_be", []), true);
  assert.equal(requiresScopedTargetPaths("impl_fe", []), true);
  assert.equal(requiresScopedTargetPaths("IMPL_BE", []), true);
  assert.equal(requiresScopedTargetPaths("IMPL_FE", []), true);

  assert.equal(requiresScopedTargetPaths("pm_spec", []), false);
  assert.equal(requiresScopedTargetPaths("arch_design", []), false);
  assert.equal(requiresScopedTargetPaths("release_pack", []), false);
  assert.equal(requiresScopedTargetPaths("", []), false);
  assert.equal(requiresScopedTargetPaths(null, []), false);
  assert.equal(requiresScopedTargetPaths(undefined, undefined), false);

  assert.equal(requiresScopedTargetPaths("pm_spec", ["src/foo.js"]), true);
  assert.equal(requiresScopedTargetPaths("arch_design", ["src/bar.js"]), true);
  assert.equal(requiresScopedTargetPaths("", ["x"]), true);
  assert.equal(requiresScopedTargetPaths("pm_spec", [null, undefined, ""]), false);
  assert.equal(requiresScopedTargetPaths("pm_spec", ["", "src/real.js"]), true);

  const implBeArtifacts = buildStepContractExpectedArtifacts({
    stepId: "impl_be",
    expectedArtifacts: ["impl/be_changes/server.js"],
  });
  assert.ok(implBeArtifacts.includes("impl/be_changes/server.js"));
  assert.ok(implBeArtifacts.includes("impl/be_changes/package.json"));
  assert.ok(implBeArtifacts.includes("impl/be_notes.md"));
  assert.ok(implBeArtifacts.includes("handoff/be_to_fe.json"));

  const passthroughArtifacts = buildStepContractExpectedArtifacts({
    stepId: "unknown_step",
    expectedArtifacts: ["foo.txt"],
  });
  assert.deepEqual(passthroughArtifacts, ["foo.txt"]);

  const full = payloadToAdapterRequest({
    provider: "opencode",
    task_prompt: "build a feature",
    artifact_root: "artifacts/release/run-1",
    expected_artifacts: ["plan/spec.md"],
    step_id: "pm_spec",
    target_paths: ["src/foo.js"],
    verification_command: "npm test",
    verification_plan: [{ command: "npm test" }],
    wall_clock_timeout_s: 300,
    execution_adapter_packet: { key: "val" },
    context_packet: { ctx: true },
    repo_map: { files: [] },
    model: "minimax-coding-plan/MiniMax-M2.7",
    execution_lane: "stable_cloud_lane",
    allow_provider_fallback: true,
    runtime_coder_config: { provider_default: "opencode" },
    run_id: "run-abc",
    task_id: "task-xyz",
    artifact_workspace_root: "/workspace/isolated",
  });

  assert.equal(full.adapter_type, "coding_executor");
  assert.equal(full.provider, "opencode");
  assert.equal(full.task_type, "coding_execution");
  assert.equal(full.payload.step_id, "pm_spec");
  assert.equal(full.payload.task_prompt, "build a feature");
  assert.equal(full.payload.artifact_root, "artifacts/release/run-1");
  assert.deepEqual(full.payload.expected_artifacts, ["plan/spec.md"]);
  assert.deepEqual(full.payload.target_paths, ["src/foo.js"]);
  assert.equal(full.payload.verification_command, "npm test");
  assert.deepEqual(full.payload.verification_plan, [{ command: "npm test" }]);
  assert.equal(full.payload.wall_clock_timeout_s, 300);
  assert.deepEqual(full.payload.execution_adapter_packet, { key: "val" });
  assert.deepEqual(full.payload.context_packet, { ctx: true });
  assert.deepEqual(full.payload.repo_map, { files: [] });
  assert.equal(full.payload.model_hint, "minimax-coding-plan/MiniMax-M2.7");
  assert.equal(full.payload.execution_lane, "stable_cloud_lane");
  assert.equal(full.payload.allow_provider_fallback, true);
  assert.deepEqual(full.payload.runtime_coder_config, { provider_default: "opencode" });
  assert.equal(full.context.run_id, "run-abc");
  assert.equal(full.context.task_id, "task-xyz");
  assert.equal(full.context.artifact_workspace_root, "/workspace/isolated");

  const empty = payloadToAdapterRequest({});
  assert.equal(empty.provider, "opencode");
  assert.equal(empty.payload.step_id, "");
  assert.equal(empty.payload.task_prompt, "");
  assert.equal(empty.payload.artifact_root, "");
  assert.deepEqual(empty.payload.expected_artifacts, []);
  assert.deepEqual(empty.payload.target_paths, []);
  assert.equal(empty.payload.verification_command, "");
  assert.deepEqual(empty.payload.verification_plan, []);
  assert.equal(empty.payload.wall_clock_timeout_s, 0);
  assert.equal(empty.payload.execution_adapter_packet, null);
  assert.equal(empty.payload.context_packet, null);
  assert.equal(empty.payload.repo_map, null);
  assert.equal(empty.payload.model_hint, "");
  assert.equal(empty.payload.execution_lane, "");
  assert.equal(empty.payload.allow_provider_fallback, false);
  assert.equal(empty.payload.runtime_coder_config, null);
  assert.equal(empty.context.run_id, "");
  assert.equal(empty.context.task_id, "");
  assert.equal(empty.context.artifact_workspace_root, "");

  const coerced = payloadToAdapterRequest({
    expected_artifacts: "plan/spec.md",
    target_paths: null,
    verification_plan: "echo ok",
  });
  assert.deepEqual(coerced.payload.expected_artifacts, []);
  assert.deepEqual(coerced.payload.target_paths, []);
  assert.deepEqual(coerced.payload.verification_plan, []);

  const fallbackTrue = payloadToAdapterRequest({ allow_provider_fallback: 1 });
  assert.equal(fallbackTrue.payload.allow_provider_fallback, true);
  const fallbackFalse = payloadToAdapterRequest({ allow_provider_fallback: 0 });
  assert.equal(fallbackFalse.payload.allow_provider_fallback, false);

  const timeout = payloadToAdapterRequest({ wall_clock_timeout_s: "600" });
  assert.equal(timeout.payload.wall_clock_timeout_s, 600);

  console.log("coding_service_pure.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("coding_service_pure.test.js: failed");
  console.error(err);
  process.exit(1);
}
