/**
 * Unit tests for pure utility functions in coding_service.js:
 *   - payloadToAdapterRequest: input normalization / shape contract
 *   - requiresScopedTargetPaths: boolean gate logic
 *
 * No filesystem, no network — pure JS only.
 */
import assert from "node:assert/strict";
import { payloadToAdapterRequest, requiresScopedTargetPaths } from "../coding_service.js";

function main() {
  // ── requiresScopedTargetPaths ─────────────────────────────────────────────

  // impl steps always require scoped paths
  assert.equal(requiresScopedTargetPaths("impl_be", []), true, "impl_be + empty array → true");
  assert.equal(requiresScopedTargetPaths("impl_fe", []), true, "impl_fe + empty array → true");
  assert.equal(requiresScopedTargetPaths("IMPL_BE", []), true, "case-insensitive impl_be → true");
  assert.equal(requiresScopedTargetPaths("IMPL_FE", []), true, "case-insensitive impl_fe → true");

  // non-impl steps without paths → false
  assert.equal(requiresScopedTargetPaths("pm_spec", []), false, "pm_spec + empty → false");
  assert.equal(requiresScopedTargetPaths("arch_design", []), false, "arch_design + empty → false");
  assert.equal(requiresScopedTargetPaths("release_pack", []), false, "release_pack + empty → false");
  assert.equal(requiresScopedTargetPaths("", []), false, "empty stepId + empty paths → false");
  assert.equal(requiresScopedTargetPaths(null, []), false, "null stepId → false");
  assert.equal(requiresScopedTargetPaths(undefined, undefined), false, "undefined inputs → false");

  // any step with non-empty target_paths → true
  assert.equal(requiresScopedTargetPaths("pm_spec", ["src/foo.js"]), true, "pm_spec + paths → true");
  assert.equal(requiresScopedTargetPaths("arch_design", ["src/bar.js"]), true, "arch_design + paths → true");
  assert.equal(requiresScopedTargetPaths("", ["x"]), true, "empty stepId + paths → true");

  // falsy items in array are filtered out
  assert.equal(requiresScopedTargetPaths("pm_spec", [null, undefined, ""]), false, "only falsy items → false");
  assert.equal(requiresScopedTargetPaths("pm_spec", ["", "src/real.js"]), true, "mixed falsy/real → true");

  // ── payloadToAdapterRequest ───────────────────────────────────────────────

  // Full payload round-trip: all fields supplied
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
    model: "qwen3-coder-plus",
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
  assert.equal(full.payload.model_hint, "qwen3-coder-plus");
  assert.equal(full.payload.execution_lane, "stable_cloud_lane");
  assert.equal(full.payload.allow_provider_fallback, true);
  assert.deepEqual(full.payload.runtime_coder_config, { provider_default: "opencode" });
  assert.equal(full.context.run_id, "run-abc");
  assert.equal(full.context.task_id, "task-xyz");
  assert.equal(full.context.artifact_workspace_root, "/workspace/isolated");

  // Defaults for missing / falsy inputs
  const empty = payloadToAdapterRequest({});
  assert.equal(empty.provider, "opencode", "provider defaults to opencode");
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

  // Non-array expected_artifacts / target_paths → coerced to []
  const coerced = payloadToAdapterRequest({
    expected_artifacts: "plan/spec.md",
    target_paths: null,
    verification_plan: "echo ok",
  });
  assert.deepEqual(coerced.payload.expected_artifacts, []);
  assert.deepEqual(coerced.payload.target_paths, []);
  assert.deepEqual(coerced.payload.verification_plan, []);

  // allow_provider_fallback: truthy string → true, 0 → false
  const fallbackTrue = payloadToAdapterRequest({ allow_provider_fallback: 1 });
  assert.equal(fallbackTrue.payload.allow_provider_fallback, true);
  const fallbackFalse = payloadToAdapterRequest({ allow_provider_fallback: 0 });
  assert.equal(fallbackFalse.payload.allow_provider_fallback, false);

  // wall_clock_timeout_s: string number → coerced
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
