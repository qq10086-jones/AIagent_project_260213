import assert from "node:assert/strict";

import { buildCodingExecutorRequest } from "../src/coding_executor.js";

function main() {
  const request = buildCodingExecutorRequest({
    provider: "opencode",
    role: "backend",
    stepId: "impl_be",
    payload: {
      workflow_id: "coding_team_v0",
      workflow_run_id: "wf-1",
      run_id: "run-1",
      role: "backend",
      step_id: "impl_be",
      task_prompt: "Implement backend",
      artifact_root: "artifacts/release/run-1",
      expected_artifacts: ["patch/diff.patch"],
      target_paths: ["sandbox/crm_site/"],
      verification_command: "node --check sandbox/crm_site/server.js",
      wall_clock_timeout_s: 480,
    },
  });

  assert.equal(
    request.payload.verification_command,
    "node --check sandbox/crm_site/server.js",
  );
  assert.equal(request.payload.wall_clock_timeout_s, 480);

  console.log("coding_executor.verification_command.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("coding_executor.verification_command.test.js: failed");
  console.error(err);
  process.exit(1);
}
