import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";

import {
  buildAutoFixPrompt,
  buildExecutionPromptContract,
  writePromptContractArtifact,
} from "../prompt_contract.js";

function main() {
  const prompt = buildAutoFixPrompt({
    originalPrompt: "Implement the fix.",
    previousFailure: {
      phase: "verification",
      error_code: "E_VERIFICATION_FAILED",
      error: "tests failed",
      command_used: "node --check sandbox/crm_site/app.js",
    },
    attemptIndex: 2,
  });
  assert.match(prompt, /\[Auto-Fix Retry\]/);
  assert.match(prompt, /Attempt: 2/);
  assert.match(prompt, /E_VERIFICATION_FAILED/);

  const contract = buildExecutionPromptContract({
    stepId: "impl_fe",
    targetPaths: ["sandbox/crm_site/app.js"],
    expectedArtifacts: ["impl/fe_patch_bundle.json", "impl/fe_notes.md"],
    verificationCommand: "node --check sandbox/crm_site/app.js",
    verificationPlan: [
      { tier: "syntax_check", command: "node --check sandbox/crm_site/app.js" },
      { tier: "build", command: "npm run build" },
    ],
    executionAdapterPacket: { execution_mode: "structured_patch_from_backend_handoff", input_artifacts: ["handoff/be_to_fe.json"] },
    contextPacket: { entrypoints: ["sandbox/crm_site/app.js"], related_tests: ["worker-coder/tests/verification_command.test.js"] },
    repoMap: { candidate_files: ["sandbox/crm_site/app.js"] },
  });
  assert.match(contract, /step_id: impl_fe/);
  assert.match(contract, /target_paths: sandbox\/crm_site\/app\.js/);
  assert.match(contract, /verification_command: node --check sandbox\/crm_site\/app\.js/);
  assert.match(contract, /verification_plan: syntax_check:node --check sandbox\/crm_site\/app\.js, build:npm run build/);

  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "prompt-contract-"));
  const taskDir = path.join(tempRoot, "artifacts", "runs", "run-1", "task_1");
  fs.mkdirSync(taskDir, { recursive: true });
  const artifact = writePromptContractArtifact({
    workspaceRoot: tempRoot,
    taskDir,
    attemptIndex: 1,
    stepId: "impl_fe",
    basePrompt: "Implement frontend changes.",
    targetPaths: ["sandbox/crm_site/app.js"],
    expectedArtifacts: ["impl/fe_patch_bundle.json"],
    verificationCommand: "node --check sandbox/crm_site/app.js",
    verificationPlan: [{ tier: "syntax_check", command: "node --check sandbox/crm_site/app.js" }],
    executionAdapterPacket: { execution_mode: "structured_patch_from_backend_handoff" },
    contextPacket: { entrypoints: ["sandbox/crm_site/app.js"] },
    repoMap: { candidate_files: ["sandbox/crm_site/app.js"] },
  });
  assert.ok(artifact.path);
  assert.ok(fs.existsSync(path.join(tempRoot, artifact.path)));

  console.log("prompt_contract.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("prompt_contract.test.js: failed");
  console.error(err);
  process.exit(1);
}
