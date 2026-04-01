import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";

import {
  summarizeTerminalText,
  persistCodingFailureMemory,
  buildDelegateFailureSummary,
} from "../failure_memory.js";

function main() {
  const summary = summarizeTerminalText("line1\nline2\ntoken=secret12345");
  assert.match(summary, /\[REDACTED\]/);

  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "failure-memory-"));
  const persisted = persistCodingFailureMemory({
    workspaceRoot,
    runId: "run-1",
    taskId: "task-1",
    stepId: "impl_fe",
    providerRequested: "opencode",
    providerUsed: "opencode",
    targetPaths: ["workspace/sandbox/crm_site/app.js"],
    failedPhase: "verification",
    terminalOutcome: "failed",
    errorCode: "E_VERIFICATION_FAILED",
    error: "password=secret1234",
    verificationCommand: "node --check workspace/sandbox/crm_site/app.js",
    stderrText: "token=secret1234",
    stdoutText: "ok",
    attemptedFixes: ["delegate_once"],
    filesChanged: ["workspace/sandbox/crm_site/app.js"],
    artifactPaths: { test_log: "artifacts/test.json" },
  });
  assert.ok(persisted.latest_path);
  const latest = JSON.parse(fs.readFileSync(path.join(workspaceRoot, persisted.latest_path), "utf8"));
  assert.equal(latest.step_id, "impl_fe");
  assert.match(latest.error, /\[REDACTED\]/);

  const delegateFailure = buildDelegateFailureSummary({
    workspaceRoot,
    runId: "run-2",
    taskId: "task-2",
    stepId: "impl_be",
    providerRequested: "opencode",
    providerUsed: "opencode",
    targetPaths: ["workspace/sandbox/crm_site/server.js"],
    finalGitSummary: {
      filesChanged: ["workspace/sandbox/crm_site/server.js"],
      diffStats: { added: 10, deleted: 2, files: 1 },
      diffPath: "artifacts/runs/run-2/task_2/delegate_diff.json",
      git: { base_ref: "SCOPED_SNAPSHOT", branch: "scoped", commit_sha: null, dirty: true },
    },
    stdoutPath: "artifacts/runs/run-2/task_2/stdout.log",
    stderrPath: "artifacts/runs/run-2/task_2/stderr.log",
    staticCheck: { checked: true, ok: true, commands: ["node_syntax:workspace/sandbox/crm_site/server.js"], logPath: "artifacts/static_check.json" },
    verification: { checked: true, ok: false, command: "node --check workspace/sandbox/crm_site/server.js", logPath: "artifacts/verification.json" },
    adapterResult: {
      model_used: "qwen",
      command_used: "mock-inline-autofix",
      command_source: "payload.opencode_command",
      diagnostics: { fallback_from: null },
    },
    verificationCommand: "node --check workspace/sandbox/crm_site/server.js",
    promptContract: { path: "artifacts/prompt_contract.json", diagnostics: { contract_version: "execution_contract_v1" } },
    redactedStdout: "ok",
    redactedStderr: "token=[REDACTED]",
    startedAt: "2026-03-11T00:00:00.000Z",
    phase: "verification",
    summaryText: "verification failed",
    testResult: "failed",
    errorCode: "E_VERIFICATION_FAILED",
    error: "verification failed",
  });
  assert.equal(delegateFailure.ok, false);
  assert.equal(delegateFailure.diagnostics.failed_phase, "verification");
  assert.ok(delegateFailure.diagnostics.coding_failure_memory.latest_path);

  console.log("failure_memory.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("failure_memory.test.js: failed");
  console.error(err);
  process.exit(1);
}
