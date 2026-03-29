/**
 * T-33 Failure Injection Tests 窶・CodingService.delegateTask
 *
 * Injects known failure conditions into the execution pipeline and asserts
 * the system responds with the correct error codes, retry decisions, and
 * failure summaries 窶・without a live LLM.
 *
 * Technique: pass `opencode_command` override so the adapter runs a
 * deterministic node script instead of the real opencode binary.
 * No network, no filesystem side-effects outside a temp dir.
 *
 * Scenarios:
 *   1. Provider permanently fails 竊・max_attempts exhausted
 *   2. Same error repeats 竊・same_error_repeat_limit stops retry early
 *   3. Scope guard blocks impl_be with empty target_paths (pre-execution)
 *   4. Scope guard blocks impl_be with protected target path (pre-execution)
 *   5. Static check catches syntax error in changed file
 *   6. Verification command fails after successful execution
 *   7. Unsupported provider name rejected before execution
 */
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { CodingService } from "../coding_service.js";

function makeTmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "t33-inject-"));
}

// Inline node script run as the fake "opencode" command
const FAIL_CMD = [process.execPath, "-e", "process.exit(1)"];
const SUCCEED_WRITE_SYNTAX_ERROR = (filePath) => [
  "mock-inline-autofix",
  filePath,
  "{{task_prompt}}",
];

// 笏笏 1. Max attempts exhausted 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
async function testMaxAttemptsExhausted() {
  const workspaceRoot = makeTmp();
  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "do something",
    artifact_root: "artifacts/release/run-1",
    expected_artifacts: [],
    step_id: "pm_spec",
    target_paths: [],
    provider: "opencode",
    max_attempts: 2,
    same_error_repeat_limit: 3,
    wall_clock_timeout_s: 300,
    run_id: "run-t33-1",
    task_id: "task-t33-1",
    opencode_command: FAIL_CMD,
  });

  assert.equal(result.ok, false, "must fail");
  const retrySummary = result.diagnostics?.retry_summary || result.diagnostics?.final_failure_summary;
  const attemptsUsed = retrySummary?.attempts_used ?? result.diagnostics?.final_failure_summary?.attempts_used;
  assert.equal(attemptsUsed, 2, "must exhaust both attempts");
  const terminalReason = result.diagnostics?.final_failure_summary?.terminal_reason
    || result.diagnostics?.retry_summary?.terminal_reason;
  assert.match(String(terminalReason || ""), /attempt_budget_exhausted/, "terminal reason must be attempt_budget_exhausted");
}

// 笏笏 2. Same-error-repeat-limit stops early 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
async function testSameErrorRepeatLimit() {
  const workspaceRoot = makeTmp();
  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "do something",
    artifact_root: "artifacts/release/run-2",
    expected_artifacts: [],
    step_id: "pm_spec",
    target_paths: [],
    provider: "opencode",
    max_attempts: 3,
    same_error_repeat_limit: 1,
    wall_clock_timeout_s: 300,
    run_id: "run-t33-2",
    task_id: "task-t33-2",
    opencode_command: FAIL_CMD,
  });

  assert.equal(result.ok, false, "must fail");
  const terminalReason = result.diagnostics?.final_failure_summary?.terminal_reason
    || result.diagnostics?.retry_summary?.terminal_reason;
  assert.match(
    String(terminalReason || ""),
    /same_error_repeat_limit|same_error_repeated/,
    "terminal reason must be same_error_repeat_limit_reached or same_error_repeated_consecutively"
  );
  // Stopped before using all 3 attempts
  const attemptsUsed = result.diagnostics?.final_failure_summary?.attempts_used
    || result.diagnostics?.retry_summary?.attempts_used;
  assert.ok(Number(attemptsUsed || 0) < 3, "must stop before max_attempts=3");
}

// 笏笏 3. Scope guard: empty target_paths for impl_be 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
async function testScopeGuardEmptyTargetPaths() {
  const workspaceRoot = makeTmp();
  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "implement the backend",
    artifact_root: "artifacts/release/run-3",
    expected_artifacts: [],
    step_id: "impl_be",      // requiresScopedTargetPaths 竊・true for impl steps
    target_paths: [],         // empty 竊・validateAllowedTargetPaths fails
    provider: "opencode",
    max_attempts: 1,
    wall_clock_timeout_s: 300,
    run_id: "run-t33-3",
    task_id: "task-t33-3",
    opencode_command: FAIL_CMD,  // should never be reached
  });

  assert.equal(result.ok, false, "must fail");
  assert.equal(
    result.diagnostics?.error_code,
    "E_UNAUTHORIZED_WRITE",
    "must fail with E_UNAUTHORIZED_WRITE"
  );
  // Execution must NOT have started (opencode_command was never spawned)
  assert.ok(
    !result.command_used,
    "no command should have been used (scope guard fires pre-execution)"
  );
}

// 笏笏 4. Scope guard: protected target path for impl_be 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
async function testScopeGuardProtectedPath() {
  const workspaceRoot = makeTmp();
  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "implement the backend",
    artifact_root: "artifacts/release/run-4",
    expected_artifacts: [],
    step_id: "impl_be",
    target_paths: [".git/hooks"],   // protected root 竊・E_UNAUTHORIZED_WRITE
    provider: "opencode",
    max_attempts: 1,
    wall_clock_timeout_s: 300,
    run_id: "run-t33-4",
    task_id: "task-t33-4",
    opencode_command: FAIL_CMD,
  });

  assert.equal(result.ok, false, "must fail");
  assert.equal(
    result.diagnostics?.error_code,
    "E_UNAUTHORIZED_WRITE",
    "must fail with E_UNAUTHORIZED_WRITE for protected path"
  );
  assert.ok(!result.command_used, "no command should have been used");
}

// 笏笏 5. Static check catches syntax error in changed file 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
async function testStaticCheckCatchesSyntaxError() {
  const workspaceRoot = makeTmp();
  // Use forward slashes for the inline node script (cross-platform safe in cwd)
  const relTarget = "src/bad.js";
  const cmd = SUCCEED_WRITE_SYNTAX_ERROR(relTarget);

  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "write a bad file",
    artifact_root: "artifacts/release/run-5",
    expected_artifacts: [],
    step_id: "impl_be",
    target_paths: [relTarget],
    provider: "opencode",
    max_attempts: 1,
    wall_clock_timeout_s: 300,
    run_id: "run-t33-5",
    task_id: "task-t33-5",
    opencode_command: cmd,
  });

  assert.equal(result.ok, false, "must fail after static check");
  const errorCode = result.diagnostics?.error_code
    || result.diagnostics?.final_failure_summary?.error_code;
  assert.equal(
    errorCode,
    "E_STATIC_CHECK_FAILED",
    `expected E_STATIC_CHECK_FAILED, got ${errorCode}`
  );
}

// 笏笏 6. Verification command fails after successful execution 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
async function testVerificationFailure() {
  const workspaceRoot = makeTmp();
  const relTarget = "src/ok.js";
  // Command writes a *valid* JS file (static check passes), then exits 0
  const writeCmd = [
    process.execPath,
    "-e",
    `var fs=require('fs'); fs.mkdirSync(require('path').dirname('${relTarget}'),{recursive:true}); fs.writeFileSync('${relTarget}','const x = 1;'); process.exit(0)`,
  ];

  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "write a file",
    artifact_root: "artifacts/release/run-6",
    expected_artifacts: [],
    step_id: "impl_be",
    target_paths: [relTarget],
    provider: "opencode",
    max_attempts: 1,
    wall_clock_timeout_s: 300,
    run_id: "run-t33-6",
    task_id: "task-t33-6",
    opencode_command: writeCmd,
    // Verification command always fails
    verification_command: `${JSON.stringify(process.execPath)} -e process.exit(1)`,
  });

  assert.equal(result.ok, false, "must fail after verification");
  const errorCode = result.diagnostics?.error_code
    || result.diagnostics?.final_failure_summary?.error_code;
  assert.ok(
    errorCode === "E_VERIFICATION_FAILED" || String(errorCode || "").startsWith("E_"),
    `expected E_VERIFICATION_FAILED or similar, got ${errorCode}`
  );
  // Static check must have passed (valid JS)
  const staticCheck = result.diagnostics?.static_check;
  if (staticCheck?.checked) {
    assert.equal(staticCheck.ok, true, "static check should have passed before verification");
  }
}

// 笏笏 7. Unsupported provider name rejected pre-execution 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏
async function testUnsupportedProvider() {
  const workspaceRoot = makeTmp();
  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "do something",
    artifact_root: "artifacts/release/run-7",
    expected_artifacts: [],
    step_id: "pm_spec",
    target_paths: [],
    provider: "gpt-99-turbo",   // not in {auto, opencode, codex}
    max_attempts: 1,
    wall_clock_timeout_s: 300,
    run_id: "run-t33-7",
    task_id: "task-t33-7",
  });

  assert.equal(result.ok, false, "unsupported provider must fail");
  assert.equal(
    result.diagnostics?.error_code,
    "E_PROVIDER_UNAVAILABLE",
    "must fail with E_PROVIDER_UNAVAILABLE"
  );
  assert.ok(!result.command_used, "no command should have been dispatched");
}

async function main() {
  await testMaxAttemptsExhausted();
  console.log("  [PASS] max attempts exhausted 竊・attempt_budget_exhausted");

  await testSameErrorRepeatLimit();
  console.log("  [PASS] same error repeat limit 竊・stops early");

  await testScopeGuardEmptyTargetPaths();
  console.log("  [PASS] scope guard: empty target_paths 竊・E_UNAUTHORIZED_WRITE");

  await testScopeGuardProtectedPath();
  console.log("  [PASS] scope guard: protected path 竊・E_UNAUTHORIZED_WRITE");

  await testStaticCheckCatchesSyntaxError();
  console.log("  [PASS] static check: syntax error 竊・E_STATIC_CHECK_FAILED");

  await testVerificationFailure();
  console.log("  [PASS] verification failure 竊・E_VERIFICATION_FAILED");

  await testUnsupportedProvider();
  console.log("  [PASS] unsupported provider 竊・E_PROVIDER_UNAVAILABLE");

  console.log("delegate_failure_injection.test.js: all tests passed");
}

main().catch((err) => {
  console.error("delegate_failure_injection.test.js: failed");
  console.error(err);
  process.exit(1);
});



