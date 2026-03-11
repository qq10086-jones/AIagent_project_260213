import assert from "node:assert/strict";

import {
  shouldRetryAutoFix,
  buildAttemptedFixes,
  buildFinalFailureSummary,
} from "../retry_policy.js";

function main() {
  const retryDecision = shouldRetryAutoFix({
    summary: {
      ok: false,
      error: "verification failed",
      diagnostics: {
        failed_phase: "verification",
        error_code: "E_VERIFICATION_FAILED",
      },
    },
    attemptIndex: 1,
    maxAttempts: 3,
    sameErrorRepeatLimit: 3,
    attemptRecords: [],
    errorCounts: new Map(),
  });
  assert.equal(retryDecision.retry, true);

  const deniedDecision = shouldRetryAutoFix({
    summary: {
      ok: false,
      error: "syntax issue",
      diagnostics: {
        failed_phase: "static_check",
        error_code: "E_STATIC_CHECK_FAILED",
      },
    },
    attemptIndex: 3,
    maxAttempts: 3,
    sameErrorRepeatLimit: 3,
    attemptRecords: [],
    errorCounts: new Map(),
  });
  assert.equal(deniedDecision.retry, false);
  assert.equal(deniedDecision.reason, "attempt_budget_exhausted");

  const attempted = buildAttemptedFixes({
    adapterResult: {
      provider_used: "opencode",
      diagnostics: { fallback_from: "codex" },
    },
    staticCheck: { checked: true, ok: false },
    verification: { checked: true, ok: true },
  });
  assert.deepEqual(attempted, [
    "delegate_once",
    "provider_fallback:codex->opencode",
    "static_check:failed",
    "verification:passed",
  ]);

  const finalSummary = buildFinalFailureSummary({
    summary: {
      error: "token=secret12345",
      command_used: "mock-inline-autofix",
      diagnostics: {
        failed_phase: "verification",
        error_code: "E_VERIFICATION_FAILED",
        verification: { command: "node --check sandbox/crm_site/app.js" },
      },
    },
    attemptRecords: [{ attempt: 1, phase: "verification" }],
    maxAttempts: 2,
    sameErrorRepeatLimit: 2,
    wallClockTimeoutS: 300,
    terminalReason: "attempt_budget_exhausted",
    startedAt: "2026-03-11T00:00:00.000Z",
  });
  assert.equal(finalSummary.attempts_used, 1);
  assert.equal(finalSummary.failed_phase, "verification");
  assert.match(finalSummary.error, /\[REDACTED\]/);

  console.log("retry_policy.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("retry_policy.test.js: failed");
  console.error(err);
  process.exit(1);
}
