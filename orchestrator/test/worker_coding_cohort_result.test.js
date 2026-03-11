import assert from "node:assert/strict";

import {
  deriveFailureAttribution,
  summarizeResult,
  verificationTargetSatisfied,
} from "../scripts/run_worker_coding_cohort.js";

function main() {
  assert.equal(
    verificationTargetSatisfied("lint + unit_test", "syntax_check + lint + unit_test"),
    true,
  );
  assert.equal(
    verificationTargetSatisfied("lint + build", "syntax_check + lint"),
    false,
  );

  assert.equal(
    deriveFailureAttribution({}, "E_UNAUTHORIZED_WRITE"),
    "scope_guard_failure",
  );

  const result = summarizeResult({
    task: {
      cohort_task_id: "c-be-01",
      task_class: "be_create",
      beta_template_id: "wc.be_create.v1",
      verification_tier_target: "lint + unit_test",
      scenario: "Create backend endpoint",
    },
    terminal: {
      run: {
        status: "succeeded",
        workflow_run_id: "wf-1",
        run_id: "run-1",
      },
      steps: [],
    },
    focusedStep: {
      step_id: "impl_be",
      status: "succeeded",
      task_id: "task-1",
      result_json: JSON.stringify({
        output: {
          files_changed: ["sandbox/crm_site/server.js"],
          artifact_check: { checked: true, missing: [] },
          diagnostics: {
            verification: {
              checked: true,
              ok: true,
              achieved_tiers: ["syntax_check", "lint", "unit_test"],
            },
          },
        },
      }),
    },
  });

  assert.equal(result.result, "pass");
  assert.equal(result.failure_attribution, "none");
  assert.equal(result.verification_tier_achieved, "syntax_check + lint + unit_test");

  console.log("worker_coding_cohort_result.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("worker_coding_cohort_result.test.js: failed");
  console.error(err);
  process.exit(1);
}
