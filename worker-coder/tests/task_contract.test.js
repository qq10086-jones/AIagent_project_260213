import assert from "node:assert/strict";

import {
  buildTaskContractMetadata,
  deriveFailureAttribution,
} from "../task_contract.js";

function testMetadataNormalization() {
  const metadata = buildTaskContractMetadata({
    taskClass: "FE_Modify",
    betaTemplateId: "tpl-fe-modify-v1",
    contextEnvelope: {
      max_files: 15,
      max_tokens: "24000",
      dependency_depth: 2.8,
      context_source: "Template",
    },
  });
  assert.deepEqual(metadata, {
    task_class: "fe_modify",
    beta_template_id: "tpl-fe-modify-v1",
    context_envelope: {
      max_files: 15,
      max_tokens: 24000,
      dependency_depth: 2,
      context_source: "template",
    },
  });
}

function testFailureAttribution() {
  assert.equal(deriveFailureAttribution({ phase: "verification", errorCode: "E_VERIFICATION_FAILED" }), "verification_failure");
  assert.equal(deriveFailureAttribution({ phase: "context_guard", errorCode: "E_CONTEXT_ENVELOPE_EXCEEDED" }), "context_failure");
  assert.equal(deriveFailureAttribution({ phase: "delegate", errorCode: "E_PROVIDER_UNAVAILABLE" }), "infrastructure_failure");
  assert.equal(deriveFailureAttribution({ phase: "static_check", errorCode: "E_STATIC_CHECK_FAILED" }), "coding_logic_failure");
}

function main() {
  testMetadataNormalization();
  testFailureAttribution();
  console.log("task_contract.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("task_contract.test.js: failed");
  console.error(err);
  process.exit(1);
}
