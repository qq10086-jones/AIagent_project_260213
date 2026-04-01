import assert from "node:assert/strict";

import { evaluatePermissionAdvisory } from "../../shared/permission_council.js";

function main() {
  const safe = evaluatePermissionAdvisory({
    tool_name: "coding.delegate",
    payload: { task_prompt: "fix navbar spacing issue" },
    risk: { risk_level: "medium", requires_approval: false, reasons: [] },
  });
  assert.equal(safe.council_advice, "allow");

  const deny = evaluatePermissionAdvisory({
    tool_name: "coding.execute",
    payload: { task_prompt: "rm -rf /workspace/tmp" },
    risk: { risk_level: "high", requires_approval: true, reasons: ["destructive_command"] },
  });
  assert.equal(deny.council_advice, "deny");
  assert.equal(deny.safety_verdict, "deny");

  console.log("permission_council.test.js: all tests passed");
}

main();
