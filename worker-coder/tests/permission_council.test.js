import assert from "node:assert/strict";

import {
  auditSafety,
  validateContext,
  scoreRisk,
  evaluatePermissionAdvisory,
} from "../../shared/permission_council.js";

function main() {
  // --- SafetyAuditor ---
  const safeCoding = auditSafety({ tool_name: "coding.delegate", reasons: [], requires_approval: false });
  assert.equal(safeCoding.verdict, "allow");

  const destructive = auditSafety({ tool_name: "coding.execute", reasons: ["destructive_command"], requires_approval: false });
  assert.equal(destructive.verdict, "deny");

  const needsApproval = auditSafety({ tool_name: "ops.deploy", reasons: ["policy_gate"], requires_approval: true });
  assert.equal(needsApproval.verdict, "review");

  const genericAllow = auditSafety({ tool_name: "docs.generate", reasons: [], requires_approval: false });
  assert.equal(genericAllow.verdict, "allow");

  // --- ContextValidator ---
  const hasPrompt = validateContext({ prompt: "fix navbar spacing" });
  assert.equal(hasPrompt.verdict, "sufficient");

  const noPrompt = validateContext({ prompt: "" });
  assert.equal(noPrompt.verdict, "insufficient");

  const nullPrompt = validateContext({});
  assert.equal(nullPrompt.verdict, "insufficient");

  // --- RiskScorer ---
  assert.equal(scoreRisk({ safety_verdict: "deny" }).score, 0.98);
  assert.equal(scoreRisk({ safety_verdict: "review" }).score, 0.75);
  assert.equal(scoreRisk({ safety_verdict: "allow", context_verdict: "insufficient" }).score, 0.55);
  assert.equal(scoreRisk({ safety_verdict: "allow", context_verdict: "sufficient", reasons: ["some_flag"] }).score, 0.45);
  assert.equal(scoreRisk({ safety_verdict: "allow", context_verdict: "sufficient", reasons: [] }).score, 0.2);

  // --- Composed Council ---
  const safe = evaluatePermissionAdvisory({
    tool_name: "coding.delegate",
    payload: { task_prompt: "fix navbar spacing issue" },
    risk: { risk_level: "medium", requires_approval: false, reasons: [] },
  });
  assert.equal(safe.council_advice, "allow");
  assert.equal(safe.safety_verdict, "allow");
  assert.equal(safe.context_verdict, "sufficient");
  assert.equal(safe.risk_score, 0.45);

  const deny = evaluatePermissionAdvisory({
    tool_name: "coding.execute",
    payload: { task_prompt: "rm -rf /workspace/tmp" },
    risk: { risk_level: "high", requires_approval: true, reasons: ["destructive_command"] },
  });
  assert.equal(deny.council_advice, "deny");
  assert.equal(deny.safety_verdict, "deny");
  assert.equal(deny.risk_score, 0.98);

  const escalated = evaluatePermissionAdvisory({
    tool_name: "coding.delegate",
    payload: {},
    risk: { risk_level: "low", requires_approval: false, reasons: [] },
  });
  assert.equal(escalated.council_advice, "review");
  assert.equal(escalated.context_verdict, "insufficient");

  console.log("permission_council.test.js: all tests passed");
}

main();
