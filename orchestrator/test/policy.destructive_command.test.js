import { test } from "node:test";
import assert from "node:assert/strict";
import { analyzeTaskRisk } from "../src/policy.js";

const HIGH = [
  "run rm -rf / please",
  "git reset --hard origin/main",
  "del /f important.txt",
  "format c: and wipe disk",
  "format /q d:",
  "mkfs.ext4 /dev/sdb1",
  "dd if=/dev/zero of=/dev/sda",
];

const SAFE = [
  "render fields in label-value format including created_at/updated_at",
  "use JSON format for output",
  "reformat this text block",
  "normal spec for CRM list view",
  "format the response as markdown",
];

test("destructive_command regex flags genuinely dangerous commands", () => {
  for (const prompt of HIGH) {
    const r = analyzeTaskRisk("coding.delegate", { task_prompt: prompt });
    assert.ok(
      r.reasons.includes("destructive_command"),
      `expected destructive_command for: ${prompt} (got ${JSON.stringify(r.reasons)})`,
    );
    assert.equal(r.risk_level, "high");
    assert.equal(r.requires_approval, true);
  }
});

test("destructive_command regex does not false-positive on prose 'format'", () => {
  for (const prompt of SAFE) {
    const r = analyzeTaskRisk("coding.delegate", { task_prompt: prompt });
    assert.ok(
      !r.reasons.includes("destructive_command"),
      `unexpected destructive_command for: ${prompt} (reasons=${JSON.stringify(r.reasons)})`,
    );
  }
});
