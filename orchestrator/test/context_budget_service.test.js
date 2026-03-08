import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { createContextBudgetService } from "../src/domain/context_budget_service.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-context-budget-"));
}

function writeFile(workspaceRoot, relativePath, size) {
  const targetPath = path.join(workspaceRoot, relativePath);
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, "x".repeat(size), "utf8");
  return targetPath;
}

test("buildReport classifies normal usage as ok", () => {
  const workspaceRoot = makeWorkspace();
  const artifactPath = writeFile(workspaceRoot, "artifacts/one.txt", 128);
  const service = createContextBudgetService();

  const report = service.buildReport({
    stepId: "impl_be",
    role: "backend",
    prompt: "short prompt",
    injectedContext: "small context",
    artifactPaths: [artifactPath],
  });

  assert.equal(report.status, "ok");
  assert.equal(report.artifact_count, 1);
  assert.equal(report.bytes_total, 128);
});

test("buildReport uses role overrides and classifies warning/overflow_risk", () => {
  const service = createContextBudgetService();
  const warning = service.buildReport({
    stepId: "arch_design",
    role: "architect",
    prompt: "a".repeat(110000),
    injectedContext: "",
    artifactBytes: [200],
  });
  const overflow = service.buildReport({
    stepId: "impl_fe",
    role: "frontend",
    prompt: "small",
    injectedContext: "b".repeat(360000),
    artifactBytes: [100],
  });

  assert.equal(warning.status, "warning");
  assert.equal(warning.threshold_source, "role_overrides.architect");
  assert.equal(overflow.status, "overflow_risk");
  assert.equal(overflow.threshold_source, "role_overrides.frontend");
});
