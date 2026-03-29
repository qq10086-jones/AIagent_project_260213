import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { validateQaVerifierArtifacts } from "../src/qa_verifier.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-qa-verifier-"));
}

test("validateQaVerifierArtifacts accepts qa_report with journey and rubric metadata", () => {
  const workspaceRoot = makeWorkspace();
  const artifactRoot = "artifacts/release/run-qa";
  const qaPath = path.join(workspaceRoot, artifactRoot, "verify", "qa_report.json");
  fs.mkdirSync(path.dirname(qaPath), { recursive: true });
  fs.writeFileSync(qaPath, JSON.stringify({
    overall_status: "pass_with_warnings",
    checks: [
      {
        check_id: "qa-1",
        layer: "semantic",
        description: "Primary journey review",
        status: "warning",
        detail: "Pending manual confirmation.",
      },
    ],
    journey_checks: [
      {
        journey_id: "journey-1",
        description: "Primary happy path",
        status: "warning",
        evidence: ["Journey evidence not yet captured."],
      },
    ],
    rubric_path: "orchestrator/configs/product_fidelity_rubric.json",
    rubric_citations: [
      {
        term: "demo_usable",
        criterion: "QA report includes journey-based evidence.",
        evidence: "Evidence is incomplete.",
        pass: false,
      },
    ],
    verified_artifacts: ["A1"],
  }, null, 2), "utf8");

  const result = validateQaVerifierArtifacts({ workspaceRoot, artifactRoot });
  assert.equal(result.ok, true);
});
