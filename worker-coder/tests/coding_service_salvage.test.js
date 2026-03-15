/**
 * Unit tests for salvageWorkflowArtifactFailure (coding_service.js)
 *
 * Tests the step-level salvage logic that fires when a task times out or
 * produces incomplete artifacts. This is the most complex private path in
 * the worker and previously had zero direct coverage.
 *
 * Approach: real filesystem (temp dir) + real orchestrator validators
 * (same pattern as artifact_scaffold.test.js and step_artifact_contract.test.js)
 */
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { salvageWorkflowArtifactFailure } from "../coding_service.js";

function makeTmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "salvage-test-"));
}

function writeJson(dir, rel, obj) {
  const abs = path.join(dir, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, JSON.stringify(obj, null, 2), "utf8");
}

function writeText(dir, rel, text) {
  const abs = path.join(dir, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, text, "utf8");
}

// Builds a minimal valid pm_spec artifact tree
function scaffoldPmSpec(workspaceRoot, artifactRoot) {
  const root = path.join(workspaceRoot, artifactRoot);
  writeText(root, "plan/spec.md", "# Scope\n## User Stories\n## Acceptance Criteria\n## Non-Goals\n## Artifact List\n");
  writeText(root, "plan/milestones.md", "# Milestones\n- M1\n");
  writeText(root, "plan/arch.md", "# Module Breakdown\n## Interfaces\n## Dependency Choices\n## Risk Notes\n");
  writeJson(root, "plan/acceptance.json", { criteria: ["c1"], artifacts: ["plan/spec.md"], owner: "pm", version: "v1" });
  writeJson(root, "metrics/context_budget_pm_spec.json", { status: "ok" });
  writeJson(root, "handoff/pm_to_architect.json", {
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "CRM feature",
    artifacts: ["plan/spec.md"],
    acceptance: { criteria: ["c1"] },
  });
}

// Builds a minimal valid arch_design artifact tree (includes pm_spec artifacts above)
function scaffoldArchDesign(workspaceRoot, artifactRoot) {
  scaffoldPmSpec(workspaceRoot, artifactRoot);
  const root = path.join(workspaceRoot, artifactRoot);
  writeText(root, "plan/interfaces.md", "# Interfaces\n## GET /api/customers\n");
  writeText(root, "plan/workplan.md", "# Workplan\n- BE: server.js\n");
  // risk_report: risks = [{level,title,mitigation}], decision_log = string[]
  writeJson(root, "risk/risk_report.json", {
    risks: [{ level: "low", title: "scope creep", mitigation: "strict contract" }],
    decision_log: ["Use REST for API endpoints"],
  });
  // architect_to_impl: modules/interfaces/risks = string[], decisions = [{adr_id,title,status}]
  writeJson(root, "handoff/architect_to_impl.json", {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe"],
    modules: ["server"],
    interfaces: ["GET /api/customers"],
    decisions: [{ adr_id: "ADR-001", title: "Use REST", status: "accepted" }],
    risks: ["auth migration"],
  });
}

function main() {
  // ── Gate: unknown step returns null ──────────────────────────────────────
  {
    const workspaceRoot = makeTmp();
    const result = salvageWorkflowArtifactFailure({
      workspaceRoot,
      artifactRoot: "artifacts/release/run-1",
      expectedArtifacts: [],
      stepId: "qa_verify",          // not in timeoutOnlySteps or implementationSteps
      taskPrompt: "test",
      result: { diagnostics: { timeout: true } },
    });
    assert.equal(result, null, "unknown step must return null");
  }

  // ── Gate: timeout-only step without timeout flag returns null ─────────────
  {
    const workspaceRoot = makeTmp();
    const result = salvageWorkflowArtifactFailure({
      workspaceRoot,
      artifactRoot: "artifacts/release/run-1",
      expectedArtifacts: [],
      stepId: "pm_spec",
      taskPrompt: "test",
      result: { diagnostics: { timeout: false } },  // no timeout
    });
    assert.equal(result, null, "pm_spec without timeout must return null");
  }

  // ── impl step without timeout takes the artifact_contract salvage path ────
  // With an empty dir: scaffold creates stub files, validators pass on stubs.
  // Result is non-null with salvageReason="artifact_contract".
  {
    const workspaceRoot = makeTmp();
    const result = salvageWorkflowArtifactFailure({
      workspaceRoot,
      artifactRoot: "artifacts/release/run-1",
      expectedArtifacts: [],
      stepId: "impl_be",
      taskPrompt: "test",
      result: {},  // no timeout needed for impl steps
    });
    assert.ok(result !== null, "impl_be without timeout takes artifact_contract path");
    assert.equal(result.salvageReason, "artifact_contract");
  }

  // ── Happy path: pm_spec with timeout + valid artifacts ────────────────────
  {
    const workspaceRoot = makeTmp();
    const artifactRoot = "artifacts/release/run-pm";
    scaffoldPmSpec(workspaceRoot, artifactRoot);

    const result = salvageWorkflowArtifactFailure({
      workspaceRoot,
      artifactRoot,
      expectedArtifacts: ["plan/spec.md", "plan/acceptance.json"],
      stepId: "pm_spec",
      taskPrompt: "write a pm spec",
      result: { diagnostics: { timeout: true } },
    });
    assert.ok(result !== null, "pm_spec with timeout + valid artifacts must salvage");
    assert.equal(result.salvageReason, "timeout");
    assert.ok(result.roleValidation?.ok, "role validation must pass");
    assert.ok(result.handoffValidation?.ok, "handoff validation must pass");
  }

  // ── Happy path: arch_design with timeout + valid artifacts ────────────────
  {
    const workspaceRoot = makeTmp();
    const artifactRoot = "artifacts/release/run-arch";
    scaffoldArchDesign(workspaceRoot, artifactRoot);

    const result = salvageWorkflowArtifactFailure({
      workspaceRoot,
      artifactRoot,
      expectedArtifacts: ["plan/arch.md", "risk/risk_report.json"],
      stepId: "arch_design",
      taskPrompt: "write an architecture spec",
      result: { diagnostics: { timeout: true } },
    });
    assert.ok(result !== null, "arch_design with timeout + valid artifacts must salvage");
    assert.equal(result.salvageReason, "timeout");
    assert.ok(result.roleValidation?.ok, "role validation must pass");
  }

  // ── salvageReason is 'artifact_contract' for impl steps ──────────────────
  // (impl_be can be salvaged via artifact_contract path even without timeout)
  // We skip this happy-path since building a valid impl_be tree is expensive,
  // and the gate/null paths above already cover the branching logic.
  // Integration-level coverage comes from coding_team_e2e canary.

  console.log("coding_service_salvage.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("coding_service_salvage.test.js: failed");
  console.error(err);
  process.exit(1);
}
