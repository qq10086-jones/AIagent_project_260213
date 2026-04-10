import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { validateArtifactPack } from "../src/artifact_pack_validator.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-artifact-pack-"));
}

function writeJson(targetPath, value) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, JSON.stringify(value, null, 2));
}

test("artifact pack passes with verify/qa_report.json covering acceptance ids", () => {
  const workspaceRoot = makeWorkspace();
  const releaseRoot = path.join(workspaceRoot, "artifacts", "release", "run-pass");
  fs.mkdirSync(releaseRoot, { recursive: true });

  writeJson(path.join(releaseRoot, "plan", "acceptance.json"), {
    criteria: [{ id: "A1" }, { id: "A2" }],
  });
  writeJson(path.join(releaseRoot, "risk", "risk_report.json"), {
    risks: [{ level: "low", title: "ok", mitigation: "ok" }],
    decision_log: ["keep scope narrow"],
  });
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass_with_warnings",
    checks: [
      {
        check_id: "qa-1",
        layer: "deterministic",
        description: "contract files exist",
        status: "pass",
        detail: "required artifacts found",
      },
    ],
    verified_artifacts: ["A1", "A2"],
    journey_checks: [
      { journey_id: "J1", description: "basic flow", status: "pass", evidence: ["tested manually"] },
    ],
    rubric_citations: [
      { term: "correctness", criterion: "outputs match spec", evidence: "verified", pass: true },
    ],
  });
  writeJson(path.join(releaseRoot, "metrics", "context_budget_qa_verify.json"), {
    step_id: "qa_verify",
    artifact_count: 1,
    bytes_total: 100,
    largest_artifact_bytes: 100,
    prompt_chars: 50,
    injected_context_bytes: 0,
    status: "ok",
    threshold_source: "default_thresholds",
  });
  writeJson(path.join(releaseRoot, "verify", "acceptance_result.json"), {
    criteria_results: [],
    deterministic_pass: 0,
    deterministic_fail: 0,
    semantic_pending: 0,
    verdict: "pass",
    generated_at: new Date().toISOString(),
  });

  const metaRoot = path.join(releaseRoot, "meta");
  fs.mkdirSync(metaRoot, { recursive: true });
  const manifestPath = path.join(metaRoot, "run_manifest.json");
  const summaryPath = path.join(metaRoot, "run_summary.md");
  writeJson(manifestPath, {
    workflow_run_id: "wf-1",
    run_id: "run-1",
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    status: "succeeded",
    steps: [{ id: "qa_verify", step_id: "qa_verify", status: "succeeded" }],
    checkpoints: [{ id: "qa_verify" }],
    step_artifacts: [[]],
    context_budget_reports: [
      {
        step_id: "qa_verify",
        report_path: "metrics/context_budget_qa_verify.json",
        status: "ok",
        threshold_source: "default_thresholds",
      },
    ],
    context_budget_summary: {
      total_steps: 1,
      ok: 1,
      warning: 0,
      overflow_risk: 0,
      missing: 0,
    },
    artifact_coverage: {},
  });
  fs.writeFileSync(summaryPath, "# Summary\n");

  const result = validateArtifactPack({
    run: { workflow_run_id: "wf-1", run_id: "run-1", workflow_id: "coding_team_v0", project_type: "webapp_crm" },
    steps: [{ id: "qa_verify", step_id: "qa_verify", status: "succeeded" }],
    checkpoints: [{ id: "qa_verify" }],
    manifestPath,
    summaryPath,
    registry: { project_types: { webapp_crm: { required_artifacts: [] } } },
  });

  assert.equal(result.ok, true);
  assert.deepEqual(result.reasons, []);
});

test("artifact pack fails when qa_report misses acceptance coverage", () => {
  const workspaceRoot = makeWorkspace();
  const releaseRoot = path.join(workspaceRoot, "artifacts", "release", "run-fail");
  fs.mkdirSync(releaseRoot, { recursive: true });

  writeJson(path.join(releaseRoot, "plan", "acceptance.json"), {
    criteria: [{ id: "A1" }, { id: "A2" }],
  });
  writeJson(path.join(releaseRoot, "risk", "risk_report.json"), {
    risks: [{ level: "low", title: "ok", mitigation: "ok" }],
    decision_log: ["keep scope narrow"],
  });
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [
      {
        check_id: "qa-1",
        layer: "deterministic",
        description: "contract files exist",
        status: "pass",
        detail: "required artifacts found",
      },
    ],
    verified_artifacts: ["A1"],
  });
  writeJson(path.join(releaseRoot, "metrics", "context_budget_qa_verify.json"), {
    step_id: "qa_verify",
    artifact_count: 1,
    bytes_total: 100,
    largest_artifact_bytes: 100,
    prompt_chars: 50,
    injected_context_bytes: 0,
    status: "ok",
    threshold_source: "default_thresholds",
  });
  writeJson(path.join(releaseRoot, "verify", "acceptance_result.json"), {
    criteria_results: [],
    deterministic_pass: 0,
    deterministic_fail: 0,
    semantic_pending: 0,
    verdict: "pass",
    generated_at: new Date().toISOString(),
  });

  const metaRoot = path.join(releaseRoot, "meta");
  fs.mkdirSync(metaRoot, { recursive: true });
  const manifestPath = path.join(metaRoot, "run_manifest.json");
  const summaryPath = path.join(metaRoot, "run_summary.md");
  writeJson(manifestPath, {
    workflow_run_id: "wf-2",
    run_id: "run-2",
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    status: "succeeded",
    steps: [{ id: "qa_verify", step_id: "qa_verify", status: "succeeded" }],
    checkpoints: [{ id: "qa_verify" }],
    step_artifacts: [[]],
    context_budget_reports: [
      {
        step_id: "qa_verify",
        report_path: "metrics/context_budget_qa_verify.json",
        status: "ok",
        threshold_source: "default_thresholds",
      },
    ],
    context_budget_summary: {
      total_steps: 1,
      ok: 1,
      warning: 0,
      overflow_risk: 0,
      missing: 0,
    },
    artifact_coverage: {},
  });
  fs.writeFileSync(summaryPath, "# Summary\n");

  const result = validateArtifactPack({
    run: { workflow_run_id: "wf-2", run_id: "run-2", workflow_id: "coding_team_v0", project_type: "webapp_crm" },
    steps: [{ id: "qa_verify", step_id: "qa_verify", status: "succeeded" }],
    checkpoints: [{ id: "qa_verify" }],
    manifestPath,
    summaryPath,
    registry: { project_types: { webapp_crm: { required_artifacts: [] } } },
  });

  assert.equal(result.ok, false);
  assert.match(result.reasons.join("\n"), /verify\/qa_report\.json:missing acceptance ids A2/);
});

test("artifact pack fails when context budget report is missing", () => {
  const workspaceRoot = makeWorkspace();
  const releaseRoot = path.join(workspaceRoot, "artifacts", "release", "run-budget-miss");
  fs.mkdirSync(releaseRoot, { recursive: true });

  writeJson(path.join(releaseRoot, "plan", "acceptance.json"), {
    criteria: [{ id: "A1" }],
  });
  writeJson(path.join(releaseRoot, "risk", "risk_report.json"), {
    risks: [{ level: "low", title: "ok", mitigation: "ok" }],
    decision_log: ["keep scope narrow"],
  });
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [
      {
        check_id: "qa-1",
        layer: "deterministic",
        description: "contract files exist",
        status: "pass",
        detail: "required artifacts found",
      },
    ],
    verified_artifacts: ["A1"],
  });
  writeJson(path.join(releaseRoot, "verify", "acceptance_result.json"), {
    criteria_results: [],
    deterministic_pass: 0,
    deterministic_fail: 0,
    semantic_pending: 0,
    verdict: "pass",
    generated_at: new Date().toISOString(),
  });

  const metaRoot = path.join(releaseRoot, "meta");
  fs.mkdirSync(metaRoot, { recursive: true });
  const manifestPath = path.join(metaRoot, "run_manifest.json");
  const summaryPath = path.join(metaRoot, "run_summary.md");
  writeJson(manifestPath, {
    workflow_run_id: "wf-3",
    run_id: "run-3",
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    status: "succeeded",
    steps: [{ id: "qa_verify", step_id: "qa_verify", status: "succeeded" }],
    checkpoints: [{ id: "qa_verify" }],
    step_artifacts: [[]],
    context_budget_reports: [],
    context_budget_summary: {
      total_steps: 0,
      ok: 0,
      warning: 0,
      overflow_risk: 0,
      missing: 1,
    },
    artifact_coverage: {},
  });
  fs.writeFileSync(summaryPath, "# Summary\n");

  const result = validateArtifactPack({
    run: { workflow_run_id: "wf-3", run_id: "run-3", workflow_id: "coding_team_v0", project_type: "webapp_crm" },
    steps: [{ id: "qa_verify", step_id: "qa_verify", status: "succeeded" }],
    checkpoints: [{ id: "qa_verify" }],
    manifestPath,
    summaryPath,
    registry: { project_types: { webapp_crm: { required_artifacts: [] } } },
  });

  assert.equal(result.ok, false);
  assert.match(result.reasons.join("\n"), /ARTIFACT_MISSING:metrics\/context_budget_qa_verify\.json/);
});
