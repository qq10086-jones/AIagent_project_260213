import fs from "fs";
import os from "os";
import path from "path";

import { createContextBudgetService } from "../src/domain/context_budget_service.js";
import { validateArtifactPack } from "../src/artifact_pack_validator.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-canary-budget-"));
}

function writeJson(targetPath, value) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, JSON.stringify(value, null, 2), "utf8");
}

function writeText(targetPath, value) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, value, "utf8");
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function main() {
  const workspaceRoot = makeWorkspace();
  const service = createContextBudgetService();
  const checks = [];

  const artifactPath = path.join(workspaceRoot, "tmp", "small.txt");
  writeText(artifactPath, "small artifact");
  const okReport = service.buildReport({
    stepId: "impl_be",
    role: "backend",
    prompt: "small prompt",
    injectedContext: "small context",
    artifactPaths: [artifactPath],
    runId: "run-ok",
    workflowRunId: "wf-ok",
  });
  assert(okReport.status === "ok", "normal-size run did not classify as ok");
  checks.push({ id: "normal_size_ok", ok: true, status: okReport.status });

  const warningReport = service.buildReport({
    stepId: "arch_design",
    role: "architect",
    prompt: "a".repeat(110000),
    injectedContext: "",
    artifactBytes: [100],
    runId: "run-warning",
    workflowRunId: "wf-warning",
  });
  assert(warningReport.status === "warning", "oversized prompt did not classify as warning");
  checks.push({ id: "oversized_warning", ok: true, status: warningReport.status });

  const overflowReport = service.buildReport({
    stepId: "impl_fe",
    role: "frontend",
    prompt: "small",
    injectedContext: "b".repeat(360000),
    artifactBytes: [100],
    runId: "run-overflow",
    workflowRunId: "wf-overflow",
  });
  assert(overflowReport.status === "overflow_risk", "oversized injected context did not classify as overflow_risk");
  checks.push({ id: "oversized_overflow_risk", ok: true, status: overflowReport.status });

  const releaseRoot = path.join(workspaceRoot, "artifacts", "release", "run-pack");
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
  writeJson(path.join(releaseRoot, "metrics", "context_budget_qa_verify.json"), {
    step_id: "qa_verify",
    artifact_count: 1,
    bytes_total: 512,
    largest_artifact_bytes: 512,
    prompt_chars: 300,
    injected_context_bytes: 0,
    status: "ok",
    threshold_source: "default_thresholds",
  });
  const manifestPath = path.join(releaseRoot, "meta", "run_manifest.json");
  const summaryPath = path.join(releaseRoot, "summary", "run_summary.md");
  writeJson(manifestPath, {
    workflow_run_id: "wf-pack",
    run_id: "run-pack",
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
  writeText(summaryPath, "# Run Summary\n\n## Context Budget\n- total_steps: 1\n- ok: 1\n");
  const packValidation = validateArtifactPack({
    run: {
      workflow_run_id: "wf-pack",
      run_id: "run-pack",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
    },
    steps: [{ id: "qa_verify", step_id: "qa_verify", status: "succeeded" }],
    checkpoints: [{ id: "qa_verify" }],
    manifestPath,
    summaryPath,
    registry: { project_types: { webapp_crm: { required_artifacts: [] } } },
  });
  assert(packValidation.ok === true, "release pack context budget aggregation validation failed");
  checks.push({
    id: "release_pack_budget_metadata",
    ok: true,
    context_budget_reports: 1,
    summary_total_steps: 1,
  });

  const overridePolicyPath = path.join(workspaceRoot, "override_policy.json");
  writeJson(overridePolicyPath, {
    version: "1.0.0",
    default_thresholds: {
      warning_prompt_chars: 80000,
      overflow_risk_prompt_chars: 120000,
      warning_artifact_bytes: 500000,
      overflow_risk_artifact_bytes: 1000000,
      warning_injected_context_bytes: 150000,
      overflow_risk_injected_context_bytes: 300000,
    },
    role_overrides: {
      backend: {
        warning_prompt_chars: 5,
        overflow_risk_prompt_chars: 10,
      },
    },
  });
  const overrideService = createContextBudgetService({ policyPath: overridePolicyPath });
  const defaultClassified = service.buildReport({
    stepId: "impl_be",
    role: "backend",
    prompt: "1234567",
    injectedContext: "",
    artifactBytes: [10],
  });
  const overrideClassified = overrideService.buildReport({
    stepId: "impl_be",
    role: "backend",
    prompt: "1234567",
    injectedContext: "",
    artifactBytes: [10],
  });
  assert(defaultClassified.status === "ok", "default policy baseline changed unexpectedly");
  assert(overrideClassified.status === "warning", "policy override did not change classification result");
  checks.push({
    id: "policy_override_changes_classification",
    ok: true,
    baseline_status: defaultClassified.status,
    override_status: overrideClassified.status,
  });

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "context_budget");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "context_budget_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    workspace_root: workspaceRoot.replace(/\\/g, "/"),
    checks,
  }, null, 2), "utf8");

  console.log("# Context Budget Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
