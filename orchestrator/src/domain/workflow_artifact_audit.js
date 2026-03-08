/**
 * workflow_artifact_audit.js
 *
 * Artifact classification, failure-payload construction, and workspace-hash
 * generation utilities.
 * Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import crypto from "crypto";
import fs from "fs";
import path from "path";

export function classifyArtifactReasons(reasons = []) {
  const missing = [];
  const invalid = [];
  for (const item of reasons || []) {
    const text = String(item || "");
    if (text.startsWith("ARTIFACT_MISSING:")) missing.push(text.replace(/^ARTIFACT_MISSING:/, ""));
    if (text.startsWith("ARTIFACT_INVALID:")) invalid.push(text.replace(/^ARTIFACT_INVALID:/, ""));
  }
  return { missing, invalid };
}

export function buildFailurePayload({
  errorCode = "WORKFLOW_FAILED",
  failedStep = "",
  missing = [],
  invalid = [],
  detail = "",
}) {
  const miss = Array.isArray(missing) ? missing : [];
  const inv = Array.isArray(invalid) ? invalid : [];
  const suggested = [];
  if (miss.length > 0) suggested.push("write all required artifacts to canonical paths");
  if (inv.length > 0) suggested.push("fix artifact schema/content quality issues");
  if (String(errorCode || "") === "STEP_ARTIFACT_MISSING") {
    suggested.push("re-run the failed step with deterministic templates enabled");
  }
  if (suggested.length === 0) suggested.push("inspect logs and re-run with strict diagnostics");
  return {
    error_code: String(errorCode || "WORKFLOW_FAILED"),
    failed_step: String(failedStep || ""),
    missing: miss,
    invalid: inv,
    suggested_fix: suggested,
    detail: String(detail || ""),
  };
}

export function buildWorkspaceHash({ workflow_run_id, step_index, task_id, status, artifacts }) {
  const raw = JSON.stringify({
    workflow_run_id,
    step_index,
    task_id,
    status,
    artifacts: Array.isArray(artifacts) ? artifacts : [],
  });
  return crypto.createHash("sha256").update(raw).digest("hex");
}

export function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath, { recursive: true });
}

export function writeJsonFile(targetPath, obj) {
  ensureDir(path.dirname(targetPath));
  fs.writeFileSync(targetPath, JSON.stringify(obj, null, 2), "utf8");
}

export function buildReleasePackPaths(workspaceRoot, run) {
  const releaseRoot = path.join(workspaceRoot, "artifacts", "release", String(run.run_id || run.workflow_run_id));
  return {
    release_root: releaseRoot,
    manifest_path: path.join(releaseRoot, "meta", "run_manifest.json"),
    summary_path: path.join(releaseRoot, "summary", "run_summary.md"),
    strict_canary_json_path: path.join(releaseRoot, "qa", "strict_canary_report.json"),
    strict_canary_report_path: path.join(releaseRoot, "qa", "strict_canary_report.md"),
    go_no_go_result_path: path.join(releaseRoot, "qa", "go_no_go_result.json"),
  };
}

export function buildStepArtifactsFromCheckpoints(steps = [], checkpoints = [], parseJsonSafeFn = null) {
  const parseJson = typeof parseJsonSafeFn === "function" ? parseJsonSafeFn : (value, fallback = []) => fallback;
  const cpArtifactByStep = new Map();
  for (const cp of checkpoints || []) {
    const key = `${Number(cp.step_index)}:${String(cp.step_id || "")}`;
    cpArtifactByStep.set(key, parseJson(cp.artifact_refs_json, []));
  }
  return (steps || []).map((step) => {
    const key = `${Number(step.step_index)}:${String(step.step_id || "")}`;
    return {
      step_index: Number(step.step_index),
      step_id: step.step_id,
      task_id: step.task_id || "",
      artifacts: cpArtifactByStep.get(key) || [],
    };
  });
}

export function inferProjectArtifactCoverage(steps) {
  const byId = Object.fromEntries((steps || []).map((s) => [String(s.step_id || ""), s]));
  return {
    spec: !!byId.pm_spec,
    arch: !!byId.arch_design,
    diff: !!byId.impl_fe || !!byId.impl_be,
    verification: !!byId.qa_verify,
    run_summary: !!byId.release_pack,
    run_manifest: true,
  };
}

export function parseStepResult(step = {}, parseJsonSafe = JSON.parse) {
  if (step && typeof step.result_json === "object" && step.result_json !== null) return step.result_json;
  try {
    return parseJsonSafe(step?.result_json || "{}");
  } catch {
    return {};
  }
}

export function buildStrictCanaryReport({ run, steps = [], stepContracts = {}, parseJson = JSON.parse }) {
  const checks = [];
  for (const s of steps || []) {
    const stepId = String(s?.step_id || "");
    const requiredArtifacts = Array.isArray(stepContracts?.[stepId]?.required_artifacts)
      ? stepContracts[stepId].required_artifacts
      : [];
    const result = parseStepResult(s, parseJson);
    const artifactCheck = result?.artifact_check && typeof result.artifact_check === "object"
      ? result.artifact_check
      : { checked: false, missing: [], found: [] };
    const missing = Array.isArray(artifactCheck.missing) ? artifactCheck.missing : [];
    const found = Array.isArray(artifactCheck.found) ? artifactCheck.found : [];
    const requiresAudit = requiredArtifacts.length > 0;
    let pass = String(s?.status || "") === "succeeded";
    let reason = "";
    if (requiresAudit && artifactCheck.checked !== true) {
      pass = false;
      reason = "artifact_check missing";
    } else if (requiresAudit && missing.length > 0) {
      pass = false;
      reason = `missing artifacts: ${missing.join(", ")}`;
    }
    checks.push({
      step_index: Number(s?.step_index),
      step_id: stepId,
      status: String(s?.status || ""),
      requires_artifact_audit: requiresAudit,
      artifact_check: {
        checked: Boolean(artifactCheck.checked),
        missing,
        found,
      },
      pass,
      reason,
    });
  }
  const failed = checks.filter((x) => !x.pass);
  const missingTotal = checks.reduce((acc, x) => acc + (x.artifact_check.missing?.length || 0), 0);
  return {
    workflow_run_id: run.workflow_run_id,
    run_id: run.run_id,
    workflow_id: run.workflow_id,
    generated_at: new Date().toISOString(),
    strict_mode_expected: true,
    verdict: failed.length === 0 ? "pass" : "fail",
    totals: {
      steps: checks.length,
      failed_steps: failed.length,
      missing_artifacts_total: missingTotal,
    },
    checks,
  };
}

export function buildStrictCanaryMarkdown(report, jsonRelPath = "") {
  const lines = [
    "# Strict Canary Report",
    "",
    `- workflow_run_id: ${report.workflow_run_id}`,
    `- run_id: ${report.run_id}`,
    `- workflow_id: ${report.workflow_id}`,
    `- generated_at: ${report.generated_at}`,
    `- verdict: ${String(report.verdict || "").toUpperCase()}`,
    `- total_steps: ${Number(report?.totals?.steps || 0)}`,
    `- failed_steps: ${Number(report?.totals?.failed_steps || 0)}`,
    `- missing_artifacts_total: ${Number(report?.totals?.missing_artifacts_total || 0)}`,
  ];
  if (jsonRelPath) lines.push(`- report_json: ${jsonRelPath}`);
  lines.push("", "## Step Checks");
  for (const item of report.checks || []) {
    const missing = Array.isArray(item?.artifact_check?.missing) ? item.artifact_check.missing : [];
    const found = Array.isArray(item?.artifact_check?.found) ? item.artifact_check.found : [];
    lines.push(
      `- [${item.pass ? "PASS" : "FAIL"}] ${item.step_index}:${item.step_id} status=${item.status} checked=${Boolean(item?.artifact_check?.checked)} missing=${missing.length} found=${found.length}${item.reason ? ` reason=${item.reason}` : ""}`
    );
  }
  return lines.join("\n");
}

export function buildGoNoGoResult({
  run,
  manifest,
  steps = [],
  validator,
  canaryReport = null,
  expectedSteps = 0,
  strict = true,
}) {
  const safeSteps = Array.isArray(steps) ? steps : [];
  const acceptanceStep =
    safeSteps.find((s) => String(s.gate_name || "") === "acceptance") ||
    safeSteps.find((s) => String(s.step_id || "") === "qa_verify") ||
    null;
  const checks = [];
  checks.push({
    name: "artifact_pack_validator",
    pass: Boolean(validator?.ok),
    detail: validator?.ok
      ? "validator passed"
      : `validator failed: ${Array.isArray(validator?.reasons) ? validator.reasons.join("; ") : "unknown"}`,
  });
  checks.push({
    name: "workflow_status",
    pass: String(manifest?.status || "") === "succeeded",
    detail: `manifest.status=${String(manifest?.status || "")}`,
  });
  checks.push({
    name: "step_success",
    pass: safeSteps.length > 0 && safeSteps.every((s) => String(s.status || "") === "succeeded"),
    detail: `succeeded=${safeSteps.filter((s) => String(s.status || "") === "succeeded").length}/${safeSteps.length}`,
  });
  if (expectedSteps > 0) {
    checks.push({
      name: "step_count",
      pass: safeSteps.length === expectedSteps,
      detail: `steps=${safeSteps.length} expected=${expectedSteps}`,
    });
  }
  checks.push({
    name: "acceptance_gate",
    pass: !!acceptanceStep && String(acceptanceStep.status || "") === "succeeded",
    detail: acceptanceStep
      ? `step=${String(acceptanceStep.step_id || "")} status=${String(acceptanceStep.status || "")}`
      : "acceptance step missing",
  });
  checks.push({
    name: "strict_canary_verdict",
    pass: !!canaryReport && String(canaryReport.verdict || "") === "pass",
    detail: canaryReport ? `verdict=${String(canaryReport.verdict || "")}` : "strict canary report missing",
  });
  checks.push({
    name: "strict_canary_missing_artifacts",
    pass: !!canaryReport && Number(canaryReport?.totals?.missing_artifacts_total || 0) === 0,
    detail: canaryReport
      ? `missing_artifacts_total=${Number(canaryReport?.totals?.missing_artifacts_total || 0)}`
      : "strict canary report missing",
  });
  const reasons = [];
  for (const c of checks) {
    if (!c.pass) reasons.push(`${c.name}: ${c.detail}`);
  }
  if (!strict) {
    const onlyStepCountFail = reasons.length > 0 && reasons.every((r) => r.startsWith("step_count:"));
    if (onlyStepCountFail) reasons.length = 0;
  }
  return {
    verdict: reasons.length === 0 ? "GO" : "NO_GO",
    workflow_run_id: run.workflow_run_id,
    run_id: run.run_id,
    workflow_id: run.workflow_id,
    project_type: run.project_type,
    strict,
    generated_at: new Date().toISOString(),
    total_checks: checks.length,
    passed_checks: checks.filter((c) => c.pass).length,
    failed_checks: checks.filter((c) => !c.pass).length,
    checks,
    reasons,
  };
}
