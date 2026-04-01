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
import { fileURLToPath } from "url";

const _MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const _DOMAIN_PACKS_DIR = path.resolve(_MODULE_DIR, "../../configs/domain_acceptance_packs");

const _DOMAIN_PACK_MAP = {
  webapp_crm: "crm",
  crm: "crm",
  crm_app: "crm",
  ecommerce: "ecommerce",
  webapp_ecommerce: "ecommerce",
  ecommerce_demo: "ecommerce",
  document_release: "document_release",
  document: "document_release",
  release_management: "document_release",
};

function loadDomainAcceptancePack(projectType = "") {
  const packKey = _DOMAIN_PACK_MAP[String(projectType || "").trim().toLowerCase()];
  if (!packKey) return null;
  try {
    const packPath = path.join(_DOMAIN_PACKS_DIR, `${packKey}.json`);
    return JSON.parse(fs.readFileSync(packPath, "utf8"));
  } catch {
    return null;
  }
}

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
  const releaseRoot = path.join(workspaceRoot, "runtime", "artifacts", "release", String(run.run_id || run.workflow_run_id));
  return {
    release_root: releaseRoot,
    manifest_path: path.join(releaseRoot, "meta", "run_manifest.json"),
    summary_path: path.join(releaseRoot, "summary", "run_summary.md"),
    strict_canary_json_path: path.join(releaseRoot, "qa", "strict_canary_report.json"),
    strict_canary_report_path: path.join(releaseRoot, "qa", "strict_canary_report.md"),
    go_no_go_result_path: path.join(releaseRoot, "qa", "go_no_go_result.json"),
    product_fidelity_report_path: path.join(releaseRoot, "qa", "product_fidelity_report.json"),
    preview_validation_report_path: path.join(releaseRoot, "qa", "preview_validation_report.json"),
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

export function getExpectedWorkflowStepCount(registry = {}, workflowId = "") {
  const safeWorkflowId = String(workflowId || "").trim();
  if (!safeWorkflowId) return 0;
  const workflowDef = registry?.workflows?.[safeWorkflowId];
  const steps = Array.isArray(workflowDef?.steps) ? workflowDef.steps : [];
  return steps.length > 0 ? steps.length : 0;
}

function safeReadText(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return "";
  }
}

function safeReadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function countNonEmptyLines(text = "") {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean).length;
}

function findPatternMatches(text = "", patterns = []) {
  const out = [];
  for (const pattern of patterns) {
    if (pattern.test(text)) out.push(pattern.source);
  }
  return out;
}

function hasVisibleUiSignals(text = "") {
  return /<button|<form|<input|onClick|submit|render|return\s*\(|createElement|fetch\(|router|route/i.test(String(text || ""));
}

function detectPreviewMismatch({ projectType = "", previewRoot = "" }) {
  const normalizedType = String(projectType || "").trim().toLowerCase();
  const normalizedRoot = String(previewRoot || "").replace(/\\/g, "/").trim().toLowerCase();
  if (!normalizedRoot) {
    return {
      mismatch: false,
      classification: "preview_missing",
      source: "missing",
    };
  }
  if (/\/sandbox\/crm_site\/?$/.test(normalizedRoot)) {
    const crmLike = ["webapp_crm", "crm", "crm_app"].includes(normalizedType);
    return {
      mismatch: !crmLike,
      classification: crmLike ? "preview_matched" : "preview_mismatch",
      source: "shared_crm_sandbox",
    };
  }
  return {
    mismatch: false,
    classification: "preview_matched",
    source: "run_scoped_or_custom",
  };
}

export function buildPreviewValidationReport({ run, releaseRoot }) {
  const previewPath = path.join(releaseRoot, "preview", "preview_runtime.json");
  const deploymentResultPath = path.join(releaseRoot, "preview", "deployment_result.json");
  const previewRuntime = safeReadJson(previewPath) || {};
  const deploymentResult = safeReadJson(deploymentResultPath) || {};
  const previewRoot = String(previewRuntime?.project_root || "").replace(/\\/g, "/");
  const previewUrl = String(
    previewRuntime?.preview_url ||
    previewRuntime?.url ||
    deploymentResult?.preview_url ||
    deploymentResult?.deployment?.preview_url ||
    ""
  ).trim();
  const previewMode = String(previewRuntime?.mode || "").trim() || null;
  const commandSource = String(previewRuntime?.command_source || "").trim() || null;
  const detection = detectPreviewMismatch({
    projectType: run?.project_type,
    previewRoot,
  });
  const domainPack = loadDomainAcceptancePack(run?.project_type);
  const checks = [
    {
      criterion: "preview_runtime_present",
      evidence: safeReadJson(previewPath) ? "preview/preview_runtime.json loaded" : "preview/preview_runtime.json missing or invalid",
      pass: Boolean(safeReadJson(previewPath)),
    },
    {
      criterion: "preview_project_root_present",
      evidence: previewRoot ? `preview.project_root=${previewRoot}` : "preview.project_root missing",
      pass: Boolean(previewRoot),
    },
    {
      criterion: "preview_matched",
      evidence: `project_type=${String(run?.project_type || "")} source=${detection.source}`,
      pass: !detection.mismatch,
    },
  ];

  if (previewMode) {
    checks.push({
      criterion: "preview_mode_recorded",
      evidence: `preview.mode=${previewMode}${commandSource ? ` command_source=${commandSource}` : ""}`,
      pass: true,
    });
  }

  if (domainPack && previewRoot && detection.source === "shared_crm_sandbox") {
    const packPattern = String(domainPack.preview_expectations?.preview_root_pattern || "");
    if (packPattern) {
      const patternMatch = new RegExp(packPattern, "i").test(previewRoot);
      checks.push({
        criterion: "preview_root_domain_pattern",
        evidence: patternMatch
          ? `preview.project_root matches domain pattern '${packPattern}' (pack=${domainPack.domain})`
          : `preview.project_root '${previewRoot}' does not match domain pattern '${packPattern}' (pack=${domainPack.domain})`,
        pass: patternMatch,
      });
    }
    const mustNotBeUnrelated = Boolean(domainPack.preview_expectations?.must_not_be_shared_crm_sandbox);
    if (mustNotBeUnrelated && detection.source === "shared_crm_sandbox" && !detection.mismatch === false) {
      checks.push({
        criterion: "preview_not_shared_sandbox",
        evidence: `preview points to shared crm_site sandbox but project_type=${String(run?.project_type || "")} is not CRM (pack=${domainPack.domain})`,
        pass: false,
      });
    }
  }

  const hasFailedDomainCheck = checks.some((c) => c.criterion === "preview_root_domain_pattern" && !c.pass);
  const PREVIEW_OPTIONAL_TYPES = ["generic_app", "single_file_html", "generic_coding_task"];
  const previewIsOptional = PREVIEW_OPTIONAL_TYPES.includes(String(run?.project_type || "").toLowerCase());
  // For generic project types, absence of a preview is expected — do not warn
  const shouldWarn = detection.mismatch || (!previewRoot && !previewIsOptional) || hasFailedDomainCheck;
  const finalClassification = detection.mismatch
    ? detection.classification
    : (!previewRoot
      ? "preview_missing"
      : (hasFailedDomainCheck ? "preview_domain_pattern_mismatch" : detection.classification));

  return {
    workflow_run_id: String(run?.workflow_run_id || ""),
    run_id: String(run?.run_id || ""),
    workflow_id: String(run?.workflow_id || ""),
    project_type: String(run?.project_type || ""),
    preview_url: previewUrl || null,
    project_root: previewRoot || null,
    preview_mode: previewMode,
    command_source: commandSource,
    domain_pack: domainPack ? { domain: domainPack.domain, version: domainPack.version } : null,
    classification: finalClassification,
    preview_source: detection.source,
    should_warn: shouldWarn,
    warning: shouldWarn ? `preview_validation_warning:${finalClassification}` : "",
    checks,
    generated_at: new Date().toISOString(),
  };
}

export function buildProductFidelityReport({ run, releaseRoot }) {
  const rubricPath = "orchestrator/configs/product_fidelity_rubric.json";
  const fePath = path.join(releaseRoot, "impl", "fe_changes", "app.js");
  const fePublicHtmlPath = path.join(releaseRoot, "impl", "fe_changes", "public", "index.html");
  const fePublicJsPath = path.join(releaseRoot, "impl", "fe_changes", "public", "app.js");
  const bePath = path.join(releaseRoot, "impl", "be_changes", "server.js");
  const qaPath = path.join(releaseRoot, "verify", "qa_report.json");
  const previewPath = path.join(releaseRoot, "preview", "preview_runtime.json");
  const domainPack = loadDomainAcceptancePack(run?.project_type);

  const feText = safeReadText(fePath);
  const fePublicHtmlText = safeReadText(fePublicHtmlPath);
  const fePublicJsText = safeReadText(fePublicJsPath);
  const beText = safeReadText(bePath);
  const qaReport = safeReadJson(qaPath) || {};
  const previewRuntime = safeReadJson(previewPath) || {};

  const GENERIC_PROJECT_TYPES = ["generic_app", "single_file_html", "generic_coding_task"];
  const normalizedProjectType = String(run?.project_type || "").toLowerCase();
  const isGenericProject = GENERIC_PROJECT_TYPES.includes(normalizedProjectType);
  const isSingleFileHtml = normalizedProjectType === "single_file_html";

  const placeholderPatterns = [
    /\bstub\b/i,
    // \bplaceholder\b only as a scaffold marker — not as an HTML input attribute (placeholder="...")
    /\bplaceholder\b(?!\s*=)/i,
    /\bscaffold\b/i,
    /\bpending human review\b/i,
    /\bauto-generated\b/i,
    /\bsample item\b/i,
    /\bsample customer\b/i,
    /\blorem ipsum\b/i,
    // Note: \btodo\b intentionally excluded — "todo" is a legitimate domain noun in todo-style apps
  ];
  // For generic projects (no defined domain), item/record are acceptable — only flag "entity" as structural placeholder
  const genericCrudPatterns = isGenericProject
    ? [/\bentity\b/i]
    : [/\bentity\b/i, /\brecord\b/i, /\bitem\b/i];
  const hasPublishedFrontendSurface = Boolean(fePublicHtmlText || fePublicJsText);
  const frontendSurfaceText = (isSingleFileHtml || hasPublishedFrontendSurface)
    ? [fePublicHtmlText, fePublicJsText].filter(Boolean).join("\n")
    : feText;
  const frontendSurfaceLines = countNonEmptyLines(frontendSurfaceText);
  const frontendSurfaceBytes = Buffer.byteLength(frontendSurfaceText || "", "utf8");
  const frontendSurfaceSources = [
    feText ? "impl/fe_changes/app.js" : "",
    fePublicJsText ? "impl/fe_changes/public/app.js" : "",
    fePublicHtmlText ? "impl/fe_changes/public/index.html" : "",
  ].filter(Boolean);
  const combinedText = [frontendSurfaceText, beText].filter(Boolean).join("\n");
  // placeholder_free checks only impl files — QA report scaffold status is handled by qa_evidence_not_scaffold_only
  const placeholderMatches = findPatternMatches(combinedText, placeholderPatterns);
  const genericCrudMatches = findPatternMatches(combinedText, genericCrudPatterns);
  const beLines = countNonEmptyLines(beText);
  const qaChecks = Array.isArray(qaReport?.checks) ? qaReport.checks : [];
  const qaPendingHumanReview = qaChecks.length > 0
    && qaChecks.every((item) => String(item?.status || "") === "warning")
    && qaChecks.every((item) => /pending human review|auto-generated/i.test(String(item?.detail || "")));
  const previewRoot = String(previewRuntime?.project_root || "").replace(/\\/g, "/");
  const previewValidation = detectPreviewMismatch({
    projectType: run?.project_type,
    previewRoot,
  });
  const previewMismatch = previewValidation.mismatch;

  const reasoning = [
    {
      criterion: "frontend_depth",
      evidence: frontendSurfaceText
        ? `frontend_surface non_empty_lines=${frontendSurfaceLines} bytes=${frontendSurfaceBytes} sources=${frontendSurfaceSources.join(", ")}`
        : "frontend surface missing or unreadable",
      pass: Boolean(frontendSurfaceText) && frontendSurfaceLines >= 8 && frontendSurfaceBytes >= 180,
    },
    {
      criterion: "backend_depth",
      evidence: beText
        ? `impl/be_changes/server.js non_empty_lines=${beLines} bytes=${Buffer.byteLength(beText, "utf8")}`
        : "impl/be_changes/server.js missing or unreadable",
      pass: Boolean(beText) && beLines >= 8 && Buffer.byteLength(beText, "utf8") >= 180,
    },
    {
      criterion: "placeholder_free",
      evidence: placeholderMatches.length > 0
        ? `matched patterns: ${placeholderMatches.join(", ")}`
        : "no placeholder/scaffold markers detected in impl or qa artifacts",
      pass: placeholderMatches.length === 0,
    },
    {
      criterion: "qa_evidence_not_scaffold_only",
      evidence: qaChecks.length > 0
        ? `overall_status=${String(qaReport?.overall_status || "")} checks=${qaChecks.length} pending_human_review=${qaPendingHumanReview}`
        : "verify/qa_report.json missing or empty",
      pass: qaChecks.length > 0 && !qaPendingHumanReview,
    },
    {
      criterion: "preview_matched",
      evidence: previewRoot
        ? `preview.project_root=${previewRoot} source=${previewValidation.source}`
        : "preview_runtime.json missing or project_root empty",
      pass: previewRoot ? !previewMismatch : true,
    },
    {
      criterion: "domain_not_generic_crud",
      evidence: genericCrudMatches.length > 0
        ? `generic CRUD nouns detected: ${genericCrudMatches.join(", ")}`
        : "no generic CRUD-only nouns detected in primary impl artifacts",
      pass: genericCrudMatches.length === 0,
    },
  ];

  if (domainPack) {
    const domainNouns = Array.isArray(domainPack.domain_nouns) ? domainPack.domain_nouns : [];
    // Use substring match to handle camelCase identifiers (e.g. 'customer' in 'renderCustomerList')
    const combinedLower = combinedText.toLowerCase();
    const foundDomainNouns = domainNouns.filter((noun) =>
      combinedLower.includes(noun.toLowerCase())
    );
    reasoning.push({
      criterion: "domain_noun_present",
      evidence: foundDomainNouns.length > 0
        ? `domain nouns found in impl: ${foundDomainNouns.join(", ")} (pack=${domainPack.domain})`
        : `no domain nouns from pack '${domainPack.domain}' detected in impl artifacts (expected: ${domainNouns.slice(0, 4).join(", ")})`,
      pass: foundDomainNouns.length >= 2,
    });
    const forbiddenNouns = Array.isArray(domainPack.forbidden_generic_nouns) ? domainPack.forbidden_generic_nouns : [];
    const foundForbidden = forbiddenNouns.filter((noun) =>
      new RegExp(`\\b${noun}\\b`, "i").test(combinedText)
    );
    reasoning.push({
      criterion: "domain_forbidden_nouns_absent",
      evidence: foundForbidden.length > 0
        ? `forbidden generic nouns found: ${foundForbidden.join(", ")} (pack=${domainPack.domain})`
        : `no forbidden generic nouns from pack '${domainPack.domain}' detected`,
      pass: foundForbidden.length === 0,
    });
  }

  const failedCriteria = reasoning.filter((item) => !item.pass).map((item) => item.criterion);

  const hasUiSignals = hasVisibleUiSignals(frontendSurfaceText);
  // Perceptual quality checks only the FE file itself — QA report scaffolding does not affect visual score
  const fePlaceholderMatches = findPatternMatches(frontendSurfaceText || "", placeholderPatterns);
  const perceptualQuality = {
    layout_intentional: frontendSurfaceText ? frontendSurfaceLines >= 12 : false,
    placeholder_text_detected: placeholderMatches.length > 0,
    interactive_affordances_visible: hasUiSignals,
    // Score "low" when:
    //   - no FE file, or FE itself has placeholder markers, or too few lines (structural issues)
    //   - OR file has sufficient lines but zero UI rendering signals (pure utility code, no visual layer)
    score: (!frontendSurfaceText || fePlaceholderMatches.length > 0 || frontendSurfaceLines < 8 || !hasUiSignals)
      ? "low"
      : (frontendSurfaceLines >= 18 ? "high" : "mid"),
  };

  // Add perceptual quality criterion to reasoning chain (P1-05)
  reasoning.push({
    criterion: "perceptual_quality_minimum",
    evidence: `perceptual_quality.score=${perceptualQuality.score} layout_intentional=${perceptualQuality.layout_intentional} interactive_affordances=${perceptualQuality.interactive_affordances_visible} placeholder_detected=${perceptualQuality.placeholder_text_detected}`,
    pass: perceptualQuality.score !== "low",
  });

  let classification = "demo_usable";
  if (previewMismatch) {
    classification = "preview_mismatch";
  } else if (
    failedCriteria.includes("placeholder_free") ||
    failedCriteria.includes("frontend_depth") ||
    failedCriteria.includes("backend_depth") ||
    qaPendingHumanReview
  ) {
    classification = "artifact_complete_but_shallow";
  } else if (failedCriteria.includes("domain_not_generic_crud")) {
    classification = "domain_misaligned";
  } else if (perceptualQuality.score === "low") {
    // All functional checks passed but visual presentation is below minimum bar (P1-05)
    classification = "visually_incomplete";
  }

  const warning = classification === "demo_usable"
    ? ""
    : `product_fidelity_warning:${classification}`;

  return {
    workflow_run_id: String(run?.workflow_run_id || ""),
    run_id: String(run?.run_id || ""),
    workflow_id: String(run?.workflow_id || ""),
    project_type: String(run?.project_type || ""),
    rubric_path: rubricPath,
    rubric_version: "2026-03-27.v1",
    domain_pack: domainPack ? { domain: domainPack.domain, version: domainPack.version } : null,
    classification,
    warning,
    should_warn: Boolean(warning),
    reasoning,
    perceptual_quality: perceptualQuality,
    generated_at: new Date().toISOString(),
  };
}

export function buildGoNoGoResult({
  run,
  manifest,
  steps = [],
  validator,
  canaryReport = null,
  productFidelityReport = null,
  productFidelityReportRelPath = "",
  previewValidationReport = null,
  previewValidationReportRelPath = "",
  expectedSteps = 0,
  strict = true,
  fidelityGateMode = "warning",
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
  if (String(fidelityGateMode || "") === "blocking") {
    if (productFidelityReport?.should_warn) {
      reasons.push(`product_fidelity_gate: ${String(productFidelityReport?.classification || "failed")} — ${String(productFidelityReport?.warning || "product_fidelity_warning")}`);
    }
    if (previewValidationReport?.should_warn) {
      reasons.push(`preview_validation_gate: ${String(previewValidationReport?.classification || "failed")} — ${String(previewValidationReport?.warning || "preview_validation_warning")}`);
    }
  }
  const productFidelityWarning = productFidelityReport?.should_warn
    ? {
      classification: String(productFidelityReport?.classification || ""),
      warning: String(productFidelityReport?.warning || ""),
      report_path: String(productFidelityReportRelPath || "").replace(/\\/g, "/"),
    }
    : null;
  const previewValidationWarning = previewValidationReport?.should_warn
    ? {
      classification: String(previewValidationReport?.classification || ""),
      warning: String(previewValidationReport?.warning || ""),
      report_path: String(previewValidationReportRelPath || "").replace(/\\/g, "/"),
    }
    : null;
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
    product_fidelity_warning: productFidelityWarning,
    preview_validation_warning: previewValidationWarning,
  };
}
