/**
 * workflow_step_validator.js
 *
 * Step-level validation helpers extracted from workflow_engine.js.
 * Covers artifact presence, implementation delta, role output, handoff, and QA evidence.
 * Extracted as part of WS-11-04 decomposition.
 */

import fs from "fs";
import path from "path";
import { validatePmOutput, validateArchitectOutput, validateReleaseOutput } from "../coding_team_validators.js";
import { validateCodingTeamHandoff } from "../coding_team_handoff_validators.js";
import { validateQaVerifierArtifacts } from "../qa_verifier.js";
import { listWorkflowSteps } from "../data/workflow_repository.js";
import { buildFailurePayload } from "./workflow_artifact_audit.js";
import { parseJsonSafe } from "./workflow_runner.js";

function normalizePathText(value) {
  return String(value || "").replace(/\\/g, "/").replace(/^\/+/, "");
}

function readTextFileSafe(absPath) {
  try {
    return fs.readFileSync(absPath, "utf8");
  } catch {
    return null;
  }
}

function readJsonFileSafe(absPath) {
  try {
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

export function validateExpectedArtifacts(payload = {}, workspaceRoot) {
  const relRoot = String(payload.artifact_root || "").trim().replace(/\\/g, "/");
  const expected = Array.isArray(payload.expected_artifacts) ? payload.expected_artifacts : [];
  if (!relRoot || expected.length === 0) return { checked: false, missing: [], found: [] };
  const rootAbs = path.resolve(workspaceRoot, relRoot);
  const found = [];
  const missing = [];
  for (const rel of expected) {
    const relNorm = String(rel || "").replace(/\\/g, "/").replace(/^\/+/, "");
    if (!relNorm) continue;
    const targetAbs = path.resolve(rootAbs, relNorm);
    // Guard against path traversal.
    if (!targetAbs.startsWith(rootAbs)) { missing.push(relNorm); continue; }
    if (fs.existsSync(targetAbs)) found.push(relNorm);
    else missing.push(relNorm);
  }
  return { checked: true, missing, found, artifact_root: relRoot };
}

function collectChangedFiles(output = {}) {
  return (Array.isArray(output?.files_changed) ? output.files_changed : [])
    .map((item) => normalizePathText(item))
    .filter(Boolean);
}

function isCodingTeamImplementationStep(run, stepId) {
  return (
    String(run?.workflow_id || "") === "coding_team_v0" &&
    (["impl_fe", "impl_fe_skeleton", "impl_fe_modules"].includes(String(stepId || "")) || String(stepId || "") === "impl_be")
  );
}

export function validateImplementationDelta({ run, stepId, output, payload, workspaceRoot }) {
  if (!isCodingTeamImplementationStep(run, stepId)) return { checked: false, ok: true };
  if (String(stepId || "") === "impl_be") {
    const relRoot = String(payload?.artifact_root || "").trim().replace(/\\/g, "/");
    const beDirRel = `${relRoot}/impl/be_changes`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const beDirAbs = path.resolve(workspaceRoot, beDirRel);
    const notesRel = `${relRoot}/impl/be_notes.md`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const notesAbs = path.resolve(workspaceRoot, notesRel);
    const handoffRel = `${relRoot}/handoff/be_to_fe.json`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const handoffAbs = path.resolve(workspaceRoot, handoffRel);
    const packageRel = `${relRoot}/impl/be_changes/package.json`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const packageAbs = path.resolve(workspaceRoot, packageRel);
    const patchRel = `${relRoot}/impl/be_patch_bundle.json`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const patchAbs = path.resolve(workspaceRoot, patchRel);
    const dirExists = fs.existsSync(beDirAbs) && fs.statSync(beDirAbs).isDirectory();
    const dirEntries = dirExists ? fs.readdirSync(beDirAbs, { withFileTypes: true }).filter((item) => item.isFile()) : [];
    const patchBundle = readJsonFileSafe(patchAbs);
    const patchMode = String(patchBundle?.mode || "");
    const patchTargets = Array.isArray(patchBundle?.target_files) ? patchBundle.target_files.map((item) => normalizePathText(item)).filter(Boolean) : [];
    const hasPatchBundle = fs.existsSync(patchAbs) && patchBundle && (patchMode === "structured_patch" || patchMode === "full_file_fallback");
    const hasFullFileOutputs = dirExists && dirEntries.length > 0;
    if ((!hasPatchBundle && !hasFullFileOutputs) || !fs.existsSync(notesAbs) || !fs.existsSync(handoffAbs) || !fs.existsSync(packageAbs)) {
      return {
        checked: true,
        ok: false,
        code: "STEP_IMPL_BE_ARTIFACTS_MISSING",
        detail: "backend implementation step requires either impl/be_patch_bundle.json or non-empty impl/be_changes/, plus impl/be_changes/package.json, impl/be_notes.md and handoff/be_to_fe.json",
        dir_exists: dirExists,
        be_changes_count: dirEntries.length,
        patch_bundle_exists: Boolean(hasPatchBundle),
      };
    }
    // Module system consistency check: package.json type vs server.js syntax
    const packageJson = readJsonFileSafe(packageAbs);
    const serverSource = readTextFileSafe(path.resolve(beDirAbs, "server.js")) || "";
    if (packageJson && String(packageJson.type || "") === "module" && /\brequire\s*\(/.test(serverSource)) {
      return {
        checked: true,
        ok: false,
        code: "STEP_MODULE_SYSTEM_MISMATCH",
        detail: 'package.json declares "type":"module" but server.js uses require() (CommonJS). Server will crash. Remove "type":"module" or convert to ESM imports.',
      };
    }
    // Architecture-implementation consistency: check persistence layer matches arch declaration
    const archMdAbs = path.resolve(workspaceRoot, relRoot, "plan/arch.md");
    const archMdText = (readTextFileSafe(archMdAbs) || "").toLowerCase();
    if (archMdText && serverSource) {
      const archDeclaresSqlite = /sqlite|sequelize|better-sqlite|knex/.test(archMdText);
      const implUsesSqlite = /sqlite|sequelize|better-sqlite|knex/.test(serverSource.toLowerCase());
      const implUsesInMemory = /\bnew Map\b|\bMap\(\)/.test(serverSource) && !/sqlite|sequelize/.test(serverSource.toLowerCase());
      if (archDeclaresSqlite && implUsesInMemory) {
        return {
          checked: true,
          ok: false,
          code: "STEP_ARCH_IMPL_MISMATCH",
          detail: "Architecture declares SQLite persistence but implementation uses in-memory Map. Implement the declared persistence layer or update the architecture.",
        };
      }
    }
    // Handoff completeness: be_to_fe.json api_contracts should cover architect interfaces
    const beToFeJson = readJsonFileSafe(handoffAbs);
    const archHandoffAbs = path.resolve(workspaceRoot, relRoot, "handoff/architect_to_impl.json");
    const archHandoff = readJsonFileSafe(archHandoffAbs);
    if (beToFeJson && archHandoff) {
      const archInterfaces = Array.isArray(archHandoff.interfaces) ? archHandoff.interfaces : [];
      const beContracts = Array.isArray(beToFeJson.api_contracts) ? beToFeJson.api_contracts : [];
      const bePaths = new Set(beContracts.map((c) => String(c.path || c.endpoint || "").toLowerCase()));
      const missingContracts = archInterfaces.filter((iface) => {
        const ifacePath = String(iface || "").replace(/^(GET|POST|PUT|PATCH|DELETE)\s+/i, "").toLowerCase();
        return ifacePath && !bePaths.has(ifacePath);
      });
      if (missingContracts.length > 0 && archInterfaces.length > 2) {
        // Warn but don't fail — handoff may legitimately subset for this sprint
      }
    }
    const scopedFiles = hasPatchBundle && patchTargets.length > 0
      ? patchTargets
      : dirEntries.map((e) => `${relRoot}/impl/be_changes/${e.name}`.replace(/\/+/g, "/").replace(/^\/+/, ""));
    return {
      checked: true,
      ok: true,
      execution_mode_used: hasPatchBundle ? patchMode : "full_file_fallback",
      be_changes_dir: "impl/be_changes",
      be_changes_count: dirEntries.length,
      package_path: "impl/be_changes/package.json",
      patch_bundle_path: hasPatchBundle ? "impl/be_patch_bundle.json" : "",
      notes_path: "impl/be_notes.md",
      handoff_path: "handoff/be_to_fe.json",
      scoped_files: scopedFiles,
    };
  }
  if (["impl_fe", "impl_fe_skeleton", "impl_fe_modules"].includes(String(stepId || ""))) {
    const relRoot = String(payload?.artifact_root || "").trim().replace(/\\/g, "/");
    const feDirRel = `${relRoot}/impl/fe_changes`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const feDirAbs = path.resolve(workspaceRoot, feDirRel);
    const fePublicDirAbs = path.join(feDirAbs, "public");
    const notesRel = `${relRoot}/impl/fe_notes.md`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const notesAbs = path.resolve(workspaceRoot, notesRel);
    const beHandoffRel = `${relRoot}/handoff/be_to_fe.json`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const beHandoffAbs = path.resolve(workspaceRoot, beHandoffRel);
    const patchRel = `${relRoot}/impl/fe_patch_bundle.json`.replace(/\/+/g, "/").replace(/^\/+/, "");
    const patchAbs = path.resolve(workspaceRoot, patchRel);
    const dirExists = fs.existsSync(feDirAbs) && fs.statSync(feDirAbs).isDirectory();
    const dirEntries = dirExists ? fs.readdirSync(feDirAbs, { withFileTypes: true }).filter((item) => item.isFile()) : [];
    const publicEntries = fs.existsSync(fePublicDirAbs) && fs.statSync(fePublicDirAbs).isDirectory()
      ? fs.readdirSync(fePublicDirAbs, { withFileTypes: true }).filter((item) => item.isFile())
      : [];
    const patchBundle = readJsonFileSafe(patchAbs);
    const patchMode = String(patchBundle?.mode || "");
    const patchTargets = Array.isArray(patchBundle?.target_files) ? patchBundle.target_files.map((item) => normalizePathText(item)).filter(Boolean) : [];
    const hasPatchBundle = fs.existsSync(patchAbs) && patchBundle && (patchMode === "structured_patch" || patchMode === "full_file_fallback");
    const hasFullFileOutputs = dirExists && (dirEntries.length > 0 || publicEntries.length > 0);
    if (!fs.existsSync(beHandoffAbs)) {
      return {
        checked: true,
        ok: false,
        code: "STEP_IMPL_FE_HANDOFF_MISSING",
        detail: "frontend step requires upstream handoff/be_to_fe.json",
      };
    }
    if ((!hasPatchBundle && !hasFullFileOutputs) || !fs.existsSync(notesAbs)) {
      return {
        checked: true,
        ok: false,
        code: "STEP_IMPL_FE_ARTIFACTS_MISSING",
        detail: "frontend implementation step requires either impl/fe_patch_bundle.json or non-empty impl/fe_changes/public/, plus impl/fe_notes.md",
        dir_exists: dirExists,
        fe_changes_count: dirEntries.length + publicEntries.length,
        patch_bundle_exists: Boolean(hasPatchBundle),
      };
    }
    const scopedFiles = hasPatchBundle && patchTargets.length > 0
      ? patchTargets
      : [
          ...dirEntries.map((e) => `${relRoot}/impl/fe_changes/${e.name}`.replace(/\/+/g, "/").replace(/^\/+/, "")),
          ...publicEntries.map((e) => `${relRoot}/impl/fe_changes/public/${e.name}`.replace(/\/+/g, "/").replace(/^\/+/, "")),
        ];
    return {
      checked: true,
      ok: true,
      execution_mode_used: hasPatchBundle ? patchMode : "full_file_fallback",
      fe_changes_dir: "impl/fe_changes",
      fe_changes_count: dirEntries.length + publicEntries.length,
      patch_bundle_path: hasPatchBundle ? "impl/fe_patch_bundle.json" : "",
      notes_path: "impl/fe_notes.md",
      consumed_handoff_path: "handoff/be_to_fe.json",
      scoped_files: scopedFiles,
    };
  }
  const changedFiles = collectChangedFiles(output || {});
  const diffFilesRaw = Number(output?.diff_stats?.files || 0);
  const diffFiles = Number.isFinite(diffFilesRaw) ? diffFilesRaw : 0;
  if (changedFiles.length === 0 || diffFiles <= 0) {
    return {
      checked: true, ok: false,
      code: "STEP_CODE_NOT_CHANGED",
      detail: "implementation step completed without real code delta",
      changed_files: changedFiles, diff_files: diffFiles, scoped_files: [],
    };
  }
  const targetPathsRaw = Array.isArray(payload?.target_paths) ? payload.target_paths : [];
  const targetPaths = (targetPathsRaw.length > 0 ? targetPathsRaw : ["workspace/sandbox/crm_site/"])
    .map((item) => normalizePathText(item).replace(/\/+$/, "") + "/");
  const scopedFiles = changedFiles.filter((f) => targetPaths.some((prefix) => f.startsWith(prefix)));
  if (scopedFiles.length === 0) {
    return {
      checked: true, ok: false,
      code: "STEP_CODE_OUT_OF_SCOPE",
      detail: `implementation delta does not touch required paths: ${targetPaths.join(", ")}`,
      changed_files: changedFiles, diff_files: diffFiles, scoped_files: [],
    };
  }
  return { checked: true, ok: true, changed_files: changedFiles, diff_files: diffFiles, scoped_files: scopedFiles, target_paths: targetPaths };
}

function findHandoffForStep(stepId, handoffContracts) {
  return (
    Object.values(handoffContracts?.handoffs || {}).find(
      (item) => String(item?.from_step || "") === String(stepId || "")
    ) || null
  );
}

export function validateDocumentHandoff({ payload, stepId, workspaceRoot, handoffContracts }) {
  const handoff = findHandoffForStep(stepId, handoffContracts);
  return validateCodingTeamHandoff({ workspaceRoot, artifactRoot: payload.artifact_root, handoff });
}

export function validateSmokeVerdict({ run, stepId, payload, workspaceRoot }) {
  if (!(String(run?.workflow_id || "") === "coding_team_v0" && String(stepId || "") === "smoke_test")) {
    return { checked: false, ok: true };
  }
  const relRoot = String(payload?.artifact_root || "").trim().replace(/\\/g, "/");
  const smokeResultPath = path.resolve(workspaceRoot, relRoot, "smoke", "smoke_result.json");
  const smokeResult = readJsonFileSafe(smokeResultPath);
  if (!smokeResult) {
    return { checked: true, ok: false, code: "SMOKE_RESULT_MISSING", detail: "smoke/smoke_result.json not found or not valid JSON" };
  }
  const verdict = String(smokeResult.verdict || smokeResult.overall_status || "").toLowerCase();
  if (verdict === "fail" || verdict === "failed") {
    const errors = Array.isArray(smokeResult.errors) ? smokeResult.errors.map((e) => String(e).slice(0, 120)).join("; ") : "";
    return {
      checked: true,
      ok: false,
      code: "SMOKE_TEST_VERDICT_FAIL",
      detail: `smoke test verdict '${verdict}': ${errors || "server could not start or endpoints failed"}`,
    };
  }
  return { checked: true, ok: true, verdict };
}

export function validateGoalFidelity({ workspaceRoot, artifactRoot, goal }) {
  if (!goal || typeof goal !== "string" || goal.trim().length < 10) {
    return { checked: false, ok: true };
  }
  const rootAbs = path.resolve(workspaceRoot, String(artifactRoot || "").replace(/\\/g, "/"));
  if (!fs.existsSync(rootAbs)) return { checked: false, ok: true };
  const goalLower = goal.toLowerCase();
  // v3.5: Extract top-level feature phrases more robustly.
  // Strategy: split on semicolons and numbered markers (1)/(2)/(3) first (module-level),
  // then fall back to comma+and splitting for simpler goals.
  let featurePhrases;
  const hasNumberedModules = /\(\d+\)/.test(goalLower);
  if (hasNumberedModules) {
    // Split by numbered markers: "(1) ... ; (2) ... ; (3) ..."
    // Also capture trailing sentence after last module (e.g. "All modules must have...")
    featurePhrases = goalLower
      .split(/\s*[;.]\s*(?:\(\d+\)\s*)?|\s*\(\d+\)\s*/)
      .map((s) => s.trim())
      .filter((s) => s.length > 5);
  } else {
    const stripped = goalLower.replace(/^.*?(?:with|that has|featuring|including)\s+/i, "");
    featurePhrases = stripped
      .split(/\s*(?:,\s*|\s+and\s+)/)
      .map((s) => s.trim())
      .filter((s) => s.length > 3);
  }
  if (featurePhrases.length === 0) return { checked: false, ok: true };
  const specText = (readTextFileSafe(path.resolve(rootAbs, "plan/spec.md")) || "").toLowerCase();
  const nonGoalsMatch = specText.match(/non[- ]?goals?[\s\S]*?(?=\n#|\n\*\*|$)/i);
  const nonGoalsText = nonGoalsMatch ? nonGoalsMatch[0] : "";
  const missing = [];
  const contradicted = [];
  for (const phrase of featurePhrases) {
    // Strip punctuation from keywords before matching
    const keywords = phrase.split(/\s+/)
      .map((w) => w.replace(/[^a-z0-9_/-]/g, ""))
      .filter((w) => w.length > 4);
    if (keywords.length === 0) continue;
    const inSpec = keywords.some((w) => specText.includes(w));
    if (!inSpec) missing.push(phrase);
    // Check Non-Goals contradiction: require the FULL PHRASE (not single keyword) to appear,
    // to avoid false positives like "between customers and agents" matching "customer management"
    const phraseInNonGoals = nonGoalsText.includes(phrase);
    if (phraseInNonGoals) contradicted.push(phrase);
  }
  if (missing.length > 0 || contradicted.length > 0) {
    const parts = [];
    if (missing.length > 0) parts.push(`missing from spec: ${missing.join(", ")}`);
    if (contradicted.length > 0) parts.push(`incorrectly placed in Non-Goals: ${contradicted.join(", ")}`);
    return {
      checked: true, ok: false,
      code: "GOAL_FIDELITY_VIOLATION",
      detail: `goal features ${parts.join("; ")}`,
      missing_features: missing,
      contradicted_features: contradicted,
      all_features: featurePhrases,
    };
  }
  return { checked: true, ok: true, features_verified: featurePhrases };
}

export function validateRoleOutput({ payload, stepId, workspaceRoot, goal }) {
  if (String(stepId || "") === "pm_spec") {
    const pmResult = validatePmOutput({ workspaceRoot, artifactRoot: payload.artifact_root });
    if (pmResult.checked && !pmResult.ok) return pmResult;
    const fidelity = validateGoalFidelity({ workspaceRoot, artifactRoot: payload.artifact_root, goal });
    if (fidelity.checked && !fidelity.ok) return fidelity;
    return pmResult.checked ? pmResult : fidelity;
  }
  if (String(stepId || "") === "arch_design") return validateArchitectOutput({ workspaceRoot, artifactRoot: payload.artifact_root });
  if (String(stepId || "") === "release_pack") return validateReleaseOutput({ workspaceRoot, artifactRoot: payload.artifact_root });
  return { checked: false, ok: true };
}

export async function validateQaEvidence({ run, workflow_run_id, payload, pool, workspaceRoot }) {
  if (!(String(run?.workflow_id || "") === "coding_team_v0" && String(payload?.step_id || "") === "qa_verify")) {
    return { checked: false, ok: true };
  }
  const qaArtifacts = validateQaVerifierArtifacts({ workspaceRoot, artifactRoot: payload.artifact_root });
  if (!qaArtifacts.ok) return qaArtifacts;

  const steps = await listWorkflowSteps(pool, workflow_run_id);
  const implStepRows = steps.filter((s) => ["impl_fe", "impl_fe_skeleton", "impl_fe_modules", "impl_be"].includes(String(s?.step_id || "")));
  const scoped = [];
  for (const step of implStepRows) {
    const result = parseJsonSafe(step.result_json, {});
    const files = Array.isArray(result?.impl_validation?.scoped_files) ? result.impl_validation.scoped_files : [];
    for (const file of files) scoped.push(normalizePathText(file));
  }
  const dedupScoped = Array.from(new Set(scoped));
  if (dedupScoped.length === 0) {
    return {
      checked: true, ok: false,
      code: "STEP_QA_NO_IMPL_DELTA",
      detail: "qa step has no upstream scoped implementation delta",
      scoped_files: [],
    };
  }

  const input = parseJsonSafe(run.input_json, {});
  const goal = String(input.goal || "").toLowerCase();
  const checks = [];
  if (/health check|健康检查/.test(goal)) checks.push({ id: "health_check", re: /health\s*check|健康检查/i });
  if (/current time|当前时间/.test(goal)) checks.push({ id: "current_time", re: /current\s*time|当前时间|tolocalestring|date\(/i });
  if (checks.length === 0) return { checked: true, ok: true, scoped_files: dedupScoped, keyword_checks: [] };

  const missing = [];
  for (const check of checks) {
    let matched = false;
    for (const rel of dedupScoped) {
      const text = readTextFileSafe(path.resolve(workspaceRoot, rel));
      if (text && check.re.test(text)) { matched = true; break; }
    }
    if (!matched) missing.push(check.id);
  }
  if (missing.length > 0) {
    return {
      checked: true, ok: false,
      code: "STEP_QA_EVIDENCE_MISSING",
      detail: `qa evidence missing keyword checks: ${missing.join(", ")}`,
      scoped_files: dedupScoped, missing_checks: missing,
    };
  }
  return { checked: true, ok: true, qa_artifacts: qaArtifacts, scoped_files: dedupScoped, keyword_checks: checks.map((c) => c.id) };
}

/**
 * Run all step-success validations in order; returns on first failure.
 * Does NOT write to DB — the caller handles DB writes and event emission.
 *
 * @returns {{ ok: boolean, mergedOutput: object, logMissingArtifacts: boolean,
 *             code?: string, message?: string, failurePayload?: object, failKey?: string }}
 */
export async function runStepSuccessValidations({
  run, step_id, payload, output, workflow_run_id,
  pool, workspaceRoot, handoffContracts,
  auditStepArtifacts, strictStepArtifacts,
}) {
  const artifactAudit = auditStepArtifacts
    ? validateExpectedArtifacts(payload, workspaceRoot)
    : { checked: false, missing: [], found: [] };

  let mergedOutput = { ...(output || {}), artifact_check: artifactAudit };
  const logMissingArtifacts = artifactAudit.checked && artifactAudit.missing.length > 0;

  if (strictStepArtifacts && artifactAudit.checked && artifactAudit.missing.length > 0) {
    const failurePayload = buildFailurePayload({
      errorCode: "STEP_ARTIFACT_MISSING",
      failedStep: step_id,
      missing: artifactAudit.missing,
      invalid: [],
      detail: `missing expected artifacts: ${artifactAudit.missing.join(", ")}`,
    });
    mergedOutput = { ...(output || {}), artifact_check: artifactAudit, failure_payload: failurePayload };
    return {
      ok: false, code: "STEP_ARTIFACT_MISSING",
      message: `missing expected artifacts: ${artifactAudit.missing.join(", ")}`,
      failurePayload, mergedOutput, logMissingArtifacts, failKey: "artifact",
    };
  }

  const implValidation = validateImplementationDelta({ run, stepId: step_id, output: output || {}, payload, workspaceRoot });
  if (implValidation.checked) mergedOutput = { ...mergedOutput, impl_validation: implValidation };
  if (implValidation.checked && !implValidation.ok) {
    const failurePayload = buildFailurePayload({
      errorCode: implValidation.code || "STEP_CODE_NOT_CHANGED",
      failedStep: step_id,
      detail: implValidation.detail || "implementation delta validation failed",
    });
    mergedOutput = { ...(output || {}), artifact_check: artifactAudit, impl_validation: implValidation, failure_payload: failurePayload };
    return {
      ok: false, code: implValidation.code || "STEP_CODE_NOT_CHANGED",
      message: implValidation.detail || "implementation delta validation failed",
      failurePayload, mergedOutput, logMissingArtifacts, failKey: "impl",
    };
  }

  const inputGoal = String(parseJsonSafe(run?.input_json, {})?.goal || "");
  const roleOutputValidation = validateRoleOutput({ payload, stepId: step_id, workspaceRoot, goal: inputGoal });
  if (roleOutputValidation.checked) mergedOutput = { ...mergedOutput, role_output_validation: roleOutputValidation };
  if (roleOutputValidation.checked && !roleOutputValidation.ok) {
    const failurePayload = buildFailurePayload({
      errorCode: roleOutputValidation.code || "ROLE_OUTPUT_VALIDATION_FAILED",
      failedStep: step_id,
      detail: roleOutputValidation.detail || "role output validation failed",
    });
    mergedOutput = { ...(output || {}), artifact_check: artifactAudit, role_output_validation: roleOutputValidation, failure_payload: failurePayload };
    return {
      ok: false, code: roleOutputValidation.code || "ROLE_OUTPUT_VALIDATION_FAILED",
      message: roleOutputValidation.detail || "role output validation failed",
      failurePayload, mergedOutput, logMissingArtifacts, failKey: "role",
    };
  }

  const handoffValidation = validateDocumentHandoff({ payload, stepId: step_id, workspaceRoot, handoffContracts });
  if (handoffValidation.checked) mergedOutput = { ...mergedOutput, handoff_validation: handoffValidation };
  if (handoffValidation.checked && !handoffValidation.ok) {
    const failurePayload = buildFailurePayload({
      errorCode: handoffValidation.code || "HANDOFF_VALIDATION_FAILED",
      failedStep: step_id,
      detail: handoffValidation.detail || "handoff validation failed",
    });
    mergedOutput = { ...(output || {}), artifact_check: artifactAudit, handoff_validation: handoffValidation, failure_payload: failurePayload };
    return {
      ok: false, code: handoffValidation.code || "HANDOFF_VALIDATION_FAILED",
      message: handoffValidation.detail || "handoff validation failed",
      failurePayload, mergedOutput, logMissingArtifacts, failKey: "handoff",
    };
  }

  const smokeValidation = validateSmokeVerdict({ run, stepId: step_id, payload, workspaceRoot });
  if (smokeValidation.checked) mergedOutput = { ...mergedOutput, smoke_validation: smokeValidation };
  if (smokeValidation.checked && !smokeValidation.ok) {
    const failurePayload = buildFailurePayload({
      errorCode: smokeValidation.code || "SMOKE_TEST_VERDICT_FAIL",
      failedStep: step_id,
      detail: smokeValidation.detail || "smoke test verdict indicates failure",
    });
    mergedOutput = { ...(output || {}), artifact_check: artifactAudit, smoke_validation: smokeValidation, failure_payload: failurePayload };
    return {
      ok: false, code: smokeValidation.code,
      message: smokeValidation.detail,
      failurePayload, mergedOutput, logMissingArtifacts, failKey: "smoke",
    };
  }

  const qaValidation = await validateQaEvidence({ run, workflow_run_id, payload, pool, workspaceRoot });
  if (qaValidation.checked) mergedOutput = { ...mergedOutput, qa_validation: qaValidation };
  if (qaValidation.checked && !qaValidation.ok) {
    const failurePayload = buildFailurePayload({
      errorCode: qaValidation.code || "STEP_QA_EVIDENCE_MISSING",
      failedStep: step_id,
      detail: qaValidation.detail || "qa evidence validation failed",
    });
    mergedOutput = { ...(output || {}), artifact_check: artifactAudit, qa_validation: qaValidation, failure_payload: failurePayload };
    return {
      ok: false, code: qaValidation.code || "STEP_QA_EVIDENCE_MISSING",
      message: qaValidation.detail || "qa evidence validation failed",
      failurePayload, mergedOutput, logMissingArtifacts, failKey: "qa",
    };
  }

  return { ok: true, mergedOutput, logMissingArtifacts };
}
