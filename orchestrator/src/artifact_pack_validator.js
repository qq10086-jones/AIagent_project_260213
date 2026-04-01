import fs from "fs";
import path from "path";

function isObj(v) {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

function safeReadJson(p) {
  try {
    const raw = fs.readFileSync(p, "utf-8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function loadJsonIfExists(p) {
  if (!p || !fs.existsSync(p)) return null;
  return safeReadJson(p);
}

function resolveRegistryFile(relPath) {
  const rel = String(relPath || "").replace(/\\/g, "/").replace(/^\/+/, "");
  const candidates = [
    path.resolve(process.cwd(), rel),
    path.resolve(process.cwd(), "orchestrator", rel),
    path.resolve(process.cwd(), "..", rel),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function normalizeContractPath(rel) {
  return String(rel || "").replace(/\\/g, "/").replace(/^\/+/, "");
}

function resolveArtifactPath(releaseRoot, relPath) {
  const rel = normalizeContractPath(relPath);
  if (!rel) return null;
  const workspaceRoot = path.resolve(releaseRoot, "..", "..", "..", "..");
  const workspaceRelativeDirect = path.resolve(workspaceRoot, rel);
  if (fs.existsSync(workspaceRelativeDirect)) return workspaceRelativeDirect;
  const releaseRelative = path.resolve(releaseRoot, rel);
  if (fs.existsSync(releaseRelative)) return releaseRelative;

  const marker = `${path.sep}artifacts${path.sep}release${path.sep}`;
  const normalizedReleaseRoot = path.resolve(releaseRoot);
  const markerIndex = normalizedReleaseRoot.lastIndexOf(marker);
  if (markerIndex !== -1) {
    const legacyWorkspaceRoot = normalizedReleaseRoot.slice(0, markerIndex);
    const workspaceRelative = path.resolve(legacyWorkspaceRoot, rel);
    if (fs.existsSync(workspaceRelative)) return workspaceRelative;
  }
  return releaseRelative;
}

function nonEmptyLines(text = "") {
  return String(text || "")
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean);
}

function validateMiniSchema(value, schema, jsonPath = "$", errors = []) {
  if (!schema || typeof schema !== "object") return errors;
  const type = schema.type;
  if (type === "object") {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      errors.push(`${jsonPath} must be object`);
      return errors;
    }
    const req = Array.isArray(schema.required) ? schema.required : [];
    for (const k of req) {
      if (value[k] === undefined) errors.push(`${jsonPath}.${k} missing`);
    }
    const props = isObj(schema.properties) ? schema.properties : {};
    for (const [k, v] of Object.entries(props)) {
      if (value[k] !== undefined) validateMiniSchema(value[k], v, `${jsonPath}.${k}`, errors);
    }
    return errors;
  }
  if (type === "array") {
    if (!Array.isArray(value)) {
      errors.push(`${jsonPath} must be array`);
      return errors;
    }
    if (Number.isInteger(schema.minItems) && value.length < Number(schema.minItems)) {
      errors.push(`${jsonPath} minItems=${schema.minItems}`);
    }
    if (schema.items && typeof schema.items === "object") {
      value.forEach((it, idx) => validateMiniSchema(it, schema.items, `${jsonPath}[${idx}]`, errors));
    }
    return errors;
  }
  if (type === "string") {
    if (typeof value !== "string") {
      errors.push(`${jsonPath} must be string`);
      return errors;
    }
    if (Number.isInteger(schema.minLength) && value.length < Number(schema.minLength)) {
      errors.push(`${jsonPath} minLength=${schema.minLength}`);
    }
    if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
      errors.push(`${jsonPath} must be one of enum`);
    }
    return errors;
  }
  if (type === "boolean") {
    if (typeof value !== "boolean") errors.push(`${jsonPath} must be boolean`);
    return errors;
  }
  if (type === "number" || type === "integer") {
    if (typeof value !== "number" || !Number.isFinite(value)) errors.push(`${jsonPath} must be number`);
    return errors;
  }
  return errors;
}

function validateMarkdownContent({ relPath: _relPath, absPath, rule = {} }) {
  const out = { ok: true, reasons: [] };
  const raw = fs.readFileSync(absPath, "utf8");
  const lines = nonEmptyLines(raw);
  const minLines = Number.isInteger(rule.min_nonempty_lines) ? Number(rule.min_nonempty_lines) : 3;
  if (lines.length < minLines) out.reasons.push(`non-empty lines < ${minLines}`);
  const denyPatterns = Array.isArray(rule.deny_patterns) ? rule.deny_patterns : [];
  for (const ptn of denyPatterns) {
    try {
      const re = new RegExp(String(ptn), "i");
      if (re.test(raw)) out.reasons.push(`contains denied pattern '${ptn}'`);
    } catch {
      // ignore malformed regex in contract
    }
  }
  if (out.reasons.length > 0) {
    out.ok = false;
  }
  return out;
}

function validateJsonContent({ relPath: _relPath, absPath, rule = {} }) {
  const out = { ok: true, reasons: [] };
  const json = safeReadJson(absPath);
  if (!json || typeof json !== "object") {
    out.ok = false;
    out.reasons.push("invalid json");
    return out;
  }
  const schemaRel = String(rule.schema || "").trim();
  if (schemaRel) {
    const schemaPath = resolveRegistryFile(schemaRel);
    if (!schemaPath) {
      out.ok = false;
      out.reasons.push(`schema not found: ${schemaRel}`);
      return out;
    }
    const schema = safeReadJson(schemaPath);
    if (!schema || typeof schema !== "object") {
      out.ok = false;
      out.reasons.push(`schema invalid json: ${schemaRel}`);
      return out;
    }
    const errs = validateMiniSchema(json, schema, "$", []);
    if (errs.length > 0) {
      out.ok = false;
      out.reasons.push(...errs);
    }
  }
  return out;
}

function deriveAcceptanceIds(releaseRoot) {
  const p = path.resolve(releaseRoot, "plan", "acceptance.json");
  const j = loadJsonIfExists(p);
  const criteria = Array.isArray(j?.criteria) ? j.criteria : [];
  const ids = [];
  for (let i = 0; i < criteria.length; i++) {
    const c = criteria[i];
    if (c && typeof c === "object" && typeof c.id === "string" && c.id.trim()) ids.push(c.id.trim());
    else ids.push(`A${i + 1}`);
  }
  return ids;
}

function validateAcceptanceVerificationMapping(releaseRoot) {
  const expectedIds = deriveAcceptanceIds(releaseRoot);
  if (expectedIds.length === 0) return { ok: true, reasons: [] };
  const vPath = path.resolve(releaseRoot, "verify", "qa_report.json");
  const vJson = loadJsonIfExists(vPath);
  if (!vJson || typeof vJson !== "object") {
    return { ok: false, reasons: ["ARTIFACT_INVALID:verify/qa_report.json:report missing"] };
  }
  if (typeof vJson.overall_status !== "string" || !vJson.overall_status.trim()) {
    return { ok: false, reasons: ["ARTIFACT_INVALID:verify/qa_report.json:overall_status missing"] };
  }
  if (!Array.isArray(vJson.checks) || vJson.checks.length < 1) {
    return { ok: false, reasons: ["ARTIFACT_INVALID:verify/qa_report.json:checks missing"] };
  }
  if (!Array.isArray(vJson.verified_artifacts) || vJson.verified_artifacts.length < 1) {
    return { ok: false, reasons: ["ARTIFACT_INVALID:verify/qa_report.json:verified_artifacts missing"] };
  }
  const actualIds = new Set(vJson.verified_artifacts.map((x) => String(x || "").trim()).filter(Boolean));
  const miss = expectedIds.filter((id) => !actualIds.has(id));
  if (miss.length > 0) {
    return {
      ok: false,
      reasons: [`ARTIFACT_INVALID:verify/qa_report.json:missing acceptance ids ${miss.join(",")}`],
    };
  }
  return { ok: true, reasons: [] };
}

function validateContextBudgetCoverage({ releaseRoot, manifest, steps }) {
  const reasons = [];
  const reports = Array.isArray(manifest?.context_budget_reports) ? manifest.context_budget_reports : null;
  if (!reports) {
    return { ok: false, reasons: ["manifest missing field: context_budget_reports"] };
  }
  const expectedSteps = Array.isArray(steps) ? steps.map((s) => String(s?.step_id || "")).filter(Boolean) : [];
  for (const stepId of expectedSteps) {
    const item = reports.find((report) => String(report?.step_id || "") === stepId);
    if (!item) {
      reasons.push(`ARTIFACT_MISSING:metrics/context_budget_${stepId}.json`);
      continue;
    }
    const rel = String(item.report_path || "").trim().replace(/\\/g, "/");
    if (!rel) {
      reasons.push(`ARTIFACT_MISSING:metrics/context_budget_${stepId}.json`);
      continue;
    }
    const abs = resolveArtifactPath(releaseRoot, rel);
    const json = loadJsonIfExists(abs);
    if (!json || typeof json !== "object") {
      reasons.push(`ARTIFACT_INVALID:${rel}:invalid json`);
      continue;
    }
    if (String(json.step_id || "") !== stepId) {
      reasons.push(`ARTIFACT_INVALID:${rel}:step_id mismatch`);
    }
    if (!["ok", "warning", "overflow_risk"].includes(String(json.status || ""))) {
      reasons.push(`ARTIFACT_INVALID:${rel}:status invalid`);
    }
    if (!String(json.threshold_source || "").trim()) {
      reasons.push(`ARTIFACT_INVALID:${rel}:threshold_source missing`);
    }
  }
  const summary = manifest?.context_budget_summary;
  if (!summary || typeof summary !== "object") {
    reasons.push("manifest missing field: context_budget_summary");
  } else {
    if (Number(summary.total_steps || 0) !== expectedSteps.length) {
      reasons.push("manifest context_budget_summary total_steps mismatch");
    }
  }
  return { ok: reasons.length === 0, reasons };
}

function validateArtifactQuality({ run, manifestPath }) {
  const workflowId = String(run?.workflow_id || "");
  const releaseRoot = path.dirname(path.dirname(manifestPath || ""));
  const contractRelCandidates = [
    `orchestrator/configs/contracts/${workflowId}_artifacts.json`,
    `configs/registry/contracts/${workflowId}_artifacts.json`,
  ];
  let contractRel = "";
  let contractPath = null;
  for (const c of contractRelCandidates) {
    const p = resolveRegistryFile(c);
    if (p) {
      contractRel = c;
      contractPath = p;
      break;
    }
  }
  if (!contractPath) {
    return {
      ok: true,
      summary: {
        contract_found: false,
        required: 0,
        present: 0,
        missing: 0,
        invalid: 0,
      },
      reasons: [],
    };
  }
  const contract = loadJsonIfExists(contractPath);
  if (!contract || !Array.isArray(contract.artifacts)) {
    return {
      ok: false,
      summary: {
        contract_found: true,
        required: 0,
        present: 0,
        missing: 0,
        invalid: 1,
      },
      reasons: [`ARTIFACT_INVALID:contract:${contractRel} malformed`],
    };
  }
  let required = 0;
  let present = 0;
  let missing = 0;
  let invalid = 0;
  const reasons = [];

  for (const item of contract.artifacts) {
    const rel = normalizeContractPath(item?.path);
    if (!rel) continue;
    const isRequired = item?.required !== false;
    if (isRequired) required += 1;
    const abs = path.resolve(releaseRoot, rel);
    if (!fs.existsSync(abs)) {
      if (isRequired) {
        missing += 1;
        reasons.push(`ARTIFACT_MISSING:${rel}`);
      }
      continue;
    }
    present += 1;
    const type = String(item?.type || "").toLowerCase();
    let check = { ok: true, reasons: [] };
    if (type === "json") check = validateJsonContent({ relPath: rel, absPath: abs, rule: item });
    if (type === "markdown" || type === "md") check = validateMarkdownContent({ relPath: rel, absPath: abs, rule: item });
    if (!check.ok) {
      invalid += 1;
      reasons.push(`ARTIFACT_INVALID:${rel}:${check.reasons.join("; ")}`);
    }
  }

  const mapping = validateAcceptanceVerificationMapping(releaseRoot);
  if (!mapping.ok) reasons.push(...mapping.reasons);
  if (!mapping.ok) invalid += 1;

  return {
    ok: missing === 0 && invalid === 0,
    summary: {
      contract_found: true,
      contract_path: contractRel,
      required,
      present,
      missing,
      invalid,
    },
    reasons,
  };
}

export function validateArtifactPack({
  run,
  steps = [],
  checkpoints = [],
  manifestPath,
  summaryPath,
  registry,
}) {
  const reasons = [];
  if (!run || !isObj(run)) reasons.push("run object missing");
  if (!Array.isArray(steps) || steps.length === 0) reasons.push("steps missing");
  if (!manifestPath || !fs.existsSync(manifestPath)) reasons.push("run_manifest.json missing");
  if (!summaryPath || !fs.existsSync(summaryPath)) reasons.push("run_summary.md missing");

  const manifest = manifestPath ? safeReadJson(manifestPath) : null;
  if (!manifest || !isObj(manifest)) {
    reasons.push("run_manifest.json invalid json");
  } else {
    const requiredTop = [
      "workflow_run_id",
      "run_id",
      "workflow_id",
      "project_type",
      "status",
      "steps",
      "checkpoints",
      "step_artifacts",
      "context_budget_reports",
      "context_budget_summary",
    ];
    for (const k of requiredTop) {
      if (manifest[k] === undefined || manifest[k] === null) reasons.push(`manifest missing field: ${k}`);
    }
    if (String(manifest.workflow_run_id || "") !== String(run.workflow_run_id || "")) {
      reasons.push("manifest workflow_run_id mismatch");
    }
    if (String(manifest.run_id || "") !== String(run.run_id || "")) {
      reasons.push("manifest run_id mismatch");
    }
    if (!Array.isArray(manifest.steps) || manifest.steps.length === 0) reasons.push("manifest steps empty");
    if (!Array.isArray(manifest.checkpoints)) reasons.push("manifest checkpoints not array");
    if (!Array.isArray(manifest.step_artifacts)) reasons.push("manifest step_artifacts not array");
    if (Array.isArray(manifest.step_artifacts) && Array.isArray(manifest.steps) && manifest.step_artifacts.length !== manifest.steps.length) {
      reasons.push("manifest step_artifacts length mismatch");
    }
    if (String(manifest.status || "") !== "succeeded") reasons.push("manifest status must be succeeded");
  }

  const succeededSteps = steps.filter((s) => String(s.status || "") === "succeeded");
  if (succeededSteps.length !== steps.length) reasons.push("not all workflow steps are succeeded");
  if (Array.isArray(checkpoints) && checkpoints.length < succeededSteps.length) {
    reasons.push("checkpoint count lower than succeeded steps");
  }

  const requiredArtifacts = registry?.project_types?.[run?.project_type]?.required_artifacts || [];
  if (manifest && isObj(manifest.artifact_coverage)) {
    for (const item of requiredArtifacts) {
      if (!manifest.artifact_coverage[item]) reasons.push(`required artifact missing: ${item}`);
    }
  }

  const quality = validateArtifactQuality({ run, manifestPath });
  if (!quality.ok) reasons.push(...quality.reasons);
  const contextBudget = validateContextBudgetCoverage({
    releaseRoot: path.dirname(path.dirname(manifestPath || "")),
    manifest,
    steps,
  });
  if (!contextBudget.ok) reasons.push(...contextBudget.reasons);

  return {
    ok: reasons.length === 0,
    reasons,
    manifest,
    quality_summary: quality.summary,
    summary_path: summaryPath || null,
    manifest_path: manifestPath || null,
  };
}
