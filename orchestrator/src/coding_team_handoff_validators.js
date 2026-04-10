import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "./schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "contracts");
const PM_HANDOFF_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_pm_handoff.schema.json"), "utf8")
);
const ARCH_HANDOFF_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_arch_handoff.schema.json"), "utf8")
);
const BE_TO_FE_HANDOFF_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_be_to_fe_handoff.schema.json"), "utf8")
);
const IMPL_TO_QA_HANDOFF_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_impl_to_qa_handoff.schema.json"), "utf8")
);
const QA_TO_RELEASE_HANDOFF_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_qa_to_release_handoff.schema.json"), "utf8")
);

function readTextFileSafe(absPath, maxBytes = 262144) {
  try {
    const st = fs.statSync(absPath);
    if (!st.isFile()) return "";
    if (st.size > maxBytes) return "";
    return fs.readFileSync(absPath, "utf8");
  } catch {
    return "";
  }
}

function readJsonFileSafe(absPath) {
  try {
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

function hasOwnPath(obj, dottedPath) {
  const parts = String(dottedPath || "").split(".").filter(Boolean);
  if (parts.length === 0) return false;
  let current = obj;
  for (const part of parts) {
    if (!current || !Object.prototype.hasOwnProperty.call(current, part)) return false;
    current = current[part];
  }
  return true;
}

function normalizeSectionToken(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getSchemaForTypedHandoff(fileName) {
  const safe = String(fileName || "").replace(/\\/g, "/");
  if (safe.endsWith("pm_to_architect.json")) return PM_HANDOFF_SCHEMA;
  if (safe.endsWith("architect_to_impl.json")) return ARCH_HANDOFF_SCHEMA;
  if (safe.endsWith("be_to_fe.json")) return BE_TO_FE_HANDOFF_SCHEMA;
  if (safe.endsWith("impl_to_qa.json")) return IMPL_TO_QA_HANDOFF_SCHEMA;
  if (safe.endsWith("qa_to_release.json")) return QA_TO_RELEASE_HANDOFF_SCHEMA;
  return null;
}

export function validateCodingTeamHandoff({ workspaceRoot, artifactRoot, handoff }) {
  if (!handoff) return { checked: false, ok: true };

  const relRoot = String(artifactRoot || "").trim().replace(/\\/g, "/");
  if (!relRoot) {
    return { checked: true, ok: false, code: "HANDOFF_ARTIFACT_ROOT_MISSING", detail: "artifact_root missing" };
  }
  const rootAbs = path.resolve(workspaceRoot, relRoot);
  const missingArtifacts = [];
  for (const rel of handoff.required_artifacts || []) {
    const abs = path.resolve(rootAbs, String(rel || "").replace(/\\/g, "/"));
    if (!fs.existsSync(abs)) missingArtifacts.push(rel);
  }
  if (missingArtifacts.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "HANDOFF_ARTIFACTS_MISSING",
      detail: `missing handoff artifacts: ${missingArtifacts.join(", ")}`,
      handoff,
    };
  }

  const searchableFiles = (handoff.required_artifacts || [])
    .filter((item) => /\.md$/i.test(String(item || "")))
    .map((item) => path.resolve(rootAbs, String(item).replace(/\\/g, "/")));
  const corpus = searchableFiles
    .map((file) => readTextFileSafe(file).toLowerCase())
    .join("\n");
  const normalizedCorpus = normalizeSectionToken(corpus);
  // Also collect top-level keys from JSON artifacts so required_sections can be
  // satisfied by a JSON handoff file (e.g. be_to_fe.json with api_contracts key).
  const jsonTopLevelKeys = new Set(
    (handoff.required_artifacts || [])
      .filter((item) => /\.json$/i.test(String(item || "")))
      .flatMap((item) => {
        const obj = readJsonFileSafe(path.resolve(rootAbs, String(item).replace(/\\/g, "/")));
        return obj && typeof obj === "object" && !Array.isArray(obj) ? Object.keys(obj) : [];
      })
      .map((k) => normalizeSectionToken(k))
  );
  // Section aliases: LLMs use different headings for the same concept.
  // Each required_section is checked against these alternative tokens.
  const SECTION_ALIASES = {
    "non_goals": ["non goals", "non goal", "out of scope"],
    "artifact_list": ["artifact list", "artifacts", "milestones", "deliverables"],
    "user_stories": ["user stories", "user story"],
    "acceptance_criteria": ["acceptance criteria", "acceptance"],
  };
  const missingSections = (handoff.required_sections || []).filter((section) => {
    const token = normalizeSectionToken(section);
    if (!token) return false;
    const tokens = SECTION_ALIASES[section] || [token];
    const inCorpus = tokens.some((t) => normalizedCorpus.includes(t));
    const inJsonKeys = tokens.some((t) => jsonTopLevelKeys.has(t));
    return !inCorpus && !inJsonKeys;
  });
  if (missingSections.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "HANDOFF_SECTIONS_MISSING",
      detail: `missing handoff sections: ${missingSections.join(", ")}`,
      handoff,
    };
  }

  const typed = handoff.typed_handoff || null;
  if (typed) {
    const manifestPath = path.resolve(rootAbs, String(typed.file || "").replace(/\\/g, "/"));
    const manifest = readJsonFileSafe(manifestPath);
    if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
      return {
        checked: true,
        ok: false,
        code: "HANDOFF_TYPED_MANIFEST_INVALID",
        detail: `invalid typed handoff manifest: ${typed.file}`,
        handoff,
      };
    }
    const missingFields = (typed.required_fields || []).filter((field) => !hasOwnPath(manifest, field));
    if (missingFields.length > 0) {
      return {
        checked: true,
        ok: false,
        code: "HANDOFF_TYPED_FIELDS_MISSING",
        detail: `missing typed handoff fields: ${missingFields.join(", ")}`,
        handoff,
      };
    }
    // Auto-repair: if architect_to_impl.json misses workplan/risks, fill from sibling artifacts
    const safeName = String(typed.file || "").replace(/\\/g, "/");
    if (safeName.endsWith("architect_to_impl.json")) {
      let repaired = false;
      if (!manifest.workplan || typeof manifest.workplan !== "object") {
        const workplanJson = readJsonFileSafe(path.resolve(rootAbs, "plan/workplan.json"));
        if (workplanJson && typeof workplanJson === "object") {
          manifest.workplan = {
            be_tasks: Array.isArray(workplanJson.be_tasks) ? workplanJson.be_tasks : [],
            fe_tasks: Array.isArray(workplanJson.fe_tasks) ? workplanJson.fe_tasks : [],
          };
          repaired = true;
        }
      }
      if (!Array.isArray(manifest.risks) || manifest.risks.length === 0) {
        const riskReport = readJsonFileSafe(path.resolve(rootAbs, "risk/risk_report.json"));
        if (riskReport && Array.isArray(riskReport.risks)) {
          manifest.risks = riskReport.risks.map((r) =>
            typeof r === "string" ? r : String(r.title || r.description || r.risk || "unknown risk")
          );
          repaired = true;
        }
      }
      if (repaired) {
        try { fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2)); } catch { /* best-effort */ }
      }
    }

    const manifestSchema = getSchemaForTypedHandoff(typed.file);
    if (manifestSchema) {
      const schemaErrors = validateJsonSchemaLite(manifestSchema, manifest, "$");
      if (schemaErrors.length > 0) {
        return {
          checked: true,
          ok: false,
          code: "HANDOFF_TYPED_SCHEMA_INVALID",
          detail: `typed handoff schema invalid: ${schemaErrors.join("; ")}`,
          handoff,
        };
      }
    }
  }

  return {
    checked: true,
    ok: true,
    handoff,
    typed_schema_checked: typed?.file ? getSchemaForTypedHandoff(typed.file)?.$id || "" : "",
  };
}
