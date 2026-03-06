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

function getValueByPath(obj, dottedPath) {
  return String(dottedPath || "")
    .split(".")
    .filter(Boolean)
    .reduce((acc, key) => (acc && Object.prototype.hasOwnProperty.call(acc, key) ? acc[key] : undefined), obj);
}

function isPresentValue(value) {
  if (typeof value === "string") return value.trim().length > 0;
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value === "boolean") return true;
  if (Array.isArray(value)) return value.length > 0;
  return value && typeof value === "object" ? Object.keys(value).length > 0 : false;
}

function getSchemaForTypedHandoff(fileName) {
  const safe = String(fileName || "").replace(/\\/g, "/");
  if (safe.endsWith("pm_to_architect.json")) return PM_HANDOFF_SCHEMA;
  if (safe.endsWith("architect_to_impl.json")) return ARCH_HANDOFF_SCHEMA;
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
  const missingSections = (handoff.required_sections || []).filter((section) => !corpus.includes(String(section || "").toLowerCase()));
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
    const missingFields = (typed.required_fields || []).filter((field) => !isPresentValue(getValueByPath(manifest, field)));
    if (missingFields.length > 0) {
      return {
        checked: true,
        ok: false,
        code: "HANDOFF_TYPED_FIELDS_MISSING",
        detail: `missing typed handoff fields: ${missingFields.join(", ")}`,
        handoff,
      };
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
