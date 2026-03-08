import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "./schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "contracts");
const QA_REPORT_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_qa_report.schema.json"), "utf8")
);

function readJson(absPath) {
  try {
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

export function validateQaVerifierArtifacts({ workspaceRoot, artifactRoot }) {
  const relRoot = String(artifactRoot || "").trim().replace(/\\/g, "/");
  if (!relRoot) {
    return { checked: true, ok: false, code: "QA_ARTIFACT_ROOT_MISSING", detail: "artifact_root missing" };
  }
  const rootAbs = path.resolve(workspaceRoot, relRoot);
  const verificationPath = path.resolve(rootAbs, "verify", "qa_report.json");

  const missingFiles = [verificationPath].filter((item) => !fs.existsSync(item));
  if (missingFiles.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "QA_REQUIRED_FILES_MISSING",
      detail: `missing QA files: ${missingFiles.map((item) => path.basename(item)).join(", ")}`,
    };
  }

  const verificationJson = readJson(verificationPath);
  const schemaErrors = validateJsonSchemaLite(QA_REPORT_SCHEMA, verificationJson, "$");
  const errors = schemaErrors.map((item) => `qa_report.json:${item}`);

  if (errors.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "QA_VERIFICATION_INVALID",
      detail: `qa verifier contract failed: ${errors.join(", ")}`,
    };
  }

  return {
    checked: true,
    ok: true,
    schema_checked: QA_REPORT_SCHEMA.$id,
    files_checked: ["verify/qa_report.json"],
  };
}
