import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "./schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "contracts");
const QA_VERIFICATION_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "qa_verification.schema.json"), "utf8")
);

const TEST_PLAN_HEADINGS = ["test plan", "verification steps", "release checklist"];
const SMOKE_HEADINGS = ["smoke report", "executed checks", "result summary"];

function readText(absPath) {
  try {
    return fs.readFileSync(absPath, "utf8");
  } catch {
    return "";
  }
}

function readJson(absPath) {
  try {
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

function extractHeadings(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^#{1,6}\s+/.test(line))
    .map((line) => line.replace(/^#{1,6}\s+/, "").trim().toLowerCase());
}

function findMissingHeadings(text, expected = []) {
  const headings = extractHeadings(text);
  return expected.filter((item) => !headings.some((heading) => heading.includes(item)));
}

export function validateQaVerifierArtifacts({ workspaceRoot, artifactRoot }) {
  const relRoot = String(artifactRoot || "").trim().replace(/\\/g, "/");
  if (!relRoot) {
    return { checked: true, ok: false, code: "QA_ARTIFACT_ROOT_MISSING", detail: "artifact_root missing" };
  }
  const rootAbs = path.resolve(workspaceRoot, relRoot);
  const testPlanPath = path.resolve(rootAbs, "tests/test_plan.md");
  const smokePath = path.resolve(rootAbs, "qa/smoke_report.md");
  const verificationPath = path.resolve(rootAbs, "qa/verification.json");

  const missingFiles = [testPlanPath, smokePath, verificationPath].filter((item) => !fs.existsSync(item));
  if (missingFiles.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "QA_REQUIRED_FILES_MISSING",
      detail: `missing QA files: ${missingFiles.map((item) => path.basename(item)).join(", ")}`,
    };
  }

  const missingHeadings = [
    ...findMissingHeadings(readText(testPlanPath), TEST_PLAN_HEADINGS).map((item) => `test_plan:${item}`),
    ...findMissingHeadings(readText(smokePath), SMOKE_HEADINGS).map((item) => `smoke_report:${item}`),
  ];
  const verificationJson = readJson(verificationPath);
  const schemaErrors = validateJsonSchemaLite(QA_VERIFICATION_SCHEMA, verificationJson, "$");
  const errors = [
    ...missingHeadings,
    ...schemaErrors.map((item) => `verification.json:${item}`),
  ];

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
    schema_checked: QA_VERIFICATION_SCHEMA.$id,
    files_checked: ["tests/test_plan.md", "qa/smoke_report.md", "qa/verification.json"],
  };
}
