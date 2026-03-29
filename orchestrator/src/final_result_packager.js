import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "./schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "contracts");
const FINAL_RESULT_PACKAGE_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "final_result_package.schema.json"), "utf8")
);

function fileMime(filePath) {
  const lower = String(filePath || "").toLowerCase();
  if (lower.endsWith(".json")) return "application/json";
  if (lower.endsWith(".md")) return "text/markdown";
  if (lower.endsWith(".html")) return "text/html";
  return "application/octet-stream";
}

function labelForPath(filePath) {
  const lower = String(filePath || "").toLowerCase();
  if (lower.endsWith("run_manifest.json")) return "run_manifest";
  if (lower.endsWith("run_summary.md")) return "run_summary";
  if (lower.endsWith("go_no_go_result.json")) return "go_no_go_result";
  if (lower.endsWith("strict_canary_report.md")) return "strict_canary_report";
  if (lower.endsWith("strict_canary_report.json")) return "strict_canary_json";
  if (lower.endsWith("preview_validation_report.json")) return "preview_validation_report";
  if (lower.endsWith("product_fidelity_report.json")) return "product_fidelity_report";
  return path.basename(filePath || "artifact");
}

export function buildFinalResultPackage({
  workflowRunId,
  runId,
  status,
  summaryPath,
  manifestPath,
  goNoGoResultPath = "",
  strictCanaryReportPath = "",
  strictCanaryJsonPath = "",
  previewValidationReportPath = "",
  productFidelityReportPath = "",
  goNoGoVerdict = "",
  strictCanaryVerdict = "",
}) {
  const artifactPaths = [
    manifestPath,
    summaryPath,
    goNoGoResultPath,
    strictCanaryReportPath,
    strictCanaryJsonPath,
    previewValidationReportPath,
    productFidelityReportPath,
  ].filter(Boolean);
  const artifacts = artifactPaths.map((item) => ({
    label: labelForPath(item),
    path: String(item).replace(/\\/g, "/"),
    mime: fileMime(item),
  }));
  const summary = [
    `run_id=${runId}`,
    `workflow_run_id=${workflowRunId}`,
    `status=${status}`,
    goNoGoVerdict ? `go_no_go=${goNoGoVerdict}` : "",
    strictCanaryVerdict ? `strict_canary=${strictCanaryVerdict}` : "",
  ].filter(Boolean).join(" | ");

  return {
    workflow_run_id: String(workflowRunId || ""),
    run_id: String(runId || ""),
    status: String(status || ""),
    summary,
    artifacts,
    go_no_go_verdict: String(goNoGoVerdict || ""),
    strict_canary_verdict: String(strictCanaryVerdict || ""),
  };
}

export function validateFinalResultPackage(pkg) {
  const errors = validateJsonSchemaLite(FINAL_RESULT_PACKAGE_SCHEMA, pkg, "$");
  return {
    ok: errors.length === 0,
    errors,
    schema_id: FINAL_RESULT_PACKAGE_SCHEMA.$id,
  };
}
