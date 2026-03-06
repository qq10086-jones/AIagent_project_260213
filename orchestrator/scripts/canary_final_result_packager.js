import fs from "fs";
import path from "path";
import { buildFinalResultPackage, validateFinalResultPackage } from "../src/final_result_packager.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const fixture = JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), "canary_inputs", "final_result_packager_min.json"), "utf8")
  );

  const pkg = buildFinalResultPackage({
    workflowRunId: "wf-1",
    runId: "run-1",
    status: "succeeded",
    summaryPath: "artifacts/release/run-1/summary/run_summary.md",
    manifestPath: "artifacts/release/run-1/meta/run_manifest.json",
    goNoGoResultPath: "artifacts/release/run-1/qa/go_no_go_result.json",
    strictCanaryReportPath: "artifacts/release/run-1/qa/strict_canary_report.md",
    strictCanaryJsonPath: "artifacts/release/run-1/qa/strict_canary_report.json",
    goNoGoVerdict: "GO",
    strictCanaryVerdict: "pass",
  });
  const checked = validateFinalResultPackage(pkg);

  assertEqual(checked.ok, true, "checked.ok");
  assertEqual(checked.schema_id, fixture.expected.schema_id, "schema_id");
  assertEqual(pkg.artifacts.length, fixture.expected.artifact_count, "artifact_count");
  assertEqual(pkg.status, fixture.expected.status, "status");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "final_result_packager");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "final_result_packager_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    package: pkg,
    checked,
  }, null, 2), "utf8");
  console.log("# Final Result Packager Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
