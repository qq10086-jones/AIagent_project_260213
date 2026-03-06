import fs from "fs";
import path from "path";
import { validateQaVerifierArtifacts } from "../src/qa_verifier.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function writeText(absPath, text) {
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, text, "utf8");
}

function writeJson(absPath, value) {
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, JSON.stringify(value, null, 2), "utf8");
}

function main() {
  const fixture = JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), "canary_inputs", "qa_verifier_min.json"), "utf8")
  );
  const root = path.resolve(process.cwd(), "artifacts", "canary", "qa_verifier_fixture");

  writeText(
    path.join(root, "ok", "tests", "test_plan.md"),
    "# Test Plan\n\n## Verification Steps\n\n## Release Checklist\n"
  );
  writeText(
    path.join(root, "ok", "qa", "smoke_report.md"),
    "# Smoke Report\n\n## Executed Checks\n\n## Result Summary\n"
  );
  writeJson(
    path.join(root, "ok", "qa", "verification.json"),
    {
      verdict: "pass",
      acceptance_mapping: [
        { acceptance_id: "A1", status: "pass", evidence: "qa/smoke_report.md" }
      ],
      summary: "QA checks passed",
    }
  );

  writeText(
    path.join(root, "bad", "tests", "test_plan.md"),
    "# Test Plan\n"
  );
  writeText(
    path.join(root, "bad", "qa", "smoke_report.md"),
    "# Smoke Report\n"
  );
  writeJson(
    path.join(root, "bad", "qa", "verification.json"),
    { verdict: "pass" }
  );

  const ok = validateQaVerifierArtifacts({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/qa_verifier_fixture/ok",
  });
  const bad = validateQaVerifierArtifacts({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/qa_verifier_fixture/bad",
  });

  assertEqual(ok.ok, true, "ok.ok");
  assertEqual(ok.schema_checked, fixture.expected.schema_id, "ok.schema_checked");
  assertEqual(bad.ok, false, "bad.ok");
  assertEqual(bad.code, fixture.expected.invalid_code, "bad.code");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "qa_verifier");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "qa_verifier_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    valid_case: ok,
    invalid_case: bad,
  }, null, 2), "utf8");
  console.log("# QA Verifier Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
