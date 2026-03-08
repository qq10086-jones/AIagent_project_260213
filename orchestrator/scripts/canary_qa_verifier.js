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

  writeJson(
    path.join(root, "ok", "verify", "qa_report.json"),
    {
      overall_status: "pass",
      checks: [
        { check_id: "det-1", layer: "deterministic", description: "artifacts exist", status: "pass", detail: "ok" },
        { check_id: "sem-1", layer: "semantic", description: "api consistency", status: "pass", detail: "ok" }
      ],
      verified_artifacts: ["impl/be_changes/server.js", "impl/fe_changes/app.js"]
    }
  );

  writeJson(
    path.join(root, "bad", "verify", "qa_report.json"),
    { overall_status: "pass" }
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
