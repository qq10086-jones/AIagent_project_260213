import fs from "fs";
import path from "path";
import { buildArtifactMetadata, validateArtifactMetadata } from "../src/artifact_registry.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const fixture = JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), "canary_inputs", "artifact_model_min.json"), "utf8")
  );

  const metadata = buildArtifactMetadata({
    taskId: "workflow_run:wf-1",
    role: "pm",
    objectKey: "artifacts/release/run-1/plan/spec.md",
    mime: "text/markdown",
    createdAt: "2026-03-06T00:00:00.000Z",
    source: "release_pack_local",
  });
  const checked = validateArtifactMetadata(metadata);

  assertEqual(checked.ok, true, "checked.ok");
  assertEqual(checked.schema_id, fixture.expected.schema_id, "schema_id");
  assertEqual(metadata.type, fixture.expected.type, "metadata.type");
  assertEqual(metadata.mime, fixture.expected.mime, "metadata.mime");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "artifact_model");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "artifact_model_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    metadata,
    checked,
  }, null, 2), "utf8");
  console.log("# Artifact Model Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
