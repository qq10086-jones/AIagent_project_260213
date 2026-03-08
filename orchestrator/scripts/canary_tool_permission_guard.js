import fs from "fs";
import path from "path";
import { validateToolPermission } from "../src/vnext/tool_permission_guard.js";
import { resolveOrchestratorArtifactPath } from "./_paths.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const pmDelegate = validateToolPermission("pm", "coding.delegate");
  assertEqual(pmDelegate.allowed, true, "pm delegate");

  const pmDocs = validateToolPermission("pm_agent", "document.read");
  assertEqual(pmDocs.allowed, true, "pm read docs");

  const pmCode = validateToolPermission("pm_agent", "bash.execute");
  assertEqual(pmCode.allowed, false, "pm run bash");

  const feBash = validateToolPermission("frontend_agent", "bash.execute");
  assertEqual(feBash.allowed, true, "fe run bash");

  const unknownAgent = validateToolPermission("rogue_agent", "bash.execute");
  assertEqual(unknownAgent.allowed, false, "unknown agent tool");

  const outDir = resolveOrchestratorArtifactPath("canary", "tool_permission_guard");
  fs.mkdirSync(outDir, { recursive: true });
  
  const reportPath = path.join(outDir, "tool_permission_guard_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    results: { pmDelegate, pmDocs, pmCode, feBash, unknownAgent }
  }, null, 2), "utf8");

  console.log("# Tool Permission Guard Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
