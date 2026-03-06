import fs from "fs";
import path from "path";
import { formatTransitionNotification, formatFailureReport } from "../src/vnext/observability_reporter.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const startMsg = formatTransitionNotification({
    previousStep: null,
    nextStep: { id: "pm_spec", role: "pm_agent" }
  });
  if (!startMsg.includes("Starting workflow execution")) throw new Error("startMsg");

  const progressMsg = formatTransitionNotification({
    previousStep: { id: "pm_spec" },
    nextStep: { id: "arch_design", role: "architect_agent" }
  });
  if (!progressMsg.includes("Completed **pm_spec**")) throw new Error("progressMsg");

  const endMsg = formatTransitionNotification({
    previousStep: { id: "qa_verify" },
    nextStep: null
  });
  if (!endMsg.includes("Workflow completed successfully")) throw new Error("endMsg");

  const failMsg = formatFailureReport({
    step_id: "backend_impl",
    role: "backend_agent",
    error_code: "COMPILATION_ERROR",
    error_message: "Failed to compile TS code",
    raw_logs: "secret_token=abc123def456ghi789jkl012mno\nerror TS2322: Type 'string' is not assignable to type 'number'."
  });
  
  if (!failMsg.includes("backend_impl")) throw new Error("failMsg step");
  if (!failMsg.includes("COMPILATION_ERROR")) throw new Error("failMsg code");
  if (!failMsg.includes("TS2322")) throw new Error("failMsg log");
  if (failMsg.includes("abc123def456ghi789jkl012mno")) throw new Error("failMsg secret leak");
  if (!failMsg.includes("***REDACTED***")) throw new Error("failMsg redact missing");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "observability_reporter");
  fs.mkdirSync(outDir, { recursive: true });
  
  const reportPath = path.join(outDir, "observability_reporter_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    results: { startMsg, progressMsg, endMsg, failMsg }
  }, null, 2), "utf8");

  console.log("# Observability Reporter Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
