import fs from "fs";
import path from "path";
import { buildDiscordCompletionReply } from "../src/vnext/discord_reply_adapter.js";
import { buildFinalResultPackage } from "../src/final_result_packager.js";
import { createTaskEnvelope } from "../src/vnext/task_envelope.js";
import { assertDispatchSuccessResponse } from "../src/vnext/contract_validator.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const pkg = buildFinalResultPackage({
    workflowRunId: "wf-123",
    runId: "run-123",
    status: "succeeded",
    summaryPath: "artifacts/release/run-123/summary/run_summary.md",
    manifestPath: "artifacts/release/run-123/meta/run_manifest.json",
    goNoGoVerdict: "GO",
    strictCanaryVerdict: "pass",
  });

  const env = createTaskEnvelope({
    task_id: "task-123",
    source: "discord",
    raw_input: "Build CRM MVP",
    intent: "coding",
    requires_orchestration: true,
    target_team: "coding_team"
  });

  const replyObj = buildDiscordCompletionReply({
    finalResultPackage: pkg,
    taskEnvelope: env,
  });

  // Must be a valid DispatchSuccessResponse
  assertDispatchSuccessResponse(replyObj);

  assertEqual(replyObj.ok, true, "replyObj.ok");
  assertEqual(replyObj.response_mode, "direct_reply", "replyObj.response_mode");
  assertEqual(replyObj.run_id, "run-123", "replyObj.run_id");

  const replyText = replyObj.reply;
  if (!replyText.includes("wf-123")) throw new Error("Reply missing workflowRunId");
  if (!replyText.includes("succeeded")) throw new Error("Reply missing status");
  if (!replyText.includes("artifacts/release/run-123/summary/run_summary.md")) throw new Error("Reply missing artifact path");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "discord_reply_adapter");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "discord_reply_adapter_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    replyObj,
  }, null, 2), "utf8");
  console.log("# Discord Reply Adapter Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
