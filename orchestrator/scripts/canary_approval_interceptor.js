import fs from "fs";
import path from "path";
import { interceptAndDispatch } from "../src/vnext/approval_interceptor.js";
import { createTaskEnvelope } from "../src/vnext/task_envelope.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const safeEnv = createTaskEnvelope({
    task_id: "t1",
    intent: "chat",
    raw_input: "Hello",
    target_team: "brain"
  });

  const safeResult = interceptAndDispatch({
    taskEnvelope: safeEnv,
    executionPlan: { tool_name: "none" },
    runId: "run1",
    taskId: "t1"
  });

  assertEqual(safeResult.response_mode, "progress_update", "safe mode");
  assertEqual(safeResult.execution.waiting_approval, false, "safe wait");

  const riskyEnv = createTaskEnvelope({
    task_id: "t2",
    intent: "coding",
    raw_input: "rm -rf /",
    target_team: "coding_team"
  });

  const riskyResult = interceptAndDispatch({
    taskEnvelope: riskyEnv,
    executionPlan: { tool_name: "bash.execute" },
    runId: "run2",
    taskId: "t2"
  });

  assertEqual(riskyResult.response_mode, "approval_request", "risky mode");
  assertEqual(riskyResult.execution.waiting_approval, true, "risky wait");

  const approvedEnv = createTaskEnvelope({
    task_id: "t3",
    intent: "coding",
    raw_input: "rm -rf /",
    target_team: "coding_team",
    constraints: { approved: true } // Pretend user already clicked yes
  });

  const approvedResult = interceptAndDispatch({
    taskEnvelope: approvedEnv,
    executionPlan: { tool_name: "bash.execute" },
    runId: "run3",
    taskId: "t3"
  });

  assertEqual(approvedResult.response_mode, "progress_update", "approved mode");
  assertEqual(approvedResult.execution.waiting_approval, false, "approved wait");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "approval_interceptor");
  fs.mkdirSync(outDir, { recursive: true });
  
  const reportPath = path.join(outDir, "approval_interceptor_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    results: { safeResult, riskyResult, approvedResult }
  }, null, 2), "utf8");

  console.log("# Approval Interceptor Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
