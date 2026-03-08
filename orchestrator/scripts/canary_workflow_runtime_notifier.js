import fs from "fs";
import path from "path";
import { buildWorkflowRuntimeNotification } from "../src/vnext/workflow_runtime_notifier.js";
import { resolveOrchestratorArtifactPath } from "./_paths.js";

function main() {
  const transition = buildWorkflowRuntimeNotification({
    workflowState: {
      steps: [
        { step_index: 0, step_id: "pm_spec", role_name: "pm", status: "succeeded" },
        { step_index: 1, step_id: "arch_design", role_name: "architect", status: "running" },
      ],
    },
    workflowTerminal: { step_index: 0, step_id: "pm_spec" },
    status: "succeeded",
    normalizedErrorCode: "",
    streamError: "",
    output: {},
  });
  if (transition.kind !== "transition") throw new Error("transition kind");
  if (!transition.message.includes("pm_spec")) throw new Error("transition previous step missing");
  if (!transition.message.includes("arch_design")) throw new Error("transition next step missing");

  const failure = buildWorkflowRuntimeNotification({
    workflowState: {
      steps: [
        { step_index: 0, step_id: "pm_spec", role_name: "pm", status: "failed" },
      ],
    },
    workflowTerminal: { step_index: 0, step_id: "pm_spec" },
    status: "failed",
    normalizedErrorCode: "STEP_FAILED",
    streamError: "compile failed",
    output: {
      stderr: "secret_token=abcdefghijklmnopqrstuvwx123456\nerror TS1005",
    },
  });
  if (failure.kind !== "failure") throw new Error("failure kind");
  if (!failure.message.includes("STEP_FAILED")) throw new Error("failure code missing");
  if (!failure.message.includes("***REDACTED***")) throw new Error("failure redaction missing");

  const outDir = resolveOrchestratorArtifactPath("canary", "workflow_runtime_notifier");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "workflow_runtime_notifier_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    transition,
    failure,
  }, null, 2), "utf8");

  console.log("# Workflow Runtime Notifier Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
