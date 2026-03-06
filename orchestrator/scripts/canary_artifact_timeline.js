import fs from "fs";
import path from "path";
import { queryWorkflowTimeline, formatTimelineAsText } from "../src/vnext/artifact_timeline.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function createMockPool() {
  return {
    async query(sql, params) {
      if (sql.includes("FROM workflow_runs")) {
        return {
          rows: [
            {
              workflow_run_id: "wf-123",
              workflow_id: "coding_team_v0",
              project_type: "webapp",
              status: "succeeded"
            }
          ]
        };
      }
      if (sql.includes("FROM workflow_steps")) {
        return {
          rows: [
            {
              step_index: 0,
              step_id: "pm_spec",
              role_name: "pm_agent",
              tool_name: "coding.delegate",
              status: "succeeded",
              task_id: "task-0",
              result_json: JSON.stringify({ artifacts: ["plan.md"] })
            },
            {
              step_index: 1,
              step_id: "arch_design",
              role_name: "architect_agent",
              tool_name: "coding.delegate",
              status: "failed",
              error_code: "ARCH_ERROR",
              task_id: "task-1",
              result_json: JSON.stringify({ error: "missing field" })
            }
          ]
        };
      }
      if (sql.includes("FROM workflow_checkpoints")) {
        return {
          rows: [
            { checkpoint_id: "cp-0", step_index: 0, step_id: "pm_spec", task_id: "task-0", workspace_hash: "hash0" }
          ]
        };
      }
      return { rows: [] };
    }
  };
}

async function main() {
  const pool = createMockPool();
  const timelineObj = await queryWorkflowTimeline({ pool, workflowRunId: "wf-123" });

  assertEqual(timelineObj.workflow_run_id, "wf-123", "workflow_run_id");
  assertEqual(timelineObj.total_steps, 2, "total_steps");
  assertEqual(timelineObj.timeline[0].step_id, "pm_spec", "step 0 id");
  assertEqual(timelineObj.timeline[0].artifacts_recorded, true, "step 0 artifacts");
  assertEqual(timelineObj.timeline[1].status, "failed", "step 1 status");
  assertEqual(timelineObj.timeline[1].artifacts_recorded, false, "step 1 artifacts");

  const formattedText = formatTimelineAsText(timelineObj);

  if (!formattedText.includes("Workflow Replay: wf-123")) throw new Error("Missing header");
  if (!formattedText.includes("[Step 0] pm_spec")) throw new Error("Missing step 0");
  if (!formattedText.includes("ARCH_ERROR")) throw new Error("Missing step 1 error");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "artifact_timeline");
  fs.mkdirSync(outDir, { recursive: true });
  
  const reportPath = path.join(outDir, "artifact_timeline_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    timelineObj,
    formattedText
  }, null, 2), "utf8");

  const textPath = path.join(outDir, "artifact_timeline_canary.txt");
  fs.writeFileSync(textPath, formattedText, "utf8");

  console.log("# Artifact Timeline Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main().catch(console.error);
