import {
  getWorkflowRunById,
  listWorkflowCheckpoints,
  listWorkflowSteps,
} from "../data/workflow_repository.js";

/**
 * Provides timeline querying and task replay capabilities.
 * Pulls together task context, step metadata, and output artifacts to form a linear execution log.
 */

export async function queryWorkflowTimeline({ pool, workflowRunId }) {
  if (!pool || !workflowRunId) {
    throw new Error("pool and workflowRunId are required");
  }

  // 1. Get workflow run
  const run = await getWorkflowRunById(pool, workflowRunId);
  if (!run) {
    throw new Error(`Workflow run not found: ${workflowRunId}`);
  }

  // 2. Get steps ordered
  const steps = await listWorkflowSteps(pool, workflowRunId);

  // 3. Get checkpoints/artifacts for each step
  const checkpoints = await listWorkflowCheckpoints(pool, workflowRunId);

  const timeline = steps.map(step => {
    const cp = checkpoints.find(c => c.step_index === step.step_index);
    let outputData = null;
    try {
      if (step.result_json) outputData = JSON.parse(step.result_json);
    } catch(e) {}

    return {
      step_index: step.step_index,
      step_id: step.step_id,
      role: step.role_name,
      tool: step.tool_name,
      status: step.status,
      error_code: step.error_code,
      task_id: step.task_id,
      artifacts_recorded: !!cp,
      output_summary: outputData ? JSON.stringify(outputData).substring(0, 100) + (JSON.stringify(outputData).length > 100 ? "..." : "") : null
    };
  });

  return {
    workflow_run_id: workflowRunId,
    workflow_id: run.workflow_id,
    project_type: run.project_type,
    status: run.status,
    total_steps: steps.length,
    timeline,
  };
}

export function formatTimelineAsText(timelineObj) {
  let text = `=================================================\n`;
  text += `Workflow Replay: ${timelineObj.workflow_run_id}\n`;
  text += `Status: ${timelineObj.status} | Template: ${timelineObj.workflow_id}\n`;
  text += `=================================================\n\n`;

  for (const step of timelineObj.timeline) {
    text += `[Step ${step.step_index}] ${step.step_id} (${step.role})\n`;
    text += `  Status: ${step.status}\n`;
    if (step.error_code) text += `  Error:  ${step.error_code}\n`;
    text += `  Tool:   ${step.tool}\n`;
    text += `  Task:   ${step.task_id || "none"}\n`;
    text += `  Output: ${step.output_summary || "empty"}\n\n`;
  }

  text += `=================================================\n`;
  text += `End of Replay\n`;
  text += `=================================================\n`;

  return text;
}
