function parseStepResult(step = {}) {
  if (step && typeof step.result_json === "object" && step.result_json !== null) {
    return step.result_json;
  }
  try {
    return JSON.parse(step?.result_json || "{}");
  } catch {
    return {};
  }
}

export function resolvePreviewResultFromSteps(steps = []) {
  for (const step of steps || []) {
    if (String(step?.step_id || "") !== "deploy_preview") continue;
    const result = parseStepResult(step);
    const previewUrl = String(
      result?.preview_url ||
      result?.result_url ||
      result?.deployment?.preview_url ||
      ""
    ).trim();
    const previewStatus = String(
      result?.preview_status ||
      result?.deployment_status ||
      result?.deployment?.status ||
      ""
    ).trim();
    const fallbackReason = String(result?.fallback_reason || result?.reason || "").trim();
    const deploymentResultPath = String(result?.deployment_result_path || "").trim();
    return {
      preview_url: previewUrl || null,
      preview_status: previewStatus || null,
      fallback_reason: fallbackReason || null,
      deployment_result_path: deploymentResultPath || null,
    };
  }
  return {
    preview_url: null,
    preview_status: null,
    fallback_reason: null,
    deployment_result_path: null,
  };
}

export function buildReleaseNotesUrl({ minio = null, minioBucket = "", run }) {
  if (!minioBucket || !run) return null;
  const endpoint = String(minio?.endpoint || "http://localhost:9001").replace(/\/+$/, "");
  const runId = String(run.run_id || run.workflow_run_id || "").trim();
  if (!runId) return null;
  return `${endpoint}/${minioBucket}/release/${runId}/release_notes.md`;
}

export function resolveWorkflowResultUrl({ steps = [], minio = null, minioBucket = "", run }) {
  const preview = resolvePreviewResultFromSteps(steps);
  return {
    ...preview,
    result_url: preview.preview_url || buildReleaseNotesUrl({ minio, minioBucket, run }),
  };
}
