import {
  validateToolAdapterRequest,
  validateToolAdapterResult,
} from "./tool_adapter_registry.js";

export function buildCodingExecutorRequest({
  provider,
  payload,
  executionPacket = null,
  role = "",
  stepId = "",
}) {
  const request = {
    adapter_type: "coding_executor",
    provider: String(provider || "opencode"),
    task_type: "coding_execution",
    payload: {
      role: String(role || payload?.role || ""),
      step_id: String(stepId || payload?.step_id || ""),
      task_prompt: String(payload?.task_prompt || payload?.prompt || ""),
      artifact_root: String(payload?.artifact_root || ""),
      expected_artifacts: Array.isArray(payload?.expected_artifacts) ? payload.expected_artifacts : [],
      target_paths: Array.isArray(payload?.target_paths) ? payload.target_paths : [],
      execution_adapter_packet: executionPacket || payload?.execution_adapter_packet || null,
      provider_hint: String(payload?.provider || provider || ""),
      model_hint: String(payload?.model || ""),
    },
    context: {
      workflow_id: String(payload?.workflow_id || ""),
      workflow_run_id: String(payload?.workflow_run_id || ""),
      run_id: String(payload?.run_id || ""),
    },
  };
  return request;
}

export function buildCodingExecutorResult({ provider, adapterResult }) {
  const result = {
    ok: Boolean(adapterResult?.ok),
    adapter_type: "coding_executor",
    provider: String(provider || adapterResult?.provider_used || ""),
    result: {
      provider_used: String(adapterResult?.provider_used || provider || ""),
      model_used: adapterResult?.model_used || null,
      command_used: adapterResult?.command_used || null,
      command_source: adapterResult?.command_source || null,
      diagnostics: adapterResult?.diagnostics || {},
      files_changed: Array.isArray(adapterResult?.files_changed) ? adapterResult.files_changed : [],
      diff_stats: adapterResult?.diff_stats || null,
      artifacts: adapterResult?.artifacts || {},
    },
  };
  if (adapterResult?.error) {
    result.error = String(adapterResult.error);
  }
  return result;
}

export function validateCodingExecutorRequest(request) {
  return validateToolAdapterRequest(request);
}

export function validateCodingExecutorResult(result) {
  return validateToolAdapterResult(result);
}
