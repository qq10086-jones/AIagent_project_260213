import { runCodexTask } from "./adapters/codex_adapter.js";
import { runOpenCodeTask } from "./adapters/opencode_adapter.js";

const SUPPORTED_PROVIDERS = new Set(["auto", "opencode", "codex"]);

function normalizeProvider(provider) {
  return String(provider || "auto").toLowerCase();
}

function normalizeLaneRegistry(runtimeCoderConfig = {}) {
  const lanes = runtimeCoderConfig && typeof runtimeCoderConfig.execution_lanes === "object"
    ? runtimeCoderConfig.execution_lanes
    : {};
  return lanes && typeof lanes === "object" ? lanes : {};
}

function resolveExecutionLane({
  adapterRequest,
  provider,
  model,
  runtimeCoderConfig = {},
}) {
  const providerRequested = normalizeProvider(provider || adapterRequest?.provider || "auto");
  const payload = adapterRequest?.payload && typeof adapterRequest.payload === "object"
    ? adapterRequest.payload
    : {};
  const laneRegistry = normalizeLaneRegistry(runtimeCoderConfig);
  const requestedLane = String(payload.execution_lane || runtimeCoderConfig.execution_lane_default || "").trim();
  const laneConfig = requestedLane && laneRegistry[requestedLane] && typeof laneRegistry[requestedLane] === "object"
    ? laneRegistry[requestedLane]
    : null;
  const resolvedProvider = normalizeProvider(
    laneConfig?.provider || providerRequested || payload.provider_hint || "auto"
  );
  const resolvedModel = String(
    laneConfig?.model || model || payload.model_hint || ""
  ).trim() || null;

  return {
    executionLane: requestedLane || null,
    laneSource: requestedLane ? (laneConfig ? "runtime_config" : "requested_unresolved") : "direct_provider",
    laneConfig,
    providerRequested,
    providerResolved: resolvedProvider,
    modelResolved: resolvedModel,
  };
}

function buildAdapterResult({
  adapterRequest,
  resolved,
  providerUsed,
  rawResult,
  fallbackFrom = null,
  fallbackTarget = null,
  fallbackTaken = false,
}) {
  const providerResolved = String(resolved?.providerResolved || resolved?.providerRequested || "");
  const diagnostics = {
    ...(rawResult?.diagnostics || {}),
    adapter_type: String(adapterRequest?.adapter_type || "coding_executor"),
    task_type: String(adapterRequest?.task_type || "coding_execution"),
    execution_lane: resolved?.executionLane || null,
    execution_lane_source: resolved?.laneSource || null,
    lane_provider_resolved: providerResolved,
    lane_model_resolved: resolved?.modelResolved || null,
    model_provider: String(providerUsed || providerResolved || ""),
    model_name: rawResult?.model_used || resolved?.modelResolved || null,
    provider_class: rawResult?.diagnostics?.provider_class || null,
    provider_error_class: rawResult?.diagnostics?.provider_error_class || null,
    fallback_from: fallbackFrom,
    fallback_target: fallbackTarget,
    fallback_taken: Boolean(fallbackTaken),
  };

  return {
    ok: Boolean(rawResult?.ok),
    provider_used: String(providerUsed || providerResolved || ""),
    provider_requested: String(resolved?.providerRequested || ""),
    model_used: rawResult?.model_used || resolved?.modelResolved || null,
    command_used: rawResult?.command_used || null,
    command_source: rawResult?.command_source || "unknown",
    execution_lane: resolved?.executionLane || null,
    stdout: rawResult?.stdout || "",
    stderr: rawResult?.stderr || "",
    diagnostics,
    error: rawResult?.error || null,
  };
}

async function runProvider({
  providerName,
  adapterRequest,
  model,
  maxRuntimeS,
  codexCommand,
  opencodeCommand,
  workspaceRoot,
  artifactWorkspaceRoot = null,
  onRuntimeEvent = null,
}) {
  const taskPrompt = String(adapterRequest?.payload?.task_prompt || "").trim();
  if (providerName === "opencode") {
    return runOpenCodeTask({
      workspaceRoot,
      artifactWorkspaceRoot,
      taskPrompt,
      model,
      maxRuntimeS,
      opencodeCommand,
      onRuntimeEvent,
    });
  }
  return runCodexTask({
    workspaceRoot,
    artifactWorkspaceRoot,
    taskPrompt,
    model,
    maxRuntimeS,
    codexCommand,
    onRuntimeEvent,
  });
}

export async function executeCodingAdapter({
  workspaceRoot,
  artifactWorkspaceRoot = null,
  adapterRequest,
  provider = "auto",
  model = null,
  maxRuntimeS = 600,
  codexCommand = null,
  opencodeCommand = null,
  allowProviderFallback = false,
  runtimeCoderConfig = {},
  onRuntimeEvent = null,
}) {
  const resolved = resolveExecutionLane({
    adapterRequest,
    provider,
    model,
    runtimeCoderConfig,
  });

  if (!SUPPORTED_PROVIDERS.has(resolved.providerRequested)) {
    return {
      ok: false,
      provider_used: resolved.providerRequested,
      provider_requested: resolved.providerRequested,
      model_used: resolved.modelResolved || null,
      command_used: null,
      command_source: "unsupported",
      execution_lane: resolved.executionLane,
      stdout: "",
      stderr: "",
      diagnostics: {
        error_code: "E_PROVIDER_UNAVAILABLE",
        provider_error_class: "PROVIDER_CONFIG_ERROR",
        adapter_type: String(adapterRequest?.adapter_type || "coding_executor"),
        task_type: String(adapterRequest?.task_type || "coding_execution"),
        execution_lane: resolved.executionLane,
      },
      error: `Unsupported provider '${resolved.providerRequested}'. Use auto/opencode/codex.`,
    };
  }

  const effectiveProvider = resolved.providerResolved === "auto" ? "opencode" : resolved.providerResolved;
  let rawResult = await runProvider({
    providerName: effectiveProvider,
    adapterRequest,
    model: resolved.modelResolved,
    maxRuntimeS,
    codexCommand,
    opencodeCommand,
    workspaceRoot,
    artifactWorkspaceRoot,
    onRuntimeEvent,
  });
  let providerUsed = effectiveProvider;
  let fallbackFrom = null;
  let fallbackTarget = null;
  let fallbackTaken = false;

  if (
    allowProviderFallback &&
    effectiveProvider === "opencode" &&
    String(rawResult?.diagnostics?.error_code || "") === "E_PROVIDER_UNAVAILABLE"
  ) {
    fallbackFrom = "opencode";
    fallbackTarget = "codex";
    rawResult = await runProvider({
      providerName: "codex",
      adapterRequest,
      model: resolved.modelResolved,
      maxRuntimeS,
      codexCommand,
      opencodeCommand,
      workspaceRoot,
      artifactWorkspaceRoot,
      onRuntimeEvent,
    });
    providerUsed = "codex";
    fallbackTaken = true;
  }

  return buildAdapterResult({
    adapterRequest,
    resolved,
    providerUsed,
    rawResult,
    fallbackFrom,
    fallbackTarget,
    fallbackTaken,
  });
}

