import { runCodexTask } from "./adapters/codex_adapter.js";
import { runOpenCodeTask } from "./adapters/opencode_adapter.js";
import { runQwenTask } from "./adapters/qwen_adapter.js";

const SUPPORTED_PROVIDERS = new Set(["auto", "opencode", "codex", "qwen"]);

function normalizeProvider(provider) {
  return String(provider || "auto").toLowerCase();
}

function buildAdapterResult({ adapterRequest, providerUsed, providerRequested, rawResult, fallbackFrom = null }) {
  return {
    ok: Boolean(rawResult?.ok),
    provider_used: String(providerUsed || providerRequested || ""),
    provider_requested: String(providerRequested || ""),
    model_used: rawResult?.model_used || null,
    command_used: rawResult?.command_used || null,
    command_source: rawResult?.command_source || "unknown",
    stdout: rawResult?.stdout || "",
    stderr: rawResult?.stderr || "",
    diagnostics: {
      ...(rawResult?.diagnostics || {}),
      adapter_type: String(adapterRequest?.adapter_type || "coding_executor"),
      task_type: String(adapterRequest?.task_type || "coding_execution"),
      fallback_from: fallbackFrom,
    },
    error: rawResult?.error || null,
  };
}

async function runProvider({ providerName, adapterRequest, model, maxRuntimeS, codexCommand, opencodeCommand, workspaceRoot }) {
  const taskPrompt = String(adapterRequest?.payload?.task_prompt || "").trim();
  if (providerName === "opencode") {
    return runOpenCodeTask({
      workspaceRoot,
      taskPrompt,
      model,
      maxRuntimeS,
      opencodeCommand,
    });
  }
  if (providerName === "qwen") {
    return runQwenTask({
      taskPrompt,
      model,
      maxRuntimeS,
    });
  }
  return runCodexTask({
    workspaceRoot,
    taskPrompt,
    model,
    maxRuntimeS,
    codexCommand,
  });
}

export async function executeCodingAdapter({
  workspaceRoot,
  adapterRequest,
  provider = "auto",
  model = null,
  maxRuntimeS = 600,
  codexCommand = null,
  opencodeCommand = null,
}) {
  const providerRequested = normalizeProvider(provider || adapterRequest?.provider || "auto");
  if (!SUPPORTED_PROVIDERS.has(providerRequested)) {
    return {
      ok: false,
      provider_used: providerRequested,
      provider_requested: providerRequested,
      model_used: model || null,
      command_used: null,
      command_source: "unsupported",
      stdout: "",
      stderr: "",
      diagnostics: {
        error_code: "E_PROVIDER_UNAVAILABLE",
        adapter_type: String(adapterRequest?.adapter_type || "coding_executor"),
        task_type: String(adapterRequest?.task_type || "coding_execution"),
      },
      error: `Unsupported provider '${providerRequested}'. Use auto/opencode/codex/qwen.`,
    };
  }

  const preferredProvider = providerRequested === "auto" ? "opencode" : providerRequested;
  let rawResult = await runProvider({
    providerName: preferredProvider,
    adapterRequest,
    model,
    maxRuntimeS,
    codexCommand,
    opencodeCommand,
    workspaceRoot,
  });
  let providerUsed = preferredProvider;
  let fallbackFrom = null;

  if (
    preferredProvider === "opencode" &&
    String(rawResult?.diagnostics?.error_code || "") === "E_PROVIDER_UNAVAILABLE"
  ) {
    fallbackFrom = "opencode";
    rawResult = await runProvider({
      providerName: "codex",
      adapterRequest,
      model,
      maxRuntimeS,
      codexCommand,
      opencodeCommand,
      workspaceRoot,
    });
    providerUsed = "codex";
  }

  return buildAdapterResult({
    adapterRequest,
    providerUsed,
    providerRequested,
    rawResult,
    fallbackFrom,
  });
}
