import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { callLocalOllamaChat, callQwenChat } from "./local_llm_client.js";
import { assertConfigPreflight } from "../config_preflight.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const LOCAL_PROVIDERS_PATH = path.resolve(MODULE_DIR, "..", "..", "configs", "llm_providers.json");
const ROOT_PROVIDERS_PATH = path.resolve(MODULE_DIR, "..", "..", "..", "configs", "llm_providers.json");
const LOCAL_ROLE_POLICY_PATH = path.resolve(MODULE_DIR, "..", "..", "configs", "llm_role_policy.json");
const ROOT_ROLE_POLICY_PATH = path.resolve(MODULE_DIR, "..", "..", "..", "configs", "llm_role_policy.json");

const PROVIDERS_PATH = fs.existsSync(LOCAL_PROVIDERS_PATH) ? LOCAL_PROVIDERS_PATH : ROOT_PROVIDERS_PATH;
const ROLE_POLICY_PATH = fs.existsSync(LOCAL_ROLE_POLICY_PATH) ? LOCAL_ROLE_POLICY_PATH : ROOT_ROLE_POLICY_PATH;

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTransportError(err) {
  const message = String(err?.message || err || "").toLowerCase();
  return err?.name === "AbortError"
    || /timeout|timed out|fetch failed|network|socket|econn|enotfound|eai_again|429|500|502|503|504/.test(message);
}

function isProviderFallbackError(err) {
  const message = String(err?.message || err || "").toLowerCase();
  return /out of memory|oom|context overflow|context length|maximum context|token limit|input too long/.test(message);
}

function makeTypedError({ code, role, provider, model, fallbackAttempted, cause }) {
  const err = new Error(`${code}: role='${role}' provider='${provider}' model='${model}'`);
  err.code = code;
  err.role = role;
  err.provider = provider;
  err.model = model;
  err.fallback_attempted = Boolean(fallbackAttempted);
  err.cause = cause;
  return err;
}

export function createLlmDispatcher({
  providers = null,
  rolePolicy = null,
  qwenClient = callQwenChat,
  ollamaClient = callLocalOllamaChat,
  fetchImpl = globalThis.fetch,
  logger = console,
  sleep = delay,
} = {}) {
  if (providers == null || rolePolicy == null) {
    assertConfigPreflight({
      runtimeConfigPath: process.env.RUNTIME_CONFIG_PATH || "",
      workspaceRoot: path.resolve(MODULE_DIR, "..", ".."),
    });
  }
  const resolvedProviders = providers ?? readJson(PROVIDERS_PATH);
  const resolvedRolePolicy = rolePolicy ?? readJson(ROLE_POLICY_PATH);

  function resolveRole(role, overrides = {}) {
    const roleName = String(role || "").trim();
    const policy = resolvedRolePolicy?.roles?.[roleName];
    if (!policy) {
      throw makeTypedError({
        code: "LLM_ROLE_UNKNOWN",
        role: roleName || "unknown",
        provider: "unknown",
        model: "unknown",
        fallbackAttempted: false,
        cause: new Error("role not found in llm_role_policy.json"),
      });
    }
    const provider = String(overrides.provider || policy.provider || "").trim();
    const model = String(overrides.model || policy.model || "").trim();
    const secondaryModel = String(overrides.secondary_model || policy.secondary_model || "").trim();
    const providerSpec = resolvedProviders?.[provider];
    if (!providerSpec) {
      throw makeTypedError({
        code: "LLM_PROVIDER_UNKNOWN",
        role: roleName,
        provider,
        model,
        fallbackAttempted: false,
        cause: new Error("provider not found in llm_providers.json"),
      });
    }
    return { roleName, provider, providerSpec, model, secondaryModel };
  }

  async function invokeProvider(providerSpec, provider, model, messages) {
    const timeoutMs = Number(providerSpec?.timeout_ms || 30000);
    const endpointEnv = String(providerSpec?.endpoint_env || "").trim();
    const endpoint = endpointEnv ? process.env[endpointEnv] : undefined;
    if (providerSpec?.type === "cloud_api") {
      return qwenClient(messages, { qwenBase: endpoint, qwenModel: model });
    }
    if (providerSpec?.type === "local_ollama") {
      return ollamaClient(model, messages, timeoutMs);
    }
    throw new Error(`unsupported provider type '${providerSpec?.type || "unknown"}'`);
  }

  async function runWithRetry({ roleName, provider, providerSpec, model, messages }) {
    const retryPolicy = resolvedRolePolicy?.retry_policy || {};
    const retries = Math.max(0, Number(retryPolicy.retries || 0));
    const baseDelayMs = Math.max(1, Number(retryPolicy.base_delay_ms || 2000));
    let lastError = null;

    for (let attempt = 0; attempt <= retries; attempt += 1) {
      const startedAt = Date.now();
      try {
        const content = await invokeProvider(providerSpec, provider, model, messages);
        logger.info?.(
          `[llm_dispatcher] role=${roleName} provider=${provider} model=${model} latency=${Date.now() - startedAt}ms status=ok retry=${attempt}`
        );
        return { content, latency_ms: Date.now() - startedAt, retry_count: attempt };
      } catch (err) {
        lastError = err;
        if (!isTransportError(err) || attempt === retries) break;
        logger.warn?.(
          `[llm_dispatcher] role=${roleName} provider=${provider} model=${model} status=retry retry=${attempt + 1}`
        );
        await sleep(baseDelayMs * (2 ** attempt));
      }
    }

    throw lastError;
  }

  async function dispatch(role, messages, overrides = {}) {
    const { roleName, provider, providerSpec, model, secondaryModel } = resolveRole(role, overrides);
    try {
      const primary = await runWithRetry({ roleName, provider, providerSpec, model, messages });
      return { content: primary.content, model, provider, latency_ms: primary.latency_ms, used_fallback: false };
    } catch (err) {
      const shouldFallback = Boolean(secondaryModel)
        && (isProviderFallbackError(err) || isTransportError(err))
        && String(resolvedRolePolicy?.fallback_policy || "") === "model_fallback";
      if (!shouldFallback) {
        throw makeTypedError({
          code: "LLM_DISPATCH_FAILED",
          role: roleName,
          provider,
          model,
          fallbackAttempted: false,
          cause: err,
        });
      }
      const fallbackStatus = isProviderFallbackError(err) ? "oom_fallback" : "timeout_fallback";
      logger.warn?.(
        `[llm_dispatcher] role=${roleName} provider=${provider} model=${model} status=${fallbackStatus} secondary=${secondaryModel}`
      );
      try {
        const secondary = await runWithRetry({
          roleName,
          provider,
          providerSpec,
          model: secondaryModel,
          messages,
        });
        return {
          content: secondary.content,
          model: secondaryModel,
          provider,
          latency_ms: secondary.latency_ms,
          used_fallback: true,
        };
      } catch (fallbackErr) {
        throw makeTypedError({
          code: "LLM_DISPATCH_FAILED",
          role: roleName,
          provider,
          model: secondaryModel,
          fallbackAttempted: true,
          cause: fallbackErr,
        });
      }
    }
  }

  async function validateProviders() {
    const results = [];
    for (const [provider, providerSpec] of Object.entries(resolvedProviders || {})) {
      try {
        if (providerSpec?.type === "cloud_api") {
          const authEnv = String(providerSpec.auth_env || "").trim();
          const endpointEnv = String(providerSpec.endpoint_env || "").trim();
          if (!process.env[authEnv]) throw new Error(`missing env ${authEnv}`);
          if (endpointEnv && !process.env[endpointEnv]) throw new Error(`missing env ${endpointEnv}`);
          results.push({ provider, status: "ok", detail: "env configured" });
          continue;
        }

        if (providerSpec?.type === "local_ollama") {
          if (typeof fetchImpl !== "function") throw new Error("fetch unavailable");
          const endpointEnv = String(providerSpec.endpoint_env || "").trim();
          const base = String(process.env[endpointEnv] || "http://host.docker.internal:11434").replace(/\/+$/, "");
          const response = await fetchImpl(`${base}/api/tags`);
          if (!response.ok) throw new Error(`OLLAMA_TAGS_HTTP_${response.status}`);
          const payload = await response.json();
          const available = new Set((payload?.models || []).map((item) => String(item?.name || "").trim()).filter(Boolean));
          const required = [];
          for (const spec of Object.values(resolvedRolePolicy?.roles || {})) {
            if (String(spec?.provider || "") !== provider) continue;
            required.push(String(spec?.model || "").trim());
            if (spec?.secondary_model) required.push(String(spec.secondary_model).trim());
          }
          const missing = [...new Set(required.filter(Boolean))].filter((name) => !available.has(name));
          if (missing.length > 0) throw new Error(`missing models: ${missing.join(", ")}`);
          results.push({ provider, status: "ok", detail: `models=${required.length}` });
          continue;
        }

        results.push({ provider, status: "degraded", detail: `unsupported provider type '${providerSpec?.type || "unknown"}'` });
      } catch (err) {
        results.push({ provider, status: "degraded", detail: err.message || String(err) });
      }
    }
    return { ok: results.every((item) => item.status === "ok"), results };
  }

  return { dispatch, validateProviders };
}

const defaultDispatcher = createLlmDispatcher();

export const dispatch = defaultDispatcher.dispatch;
export const validateProviders = defaultDispatcher.validateProviders;
