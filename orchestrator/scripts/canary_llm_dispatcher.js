import fs from "fs";
import path from "path";

import { createLlmDispatcher } from "../src/vnext/llm_dispatcher.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function main() {
  const providers = {
    cloud_qwen: {
      type: "cloud_api",
      endpoint_env: "QWEN_BASE_URL",
      auth_env: "QWEN_API_KEY",
      timeout_ms: 30000,
    },
    local_ollama: {
      type: "local_ollama",
      endpoint_env: "OLLAMA_BASE_URL",
      timeout_ms: 240000,
    },
  };
  const rolePolicy = {
    roles: {
      pm: { provider: "cloud_qwen", model: "qwen-max" },
      backend: { provider: "local_ollama", model: "deepseek-r1:32b", secondary_model: "qwen2.5-coder:7b" },
      unknown_provider_role: { provider: "missing_provider", model: "bad-model" },
    },
    retry_policy: { strategy: "exponential_backoff", retries: 1, base_delay_ms: 1 },
    fallback_policy: "model_fallback",
  };

  process.env.QWEN_API_KEY = "stub-key";
  process.env.QWEN_BASE_URL = "https://qwen.stub";
  process.env.OLLAMA_BASE_URL = "http://ollama.stub";

  const dispatcher = createLlmDispatcher({
    providers,
    rolePolicy,
    qwenClient: async () => "cloud-ok",
    ollamaClient: async (model) => `local-ok:${model}`,
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return {
          models: [
            { name: "deepseek-r1:32b" },
            { name: "qwen2.5-coder:7b" },
          ],
        };
      },
    }),
    sleep: async () => {},
  });

  const cloudResult = await dispatcher.dispatch("pm", [{ role: "user", content: "plan" }]);
  assert(cloudResult.provider === "cloud_qwen", "cloud dispatch provider mismatch");

  const localResult = await dispatcher.dispatch("backend", [{ role: "user", content: "code" }]);
  assert(localResult.provider === "local_ollama", "local dispatch provider mismatch");

  const overrideResult = await dispatcher.dispatch(
    "backend",
    [{ role: "user", content: "code" }],
    { model: "qwen2.5-coder:7b" }
  );
  assert(overrideResult.model === "qwen2.5-coder:7b", "override model mismatch");

  let unknownRoleOk = false;
  try {
    await dispatcher.dispatch("missing_role", [{ role: "user", content: "x" }]);
  } catch (err) {
    unknownRoleOk = err.code === "LLM_ROLE_UNKNOWN";
  }
  assert(unknownRoleOk, "unknown role typed error missing");

  let unknownProviderOk = false;
  try {
    await dispatcher.dispatch("unknown_provider_role", [{ role: "user", content: "x" }]);
  } catch (err) {
    unknownProviderOk = err.code === "LLM_PROVIDER_UNKNOWN";
  }
  assert(unknownProviderOk, "unknown provider typed error missing");

  const validation = await dispatcher.validateProviders();
  assert(validation.ok === true, "validateProviders expected ok=true");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "llm_dispatcher");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "llm_dispatcher_canary.json");
  fs.writeFileSync(outPath, JSON.stringify({
    ok: true,
    cloudResult,
    localResult,
    overrideResult,
    validation,
    generated_at: new Date().toISOString(),
  }, null, 2));
  console.log(`# LLM Dispatcher Canary`);
  console.log(`- report: ${outPath.replace(/\\/g, "/")}`);
}

main().catch((err) => {
  console.error(`[llm_dispatcher_canary] failed: ${err.message}`);
  process.exit(1);
});
