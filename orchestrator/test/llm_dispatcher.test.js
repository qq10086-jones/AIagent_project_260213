import test from "node:test";
import assert from "node:assert/strict";

import { createLlmDispatcher } from "../src/vnext/llm_dispatcher.js";

const baseProviders = {
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

const baseRolePolicy = {
  roles: {
    pm: { provider: "cloud_qwen", model: "qwen-max" },
    backend: { provider: "local_ollama", model: "deepseek-r1:32b", secondary_model: "qwen2.5-coder:7b" },
  },
  retry_policy: { strategy: "exponential_backoff", retries: 3, base_delay_ms: 1 },
  fallback_policy: "model_fallback",
};

test("dispatch routes cloud_api role", async () => {
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
    qwenClient: async (messages, options) => {
      assert.equal(messages[0].content, "plan this");
      assert.equal(options.qwenModel, "qwen-max");
      return "cloud reply";
    },
    ollamaClient: async () => {
      throw new Error("should not call ollama");
    },
  });

  const result = await dispatcher.dispatch("pm", [{ role: "user", content: "plan this" }]);
  assert.equal(result.content, "cloud reply");
  assert.equal(result.provider, "cloud_qwen");
  assert.equal(result.model, "qwen-max");
  assert.equal(result.used_fallback, false);
});

test("dispatch routes local_ollama role", async () => {
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
    qwenClient: async () => {
      throw new Error("should not call qwen");
    },
    ollamaClient: async (model, messages) => {
      assert.equal(model, "deepseek-r1:32b");
      assert.equal(messages[0].content, "write code");
      return "local reply";
    },
  });

  const result = await dispatcher.dispatch("backend", [{ role: "user", content: "write code" }]);
  assert.equal(result.content, "local reply");
  assert.equal(result.provider, "local_ollama");
  assert.equal(result.model, "deepseek-r1:32b");
});

test("dispatch applies explicit override model", async () => {
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
    ollamaClient: async (model) => {
      assert.equal(model, "override-model");
      return "override reply";
    },
  });

  const result = await dispatcher.dispatch("backend", [{ role: "user", content: "write code" }], { model: "override-model" });
  assert.equal(result.model, "override-model");
  assert.equal(result.content, "override reply");
});

test("dispatch retries transport errors up to three times", async () => {
  let attempts = 0;
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
    qwenClient: async () => {
      attempts += 1;
      if (attempts < 4) throw new Error("network timeout");
      return "recovered";
    },
    sleep: async () => {},
  });

  const result = await dispatcher.dispatch("pm", [{ role: "user", content: "plan this" }]);
  assert.equal(result.content, "recovered");
  assert.equal(attempts, 4);
});

test("dispatch falls back to secondary_model on OOM", async () => {
  const seen = [];
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
    ollamaClient: async (model) => {
      seen.push(model);
      if (model === "deepseek-r1:32b") throw new Error("OOM while loading model");
      return "fallback reply";
    },
  });

  const result = await dispatcher.dispatch("backend", [{ role: "user", content: "write code" }]);
  assert.deepEqual(seen, ["deepseek-r1:32b", "qwen2.5-coder:7b"]);
  assert.equal(result.used_fallback, true);
  assert.equal(result.model, "qwen2.5-coder:7b");
});

test("dispatch surfaces typed error when fallback also fails", async () => {
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
    ollamaClient: async () => {
      throw new Error("context overflow");
    },
  });

  await assert.rejects(
    dispatcher.dispatch("backend", [{ role: "user", content: "write code" }]),
    (err) => err.code === "LLM_DISPATCH_FAILED" && err.fallback_attempted === true
  );
});

test("dispatch rejects unknown role", async () => {
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
  });

  await assert.rejects(
    dispatcher.dispatch("release", [{ role: "user", content: "ship it" }]),
    (err) => err.code === "LLM_ROLE_UNKNOWN"
  );
});

test("validateProviders returns ok when env and models are present", async () => {
  process.env.QWEN_API_KEY = "test-key";
  process.env.QWEN_BASE_URL = "https://qwen.example";
  process.env.OLLAMA_BASE_URL = "http://ollama.local";
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
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
  });

  const result = await dispatcher.validateProviders();
  assert.equal(result.ok, true);
  assert.equal(result.results.every((item) => item.status === "ok"), true);
});

test("validateProviders returns degraded when models are missing", async () => {
  process.env.QWEN_API_KEY = "test-key";
  process.env.QWEN_BASE_URL = "https://qwen.example";
  process.env.OLLAMA_BASE_URL = "http://ollama.local";
  const dispatcher = createLlmDispatcher({
    providers: baseProviders,
    rolePolicy: baseRolePolicy,
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return { models: [{ name: "deepseek-r1:32b" }] };
      },
    }),
  });

  const result = await dispatcher.validateProviders();
  assert.equal(result.ok, false);
  assert.equal(result.results.some((item) => item.status === "degraded"), true);
});
