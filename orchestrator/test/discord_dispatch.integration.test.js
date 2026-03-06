import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";

import { loadRegistryOrThrow } from "../src/registry.js";
import { normalizeInputRequest } from "../src/vnext/input_normalizer.js";
import { routeTaskRequest } from "../src/vnext/brain_router.js";
import { parseCoderDirectiveOptions, DEFAULT_CODER_MODEL } from "../src/vnext/coder_directive.js";

import { fileURLToPath } from "url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const registry = loadRegistryOrThrow(path.resolve(__dirname, "..", "..", "configs", "registry", "capability_registry.json"));

test("discord event normalization produces canonical source request", () => {
  const normalized = normalizeInputRequest({
    discord_event: {
      content: "Build CRM MVP",
      channel_id: "chan-1",
      thread_id: "thread-1",
      author: { id: "user-1", username: "lin" },
      attachments: [{ filename: "spec.png", url: "https://example.com/spec.png", content_type: "image/png" }],
    },
  });

  assert.equal(normalized.source, "discord");
  assert.equal(normalized.raw_input, "Build CRM MVP");
  assert.equal(normalized.context.channel_id, "chan-1");
  assert.equal(normalized.normalized_input.attachments.length, 1);
});

test("coder directive defaults to qwen-coder-next", () => {
  const parsed = parseCoderDirectiveOptions("Build a CRM MVP", { provider: "opencode" });
  assert.equal(parsed.model, DEFAULT_CODER_MODEL);
  assert.equal(parsed.provider, "opencode");
  assert.equal(parsed.taskPrompt, "Build a CRM MVP");
});

test("coder directive accepts explicit model and provider override", () => {
  const parsed = parseCoderDirectiveOptions("@provider=codex @model=deepseek-coder-v3 Build a CRM MVP", {
    provider: "opencode",
    model: DEFAULT_CODER_MODEL,
  });

  assert.equal(parsed.provider, "codex");
  assert.equal(parsed.model, "deepseek-coder-v3");
  assert.equal(parsed.taskPrompt, "Build a CRM MVP");
});

test("normalized discord coding request routes into coding workflow", () => {
  const normalized = normalizeInputRequest({
    source: "discord",
    raw_input: "Build a full-stack CRM MVP with frontend, backend, QA and tests",
    channel_id: "chan-2",
    user_id: "user-2",
  });

  const routed = routeTaskRequest({
    ...normalized,
    registry,
  });

  assert.equal(routed.task_envelope.target_team, "coding_team");
  assert.equal(routed.decision, "orchestrated_workflow");
});
