import fs from "fs";
import path from "path";
import { loadRegistryOrThrow } from "../src/registry.js";
import { normalizeInputRequest } from "../src/vnext/input_normalizer.js";
import { routeTaskRequest } from "../src/vnext/brain_router.js";
import { buildRouteContractResponse } from "../src/vnext/route_contract.js";
import { buildDispatchContractPreview } from "../src/vnext/dispatch_contract.js";
import { parseCoderDirectiveOptions, DEFAULT_CODER_MODEL } from "../src/vnext/coder_directive.js";
import {
  makeDirectReplyResponse,
  makeTaskQueuedResponse,
  makeWorkflowQueuedResponse,
  makeErrorResponse,
} from "../src/vnext/response_protocol.js";

function arg(name, fallback = "") {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return String(process.argv[idx + 1]);
  return fallback;
}

function parseBool(v, fallback = true) {
  const s = String(v || "").trim().toLowerCase();
  if (!s) return fallback;
  return ["1", "true", "yes", "on"].includes(s);
}

function loadJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function makeMarkdownReport(report) {
  const lines = [
    "# Canary Report (brain_router_dod_close_v1)",
    "",
    `- generated_at: ${report.generated_at}`,
    `- total_cases: ${report.total_cases}`,
    `- passed_cases: ${report.passed_cases}`,
    `- failed_cases: ${report.failed_cases}`,
    "",
    "## Cases",
  ];

  for (const item of report.cases) {
    lines.push(`- ${item.name}: ${item.ok ? "PASS" : "FAIL"}`);
    if (!item.ok) lines.push(`  - detail: ${item.detail}`);
  }
  return lines.join("\n");
}

function assertEqual(actual, expected, label) {
  if (expected === undefined) return;
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function runRoutingCase(fixture, registry) {
  const normalized = normalizeInputRequest(fixture.request || {});
  const routed = routeTaskRequest({
    ...normalized,
    registry,
  });
  const expected = fixture.expected || {};
  assertEqual(routed.decision, expected.decision, "decision");
  assertEqual(routed.task_envelope.intent, expected.intent, "intent");
  assertEqual(routed.task_envelope.target_team, expected.target_team, "target_team");
  if (expected.requires_orchestration !== undefined) {
    assertEqual(routed.task_envelope.requires_orchestration, expected.requires_orchestration, "requires_orchestration");
  }
  if (expected.tool_name) {
    assertEqual(routed.task_envelope.execution_plan?.tool_name || "", expected.tool_name, "tool_name");
  }
  if (expected.workflow_id) {
    assertEqual(routed.task_envelope.execution_plan?.workflow_id || "", expected.workflow_id, "workflow_id");
  }
}

function runCoderCase(fixture, defaults) {
  const text = String(fixture.coder_directive || "");
  const body = text.replace(/^\/coder(?::|：|\s+)/i, "").trim();
  const parsed = parseCoderDirectiveOptions(body, defaults);
  const expected = fixture.expected || {};
  assertEqual(parsed.provider, expected.provider, "provider");
  assertEqual(parsed.model, expected.model, "model");
  assertEqual(parsed.taskPrompt, expected.task_prompt, "task_prompt");
}

function buildStubEnvelope() {
  return {
    task_id: "task-1",
    source: "discord",
    raw_input: "Build a CRM MVP",
    normalized_input: {
      text: "Build a CRM MVP",
      attachments: [],
      metadata: { user_id: "u1", channel_id: "c1", thread_id: "t1" },
    },
    intent: "coding",
    sub_intent: "coder_directive",
    requires_orchestration: true,
    target_team: "coding_team",
    expected_outputs: ["design_doc", "repo_changes"],
    constraints: { local_only: true, approval_mode: "manual", risk_level: "medium", complexity: "complex" },
    context: { channel_id: "c1", thread_id: "t1", user_id: "u1", attachments: [] },
    decision: "orchestrated_workflow",
    execution_plan: { mode: "orchestrated_workflow", workflow_id: "coding_team_v0", project_type: "webapp_crm" },
  };
}

function runResponseCase(fixture) {
  const stubEnvelope = buildStubEnvelope();
  let response = null;
  if (fixture.response_case === "direct_reply") {
    response = makeDirectReplyResponse({ run_id: "run-1", task_envelope: stubEnvelope, reply: "ok" });
  } else if (fixture.response_case === "progress_update") {
    response = makeWorkflowQueuedResponse({
      run_id: "run-1",
      task_envelope: stubEnvelope,
      workflow_run_id: "wf-1",
      workflow_id: "coding_team_v0",
      first_step: null,
    });
  } else if (fixture.response_case === "approval_request") {
    response = makeTaskQueuedResponse({
      run_id: "run-1",
      task_envelope: stubEnvelope,
      task_id: "task-2",
      tool_name: "coding.delegate",
      waiting_approval: true,
    });
  } else if (fixture.response_case === "error") {
    response = makeErrorResponse({
      run_id: "run-1",
      error: "raw_input/message is required",
      error_code: "BAD_REQUEST",
      task_envelope: null,
    });
  } else if (fixture.response_case === "dispatch_error_shape") {
    response = makeErrorResponse({
      run_id: "run-1",
      error: "raw_input/message is required",
      error_code: "BAD_REQUEST",
      task_envelope: null,
    });
  } else {
    throw new Error(`unknown response_case: ${fixture.response_case}`);
  }

  const expected = fixture.expected || {};
  assertEqual(response.ok, expected.ok, "ok");
  assertEqual(response.response_mode, expected.response_mode, "response_mode");
  if (expected.error_code) {
    assertEqual(response.error_code, expected.error_code, "error_code");
  }
}

function runHttpRouteCase(fixture, registry) {
  const result = buildRouteContractResponse({
    body: fixture.request || {},
    analyzerResult: null,
    registry,
  });
  const expected = fixture.expected || {};
  assertEqual(result.ok, true, "ok");
  assertEqual(result.decision, expected.decision, "decision");
  assertEqual(result.task_envelope.intent, expected.intent, "intent");
  if (expected.workflow_id) {
    assertEqual(result.task_envelope.execution_plan?.workflow_id || "", expected.workflow_id, "workflow_id");
  }
}

function runHttpDispatchCase(fixture, registry) {
  const result = buildDispatchContractPreview({
    body: fixture.request || {},
    analyzerResult: null,
    registry,
    routeOverride: fixture.route_override || null,
    previewSeed: fixture.name || "preview",
  });
  const expected = fixture.expected || {};
  assertEqual(result.ok, expected.ok, "ok");
  assertEqual(result.response_mode, expected.response_mode, "response_mode");
  if (expected.error_code) {
    assertEqual(result.error_code, expected.error_code, "error_code");
  }
  if (expected.intent) {
    assertEqual(result.task_envelope?.intent || "", expected.intent, "intent");
  }
  if (expected.tool_name) {
    assertEqual(result.execution?.tool_name || "", expected.tool_name, "tool_name");
  }
  if (expected.workflow_id) {
    assertEqual(result.execution?.workflow_id || "", expected.workflow_id, "workflow_id");
  }
}

function main() {
  const strict = parseBool(arg("strict", "true"), true);
  const workspaceRoot = path.resolve(arg("workspace-root", process.cwd()));
  const inputs = [
    "brain_router_chat.json",
    "brain_router_coding_simple.json",
    "brain_router_coding_complex.json",
    "brain_router_coder_default.json",
    "brain_router_coder_explicit.json",
    "dispatch_response_direct_reply.json",
    "dispatch_response_progress.json",
    "dispatch_response_approval.json",
    "dispatch_response_error.json",
    "http_dispatch_contract_error_shape.json",
    "http_route_chat.json",
    "http_route_coding.json",
    "http_dispatch_chat.json",
    "http_dispatch_coding_simple.json",
    "http_dispatch_coder.json",
    "http_dispatch_error.json",
  ];
  const registry = loadRegistryOrThrow(path.resolve(process.cwd(), "..", "configs", "registry", "capability_registry.json"));
  const defaults = {
    provider: "opencode",
    model: DEFAULT_CODER_MODEL,
  };

  const cases = [];
  for (const name of inputs) {
    const fixturePath = path.resolve(process.cwd(), "canary_inputs", name);
    const fixture = loadJson(fixturePath);
    const item = { name: fixture.name || name, ok: true, detail: "" };
    try {
      if (fixture.kind === "response") runResponseCase(fixture);
      else if (fixture.kind === "http_route") runHttpRouteCase(fixture, registry);
      else if (fixture.kind === "http_dispatch") runHttpDispatchCase(fixture, registry);
      else if (fixture.coder_directive) runCoderCase(fixture, defaults);
      else if (fixture.request) runRoutingCase(fixture, registry);
      else throw new Error("fixture missing request or coder_directive");
    } catch (err) {
      item.ok = false;
      item.detail = err.message || String(err);
    }
    cases.push(item);
  }

  const passedCases = cases.filter((item) => item.ok).length;
  const failedCases = cases.length - passedCases;
  const report = {
    generated_at: new Date().toISOString(),
    total_cases: cases.length,
    passed_cases: passedCases,
    failed_cases: failedCases,
    cases,
  };

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir = path.resolve(workspaceRoot, "artifacts", "canary", "brain_router", stamp);
  fs.mkdirSync(outDir, { recursive: true });
  const jsonPath = path.join(outDir, "canary_report.json");
  const mdPath = path.join(outDir, "canary_report.md");
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), "utf8");
  fs.writeFileSync(mdPath, makeMarkdownReport(report), "utf8");

  console.log("# Canary Report");
  console.log(`- json: ${jsonPath.replace(/\\/g, "/")}`);
  console.log(`- md: ${mdPath.replace(/\\/g, "/")}`);
  console.log(`- passed_cases: ${passedCases}`);
  console.log(`- failed_cases: ${failedCases}`);
  if (strict && failedCases > 0) process.exit(1);
}

main();
