import fs from "fs";
import path from "path";
import { buildBackendExecutionPacket } from "../src/coding_execution_adapters.js";
import {
  buildCodingExecutorRequest,
  buildCodingExecutorResult,
  validateCodingExecutorRequest,
  validateCodingExecutorResult,
} from "../src/coding_executor.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const fixture = JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), "canary_inputs", "tool_adapter_interface_min.json"), "utf8")
  );

  const executionPacket = buildBackendExecutionPacket({
    stepDef: { id: "impl_be", role: "backend" },
    payload: {
      workflow_id: "coding_team_v0",
      workflow_run_id: "wf-1",
      run_id: "run-1",
      artifact_root: "artifacts/release/run-1",
      expected_artifacts: ["impl/be_changes/server.js", "impl/be_notes.md", "handoff/be_to_fe.json"],
      target_paths: ["sandbox/crm_site/"],
      task_prompt: "Implement backend login API",
    },
    provider: "opencode",
    model: "qwen-coder-next",
  });

  const request = buildCodingExecutorRequest({
    provider: "opencode",
    payload: {
      role: "backend",
      step_id: "impl_be",
      workflow_id: "coding_team_v0",
      workflow_run_id: "wf-1",
      run_id: "run-1",
      task_prompt: "Implement backend login API",
      artifact_root: "artifacts/release/run-1",
      expected_artifacts: ["impl/be_changes/server.js", "impl/be_notes.md", "handoff/be_to_fe.json"],
      target_paths: ["sandbox/crm_site/"],
      execution_adapter_packet: executionPacket,
    },
    executionPacket,
    role: "backend",
    stepId: "impl_be",
  });
  const requestChecked = validateCodingExecutorRequest(request);

  const result = buildCodingExecutorResult({
    provider: "opencode",
    adapterResult: {
      ok: true,
      provider_used: "opencode",
      model_used: "qwen-coder-next",
      command_used: "opencode qwen-coder-next",
      command_source: "opencode_cli",
      diagnostics: { error_code: null, timeout: false },
      files_changed: ["sandbox/crm_site/server.js"],
      diff_stats: { files: 1 },
      artifacts: { diff_bundle: "artifacts/runs/run-1/task_t1/diff.patch" },
      error: null,
    },
  });
  const resultChecked = validateCodingExecutorResult(result);

  assertEqual(request.adapter_type, fixture.expected.adapter_type, "request.adapter_type");
  assertEqual(request.provider, fixture.expected.provider, "request.provider");
  assertEqual(requestChecked.ok, true, "requestChecked.ok");
  assertEqual(requestChecked.schema_id, fixture.expected.request_schema_id, "requestChecked.schema_id");
  assertEqual(resultChecked.ok, true, "resultChecked.ok");
  assertEqual(resultChecked.schema_id, fixture.expected.result_schema_id, "resultChecked.schema_id");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "tool_adapter_interface");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "tool_adapter_interface_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    request,
    request_checked: requestChecked,
    result,
    result_checked: resultChecked,
  }, null, 2), "utf8");
  console.log("# Tool Adapter Interface Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
