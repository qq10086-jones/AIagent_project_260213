import fs from "fs";
import path from "path";
import { executeCodingAdapter } from "../coding_executor_runtime.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

async function main() {
  const adapterRequest = {
    adapter_type: "coding_executor",
    provider: "unsupported-provider",
    task_type: "coding_execution",
    payload: {
      step_id: "impl_be",
      task_prompt: "Implement backend login API",
      artifact_root: "artifacts/release/run-1",
      expected_artifacts: ["patch/diff.patch"],
      execution_adapter_packet: {
        adapter_id: "backend.execution.v1",
      },
    },
    context: {
      run_id: "run-1",
      task_id: "task-1",
    },
  };

  const unsupported = await executeCodingAdapter({
    workspaceRoot: process.cwd(),
    adapterRequest,
    provider: "unsupported-provider",
    model: "qwen-coder-next",
    maxRuntimeS: 60,
  });

  assertEqual(unsupported.ok, false, "unsupported.ok");
  assertEqual(unsupported.diagnostics.error_code, "E_PROVIDER_UNAVAILABLE", "unsupported.error_code");
  assertEqual(unsupported.diagnostics.adapter_type, "coding_executor", "unsupported.adapter_type");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "coding_executor_runtime");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "coding_executor_runtime_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    unsupported_provider_case: unsupported,
  }, null, 2), "utf8");
  console.log("# Coding Executor Runtime Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
