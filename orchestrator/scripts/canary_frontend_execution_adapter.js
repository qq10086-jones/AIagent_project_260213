import fs from "fs";
import path from "path";
import { buildFrontendExecutionPacket, validateFrontendExecutionPacket } from "../src/coding_execution_adapters.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const fixturePath = path.resolve(process.cwd(), "canary_inputs", "frontend_execution_adapter_min.json");
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));

  const packet = buildFrontendExecutionPacket({
    stepDef: { id: "impl_fe", role: "frontend" },
    payload: {
      expected_artifacts: ["patch/diff.patch", "tests/frontend_test_report.md", "run/run_frontend.md"],
      target_paths: ["sandbox/crm_site/"],
    },
    provider: "qwen",
    model: "qwen-coder-next",
  });
  const checked = validateFrontendExecutionPacket(packet);

  assertEqual(packet.adapter_id, fixture.expected.adapter_id, "adapter_id");
  assertEqual(packet.role, fixture.expected.role, "role");
  assertEqual(packet.step_id, fixture.expected.step_id, "step_id");
  assertEqual(checked.ok, true, "checked.ok");
  assertEqual(checked.schema_id, fixture.expected.schema_id, "schema_id");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "frontend_execution_adapter");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "frontend_execution_adapter_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    packet,
    checked,
  }, null, 2), "utf8");
  console.log("# Frontend Execution Adapter Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
