import fs from "fs";
import path from "path";
import {
  getDefaultAgentRegistryDir,
  loadAgentContractsOrThrow,
} from "../src/agent_contract_registry.js";
import {
  getDefaultPromptScriptRegistryPath,
  loadPromptScriptRegistryOrThrow,
  validatePromptScriptsAgainstAgents,
} from "../src/prompt_script_registry.js";
import { loadRegistryOrThrow } from "../src/registry.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const fixturePath = path.resolve(process.cwd(), "canary_inputs", "runtime_contract_hardening_min.json");
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  const agentRegistry = loadAgentContractsOrThrow(getDefaultAgentRegistryDir());
  const promptRegistry = loadPromptScriptRegistryOrThrow(getDefaultPromptScriptRegistryPath());
  const capabilityRegistry = loadRegistryOrThrow(path.resolve(process.cwd(), "..", "configs", "registry", "capability_registry.json"));

  const binding = validatePromptScriptsAgainstAgents({ promptRegistry, agentRegistry });
  assertEqual(binding.ok, fixture.expected?.agent_binding_ok, "agent_binding_ok");

  const workflow = capabilityRegistry.workflows?.coding_team_v0;
  if (!workflow) throw new Error("coding_team_v0 not found");
  for (const [stepId, scriptId] of Object.entries(fixture.expected?.workflow_prompt_bindings || {})) {
    const step = (workflow.steps || []).find((item) => String(item.id || "") === stepId);
    if (!step) throw new Error(`workflow step '${stepId}' missing`);
    assertEqual(String(step.prompt_script_id || ""), scriptId, `${stepId}.prompt_script_id`);
    if (!promptRegistry.scripts?.[scriptId]) {
      throw new Error(`prompt script '${scriptId}' missing from runtime registry`);
    }
  }

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "runtime_contract_hardening");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "runtime_contract_hardening_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    binding_errors: binding.errors,
  }, null, 2), "utf8");
  console.log("# Runtime Contract Hardening Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
