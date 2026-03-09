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


function main() {
  const fixturePath = path.resolve(process.cwd(), "canary_inputs", "agent_contract_layer_min.json");
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  const agentRegistry = loadAgentContractsOrThrow(getDefaultAgentRegistryDir());
  const promptRegistry = loadPromptScriptRegistryOrThrow(getDefaultPromptScriptRegistryPath());
  const bindingCheck = validatePromptScriptsAgainstAgents({ promptRegistry, agentRegistry });
  if (!bindingCheck.ok) {
    throw new Error(bindingCheck.errors.join("; "));
  }

  const expectedAgents = fixture.expected?.agents || [];
  for (const agentId of expectedAgents) {
    if (!agentRegistry.agents?.[agentId]) {
      throw new Error(`missing agent '${agentId}'`);
    }
  }

  const boundScripts = fixture.expected?.bound_scripts || {};
  for (const [scriptId, agentId] of Object.entries(boundScripts)) {
    const script = promptRegistry.scripts?.[scriptId];
    if (!script) throw new Error(`missing script '${scriptId}'`);
    if (String(script.agent_id || "") !== agentId) {
      throw new Error(`script '${scriptId}' agent_id mismatch`);
    }
  }

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "agent_contract_layer");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "agent_contract_layer_canary.json");
  const report = {
    ok: true,
    generated_at: new Date().toISOString(),
    agent_dir: getDefaultAgentRegistryDir(),
    prompt_registry: getDefaultPromptScriptRegistryPath(),
    scripts_checked: Object.keys(boundScripts).length,
    agents_checked: expectedAgents.length,
  };
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log("# Agent Contract Layer Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
  console.log(`- agents_checked: ${expectedAgents.length}`);
  console.log(`- scripts_checked: ${Object.keys(boundScripts).length}`);
}

main();
