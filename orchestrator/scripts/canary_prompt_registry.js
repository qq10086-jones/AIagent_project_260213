import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { loadRegistryOrThrow } from "../src/registry.js";
import {
  loadPromptScriptRegistryOrThrow,
} from "../src/prompt_script_registry.js";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const fixturePath = path.resolve(process.cwd(), "canary_inputs", "prompt_script_registry_min.json");
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  const promptRegistry = loadPromptScriptRegistryOrThrow(
    path.resolve(SCRIPT_DIR, "..", "..", "configs", "prompt_scripts", "registry.json")
  );
  const capabilityRegistry = loadRegistryOrThrow(
    path.resolve(SCRIPT_DIR, "..", "..", "configs", "registry", "capability_registry.json")
  );

  const expectedScripts = fixture.expected?.scripts || [];
  for (const scriptId of expectedScripts) {
    if (!promptRegistry.scripts?.[scriptId]) {
      throw new Error(`missing script '${scriptId}'`);
    }
  }

  const workflow = capabilityRegistry.workflows?.coding_team_v0;
  if (!workflow) throw new Error("coding_team_v0 not found");
  const bindings = fixture.expected?.workflow_bindings || {};
  for (const [stepId, scriptId] of Object.entries(bindings)) {
    const step = (workflow.steps || []).find((item) => String(item.id || "") === stepId);
    if (!step) throw new Error(`workflow step '${stepId}' not found`);
    assertEqual(String(step.prompt_script_id || ""), scriptId, `${stepId}.prompt_script_id`);
  }

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "prompt_registry");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "prompt_registry_canary.json");
  const report = {
    ok: true,
    checked_scripts: expectedScripts,
    workflow_bindings: bindings,
    generated_at: new Date().toISOString(),
  };
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`# Prompt Registry Canary`);
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
  console.log(`- scripts_checked: ${expectedScripts.length}`);
}

main();
