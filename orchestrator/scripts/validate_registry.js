import fs from "fs";
import path from "path";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow } from "../src/prompt_script_registry.js";
import { resolveRepoPath } from "./_paths.js";

const registryPath = process.env.REGISTRY_PATH && process.env.REGISTRY_PATH.trim()
  ? process.env.REGISTRY_PATH.trim()
  : getDefaultRegistryPath();

const schemaPath = resolveRepoPath("configs", "registry", "schemas", "capability_registry.schema.json");
const promptRegistryPaths = [
  resolveRepoPath("orchestrator", "configs", "prompt_scripts", "registry.json"),
  resolveRepoPath("configs", "prompt_scripts", "registry.json"),
];
const llmProvidersPath = resolveRepoPath("orchestrator", "configs", "llm_providers.json");
const llmRolePolicyPath = resolveRepoPath("orchestrator", "configs", "llm_role_policy.json");
const contextBudgetPolicyPath = resolveRepoPath("orchestrator", "configs", "context_budget_policy.json");
const requiredContractSchemas = [
  resolveRepoPath("orchestrator", "contracts", "coding_team_be_to_fe_handoff.schema.json"),
  resolveRepoPath("orchestrator", "contracts", "coding_team_patch_bundle.schema.json"),
  resolveRepoPath("orchestrator", "contracts", "coding_team_impl_to_qa_handoff.schema.json"),
  resolveRepoPath("orchestrator", "contracts", "context_budget_report.schema.json"),
  resolveRepoPath("orchestrator", "contracts", "coding_team_qa_to_release_handoff.schema.json"),
];

function ensureJsonFile(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`file not found: ${filePath}`);
  }
  JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function main() {
  try {
    if (!fs.existsSync(registryPath)) {
      throw new Error(`registry file not found: ${registryPath}`);
    }
    if (!fs.existsSync(schemaPath)) {
      throw new Error(`schema file not found: ${schemaPath}`);
    }
    // We intentionally do structural validation in code to avoid adding extra deps.
    JSON.parse(fs.readFileSync(schemaPath, "utf8"));
    const reg = loadRegistryOrThrow(registryPath);
    ensureJsonFile(llmProvidersPath);
    ensureJsonFile(llmRolePolicyPath);
    ensureJsonFile(contextBudgetPolicyPath);
    for (const item of requiredContractSchemas) ensureJsonFile(item);
    for (const promptRegistryPath of promptRegistryPaths) {
      const promptRegistry = loadPromptScriptRegistryOrThrow(promptRegistryPath);
      for (const [scriptId, spec] of Object.entries(promptRegistry.scripts || {})) {
        if (Object.prototype.hasOwnProperty.call(spec, "model")) {
          throw new Error(`prompt script '${scriptId}' contains deprecated model field`);
        }
        if (!String(spec.llm_role || "").trim()) {
          throw new Error(`prompt script '${scriptId}' missing llm_role`);
        }
      }
    }

    const projectTypeCount = Object.keys(reg.project_types || {}).length;
    const roleCount = Object.keys(reg.roles || {}).length;
    const toolCount = Object.keys(reg.tools || {}).length;
    const workflowCount = Object.keys(reg.workflows || {}).length;
    console.log(`[registry] valid: ${registryPath}`);
    console.log(`[registry] counts: project_types=${projectTypeCount}, roles=${roleCount}, tools=${toolCount}, workflows=${workflowCount}`);
    console.log(`[registry] prompt_registries=${promptRegistryPaths.map((item) => path.relative(process.cwd(), item).replace(/\\/g, "/")).join(",")}`);
    console.log(`[registry] llm_configs=ok context_budget_policy=ok contract_schemas=${requiredContractSchemas.length}`);
  } catch (err) {
    console.error(`[registry] invalid: ${err.message}`);
    process.exit(1);
  }
}

main();
