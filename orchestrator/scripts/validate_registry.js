import fs from "fs";
import path from "path";
import { loadRegistryOrThrow } from "../src/registry.js";

const defaultRegistry = path.join(process.cwd(), "configs", "registry", "capability_registry.json");
const parentRegistry = path.join(process.cwd(), "..", "configs", "registry", "capability_registry.json");
const fallbackRegistry = path.join(process.cwd(), "configs", "capability_registry.json");
const parentFallbackRegistry = path.join(process.cwd(), "..", "configs", "capability_registry.json");
const registryPath = process.env.REGISTRY_PATH && process.env.REGISTRY_PATH.trim()
  ? process.env.REGISTRY_PATH.trim()
  : (
    fs.existsSync(defaultRegistry) ? defaultRegistry
      : (fs.existsSync(parentRegistry) ? parentRegistry
        : (fs.existsSync(fallbackRegistry) ? fallbackRegistry : parentFallbackRegistry))
  );

const schemaPath = path.join(process.cwd(), "configs", "registry", "schemas", "capability_registry.schema.json");
const parentSchemaPath = path.join(process.cwd(), "..", "configs", "registry", "schemas", "capability_registry.schema.json");

function main() {
  try {
    if (!fs.existsSync(registryPath)) {
      throw new Error(`registry file not found: ${registryPath}`);
    }
    const schemaToUse = fs.existsSync(schemaPath) ? schemaPath : parentSchemaPath;
    if (!fs.existsSync(schemaToUse)) {
      throw new Error(`schema file not found: ${schemaPath} or ${parentSchemaPath}`);
    }
    // We intentionally do structural validation in code to avoid adding extra deps.
    JSON.parse(fs.readFileSync(schemaToUse, "utf8"));
    const reg = loadRegistryOrThrow(registryPath);
    const projectTypeCount = Object.keys(reg.project_types || {}).length;
    const roleCount = Object.keys(reg.roles || {}).length;
    const toolCount = Object.keys(reg.tools || {}).length;
    const workflowCount = Object.keys(reg.workflows || {}).length;
    console.log(`[registry] valid: ${registryPath}`);
    console.log(`[registry] counts: project_types=${projectTypeCount}, roles=${roleCount}, tools=${toolCount}, workflows=${workflowCount}`);
  } catch (err) {
    console.error(`[registry] invalid: ${err.message}`);
    process.exit(1);
  }
}

main();
