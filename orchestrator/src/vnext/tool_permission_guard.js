import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "../schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "..", "contracts", "guardrails");
const MATRIX_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "tool_permission.schema.json"), "utf8")
);

// Hardcoded matrix mapping roles to allowed tools.
// In a full system, this would be loaded dynamically from a config file.
const DEFAULT_MATRIX = {
  "pm_agent": ["coding.delegate", "document.read", "document.write"],
  "architect_agent": ["coding.delegate", "document.read", "document.write", "system.config_read"],
  "frontend_agent": ["coding.delegate", "file.read", "file.write", "bash.execute", "browser.test"],
  "backend_agent": ["coding.delegate", "file.read", "file.write", "bash.execute", "database.query"],
  "qa_agent": ["coding.delegate", "file.read", "bash.execute", "browser.test", "document.write"],
  "quant_agent": ["broker.query", "broker.trade", "data.fetch", "file.write"],
  "brain": ["none"]
};

export function validateToolPermission(roleName, toolName, matrix = DEFAULT_MATRIX) {
  const validation = validateJsonSchemaLite(MATRIX_SCHEMA, matrix, "$");
  if (validation.length > 0) {
    throw new Error(`Invalid tool permission matrix: ${validation.join(", ")}`);
  }

  const allowedTools = matrix[roleName] || [];
  
  if (allowedTools.includes("*")) {
    return { allowed: true };
  }

  if (allowedTools.includes(toolName)) {
    return { allowed: true };
  }

  return {
    allowed: false,
    reason: `Role '${roleName}' is not permitted to use tool '${toolName}'. Allowed tools: ${allowedTools.join(", ")}`
  };
}
