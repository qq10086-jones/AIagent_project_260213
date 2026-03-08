import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "./schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "contracts");
const REQUEST_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "tool_adapter_request.schema.json"), "utf8")
);
const RESULT_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "tool_adapter_result.schema.json"), "utf8")
);

const TOOL_ADAPTERS = {
  coding_executor: {
    providers: ["opencode", "codex"],
    task_types: ["coding_execution"],
  },
};

export function getToolAdapterSpec(adapterType) {
  return TOOL_ADAPTERS[String(adapterType || "")] || null;
}

export function validateToolAdapterRequest(request) {
  const errors = validateJsonSchemaLite(REQUEST_SCHEMA, request, "$");
  const spec = getToolAdapterSpec(request?.adapter_type);
  if (!spec) errors.push(`$.adapter_type unsupported: ${String(request?.adapter_type || "")}`);
  if (spec && !spec.providers.includes(String(request?.provider || ""))) {
    errors.push(`$.provider unsupported for ${request.adapter_type}: ${String(request?.provider || "")}`);
  }
  if (spec && !spec.task_types.includes(String(request?.task_type || ""))) {
    errors.push(`$.task_type unsupported for ${request.adapter_type}: ${String(request?.task_type || "")}`);
  }
  return {
    ok: errors.length === 0,
    errors,
    schema_id: REQUEST_SCHEMA.$id,
  };
}

export function validateToolAdapterResult(result) {
  const errors = validateJsonSchemaLite(RESULT_SCHEMA, result, "$");
  return {
    ok: errors.length === 0,
    errors,
    schema_id: RESULT_SCHEMA.$id,
  };
}
