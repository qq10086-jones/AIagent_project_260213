import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "./schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "contracts");
const BACKEND_EXECUTION_PACKET_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "backend_execution_packet.schema.json"), "utf8")
);
const FRONTEND_EXECUTION_PACKET_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "frontend_execution_packet.schema.json"), "utf8")
);

export function buildBackendExecutionPacket({ stepDef, payload = {}, provider = "", model = "" }) {
  return {
    adapter_id: "backend.execution.v1",
    role: String(stepDef?.role || "backend"),
    step_id: String(stepDef?.id || "impl_be"),
    target_paths: Array.isArray(payload.target_paths) && payload.target_paths.length > 0
      ? payload.target_paths
      : ["sandbox/crm_site/"],
    required_outputs: Array.isArray(payload.expected_artifacts) && payload.expected_artifacts.length > 0
      ? payload.expected_artifacts
      : ["patch/diff.patch", "tests/backend_test_report.md", "run/run_backend.md"],
    input_artifacts: [
      "plan/arch.md",
      "risk/risk_report.json",
      "plan/workplan.md",
      "handoff/architect_to_impl.json",
    ],
    execution_mode: "patch_and_verify",
    verification_hint: "return diff summary, changed files, backend test evidence, and runbook",
    provider_hint: String(provider || ""),
    model_hint: String(model || ""),
  };
}

export function validateBackendExecutionPacket(packet) {
  const errors = validateJsonSchemaLite(BACKEND_EXECUTION_PACKET_SCHEMA, packet, "$");
  return {
    ok: errors.length === 0,
    errors,
    schema_id: BACKEND_EXECUTION_PACKET_SCHEMA.$id,
  };
}

export function buildFrontendExecutionPacket({ stepDef, payload = {}, provider = "", model = "" }) {
  return {
    adapter_id: "frontend.execution.v1",
    role: String(stepDef?.role || "frontend"),
    step_id: String(stepDef?.id || "impl_fe"),
    target_paths: Array.isArray(payload.target_paths) && payload.target_paths.length > 0
      ? payload.target_paths
      : ["sandbox/crm_site/"],
    required_outputs: Array.isArray(payload.expected_artifacts) && payload.expected_artifacts.length > 0
      ? payload.expected_artifacts
      : ["patch/diff.patch", "tests/frontend_test_report.md", "run/run_frontend.md"],
    input_artifacts: [
      "plan/arch.md",
      "plan/workplan.md",
      "handoff/architect_to_impl.json",
    ],
    execution_mode: "ui_patch_and_verify",
    verification_hint: "return diff summary, changed files, frontend test evidence, and runbook",
    provider_hint: String(provider || ""),
    model_hint: String(model || ""),
  };
}

export function validateFrontendExecutionPacket(packet) {
  const errors = validateJsonSchemaLite(FRONTEND_EXECUTION_PACKET_SCHEMA, packet, "$");
  return {
    ok: errors.length === 0,
    errors,
    schema_id: FRONTEND_EXECUTION_PACKET_SCHEMA.$id,
  };
}
