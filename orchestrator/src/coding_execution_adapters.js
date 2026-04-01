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
  const diffFirst = String(payload?.execution_mode_requested || "") === "structured_patch";
  return {
    adapter_id: "backend.execution.v1",
    role: String(stepDef?.role || "backend"),
    step_id: String(stepDef?.id || "impl_be"),
    target_paths: Array.isArray(payload.target_paths) && payload.target_paths.length > 0
      ? payload.target_paths
      : ["workspace/sandbox/app/"],
    required_outputs: Array.isArray(payload.expected_artifacts) && payload.expected_artifacts.length > 0
      ? payload.expected_artifacts
      : diffFirst
        ? ["impl/be_patch_bundle.json", "impl/be_notes.md", "handoff/be_to_fe.json"]
        : ["impl/be_changes/server.js", "impl/be_notes.md", "handoff/be_to_fe.json"],
    input_artifacts: [
      "plan/arch.md",
      "plan/interfaces.md",
      "risk/risk_report.json",
      "plan/workplan.md",
      "plan/workplan.json",
      "handoff/architect_to_impl.json",
    ],
    execution_mode: diffFirst ? "structured_patch_and_handoff" : "file_output_and_handoff",
    verification_hint: diffFirst
      ? "return a structured backend patch bundle, backend notes, and a typed BE-to-FE handoff manifest"
      : "return complete backend files, backend notes, and a typed BE-to-FE handoff manifest",
    verification_command: String(payload?.verification_command || ""),
    max_attempts: Number(payload?.max_attempts || 1),
    same_error_repeat_limit: Number(payload?.same_error_repeat_limit || 1),
    wall_clock_timeout_s: Number(payload?.wall_clock_timeout_s || 0),
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
  const diffFirst = String(payload?.execution_mode_requested || "") === "structured_patch";
  return {
    adapter_id: "frontend.execution.v1",
    role: String(stepDef?.role || "frontend"),
    step_id: String(stepDef?.id || "impl_fe"),
    target_paths: Array.isArray(payload.target_paths) && payload.target_paths.length > 0
      ? payload.target_paths
      : ["workspace/sandbox/app/"],
    required_outputs: Array.isArray(payload.expected_artifacts) && payload.expected_artifacts.length > 0
      ? payload.expected_artifacts
      : diffFirst
        ? ["impl/fe_patch_bundle.json", "impl/fe_notes.md"]
        : ["impl/fe_changes/app.js", "impl/fe_notes.md"],
    input_artifacts: [
      "plan/arch.md",
      "plan/interfaces.md",
      "plan/workplan.md",
      "plan/workplan.json",
      "handoff/architect_to_impl.json",
      "handoff/be_to_fe.json",
    ],
    execution_mode: diffFirst ? "structured_patch_from_backend_handoff" : "file_output_from_backend_handoff",
    verification_hint: diffFirst
      ? "return a structured frontend patch bundle and frontend notes that match the backend handoff contract"
      : "return complete frontend files and frontend notes that match the backend handoff contract",
    verification_command: String(payload?.verification_command || ""),
    max_attempts: Number(payload?.max_attempts || 1),
    same_error_repeat_limit: Number(payload?.same_error_repeat_limit || 1),
    wall_clock_timeout_s: Number(payload?.wall_clock_timeout_s || 0),
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
