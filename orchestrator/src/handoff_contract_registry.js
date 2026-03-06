import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));

export function getDefaultHandoffContractPath() {
  const candidates = [
    path.resolve(process.cwd(), "configs", "contracts", "coding_team_v0_handoffs.json"),
    path.resolve(process.cwd(), "..", "orchestrator", "configs", "contracts", "coding_team_v0_handoffs.json"),
    path.resolve(MODULE_DIR, "..", "configs", "contracts", "coding_team_v0_handoffs.json"),
  ];
  for (const item of candidates) {
    if (fs.existsSync(item)) return item;
  }
  return candidates[0];
}

export function loadHandoffContractsOrThrow(contractPath) {
  const raw = fs.readFileSync(contractPath, "utf8");
  const parsed = JSON.parse(raw);
  const errors = [];
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    errors.push("handoff contract root must be an object");
  }
  if (!String(parsed.workflow_id || "").trim()) errors.push("workflow_id is required");
  if (!parsed.handoffs || typeof parsed.handoffs !== "object" || Array.isArray(parsed.handoffs)) {
    errors.push("handoffs must be an object");
  }
  for (const [handoffId, spec] of Object.entries(parsed.handoffs || {})) {
    if (!spec || typeof spec !== "object" || Array.isArray(spec)) {
      errors.push(`handoff '${handoffId}' must be an object`);
      continue;
    }
    if (!String(spec.from_step || "").trim()) errors.push(`handoff '${handoffId}' missing from_step`);
    if (!Array.isArray(spec.to_steps)) errors.push(`handoff '${handoffId}' missing to_steps[]`);
    if (!Array.isArray(spec.required_artifacts)) errors.push(`handoff '${handoffId}' missing required_artifacts[]`);
    if (!Array.isArray(spec.required_sections)) errors.push(`handoff '${handoffId}' missing required_sections[]`);
    if (spec.typed_handoff != null) {
      if (!spec.typed_handoff || typeof spec.typed_handoff !== "object" || Array.isArray(spec.typed_handoff)) {
        errors.push(`handoff '${handoffId}' typed_handoff must be an object`);
      } else {
        if (!String(spec.typed_handoff.file || "").trim()) errors.push(`handoff '${handoffId}' typed_handoff.file missing`);
        if (!Array.isArray(spec.typed_handoff.required_fields)) {
          errors.push(`handoff '${handoffId}' typed_handoff.required_fields missing`);
        }
      }
    }
  }
  if (errors.length > 0) {
    throw new Error(`handoff contracts invalid: ${errors.join("; ")}`);
  }
  return parsed;
}
