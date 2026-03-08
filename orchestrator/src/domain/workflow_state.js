/**
 * workflow_state.js
 *
 * Step status normalisation, step contract definitions, and prompt-building
 * helpers.  Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import path from "path";

export function normalizeStepStatus(status) {
  const s = String(status || "").toLowerCase();
  if (["pending", "queued", "waiting_approval", "running", "succeeded", "failed", "partial_failure"].includes(s)) return s;
  return "pending";
}

export const STEP_CONTRACTS = {
  pm_spec: {
    title: "PM Specification",
    required_artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    instructions: [
      "Define user stories, scope boundaries, and non-goals for a minimal CRM web app.",
      "Write measurable acceptance criteria in plan/acceptance.json.",
      "Create phased milestones in plan/milestones.md.",
    ],
  },
  arch_design: {
    title: "Architecture Design",
    required_artifacts: ["plan/arch.md", "plan/interfaces.md", "risk/risk_report.json", "plan/workplan.md"],
    instructions: [
      "Provide architecture decisions with tradeoffs and module boundaries.",
      "Publish top risks and mitigations in risk/risk_report.json.",
      "Split implementation work for FE/BE/QA in plan/workplan.md.",
      "Define all API endpoints or internal interfaces in plan/interfaces.md.",
    ],
  },
  impl_fe: {
    title: "Frontend Implementation",
    required_artifacts: ["impl/fe_changes/app.js", "impl/fe_notes.md"],
    instructions: [
      "Implement frontend outputs as complete files under impl/fe_changes/.",
      "Consume handoff/be_to_fe.json as the backend contract source for API usage.",
      "Write impl/fe_notes.md with UI decisions, assumptions, and run instructions.",
    ],
  },
  impl_be: {
    title: "Backend Implementation",
    required_artifacts: ["impl/be_changes/server.js", "impl/be_notes.md", "handoff/be_to_fe.json"],
    instructions: [
      "Implement backend/API/data layer outputs as complete files under impl/be_changes/.",
      "Write impl/be_notes.md with implementation decisions, assumptions, and run instructions.",
      "Emit handoff/be_to_fe.json with api_contracts, shared_types, and explicit scope_constraints.",
    ],
  },
  qa_verify: {
    title: "QA Verification",
    required_artifacts: ["verify/qa_report.json"],
    instructions: [
      "Run deterministic checks on required artifacts and typed handoffs.",
      "Run semantic checks against backend/frontend consistency and acceptance criteria.",
      "Publish the result in verify/qa_report.json.",
    ],
  },
  release_pack: {
    title: "Release Pack",
    required_artifacts: ["release/release_notes.md", "release/artifact_manifest.json"],
    instructions: [
      "Assemble release/release_notes.md as the human-readable package summary.",
      "Assemble release/artifact_manifest.json as the machine-readable package manifest.",
      "Ensure artifact references are complete and traceable.",
    ],
  },
};

export function buildStepPrompt({ run, stepDef, input, payload, promptScript = null }) {
  const c = STEP_CONTRACTS[stepDef.id] || null;
  const goal = String(input.goal || input.task_prompt || input.prompt || "Build a minimal CRM web app").trim();
  const title = c?.title || stepDef.id;
  const required = Array.isArray(c?.required_artifacts) ? c.required_artifacts : [];
  const lines = Array.isArray(c?.instructions) ? c.instructions : [];
  const artifactRoot = String(payload.artifact_root || "");
  const artifactAbs = path.posix.join("/workspace", artifactRoot).replace(/\/+/g, "/");

  const outputReq = required.length > 0
    ? `Required artifacts (relative to ${artifactRoot}):\n- ${required.join("\n- ")}`
    : "Required artifacts: follow workflow step contract.";
  const handoffReq = payload?.handoff_contract_out
    ? [
        `Handoff artifacts for next stage:`,
        `- ${Array.isArray(payload.handoff_contract_out.required_artifacts) ? payload.handoff_contract_out.required_artifacts.join("\n- ") : ""}`,
        payload.handoff_contract_out.typed_handoff?.file
          ? `- typed handoff manifest: ${payload.handoff_contract_out.typed_handoff.file}`
          : "",
      ].filter(Boolean).join("\n")
    : "";
  const guidance = lines.length > 0
    ? `Execution requirements:\n- ${lines.join("\n- ")}`
    : "Execution requirements: complete this step with verifiable outputs.";
  const promptScriptNote = promptScript
    ? [
        `Prompt Script ID: ${promptScript.script_id}`,
        `Prompt Script LLM Role: ${promptScript.llm_role || promptScript.role || ""}`,
        `Prompt Script Artifact Type: ${promptScript.artifact_type}`,
        `Prompt Script Validation: ${JSON.stringify(promptScript.validation || {})}`,
        promptScript.system_prompt ? `Prompt Script Goal: ${promptScript.system_prompt}` : "",
      ].filter(Boolean).join("\n")
    : "";
  const executionAdapterNote = payload?.execution_adapter_packet
    ? [
        `Execution Adapter: ${payload.execution_adapter_packet.adapter_id}`,
        `Execution Target Paths: ${Array.isArray(payload.execution_adapter_packet.target_paths) ? payload.execution_adapter_packet.target_paths.join(", ") : ""}`,
        `Execution Required Outputs: ${Array.isArray(payload.execution_adapter_packet.required_outputs) ? payload.execution_adapter_packet.required_outputs.join(", ") : ""}`,
        payload.tool_adapter_request?.adapter_type
          ? `Tool Adapter Request: ${payload.tool_adapter_request.adapter_type}/${payload.tool_adapter_request.provider}`
          : "",
      ].filter(Boolean).join("\n")
    : "";
  return [
    `[CodingTeam Step] ${title}`,
    `Workflow: ${run.workflow_id}`,
    `Project Type: ${run.project_type}`,
    `Step ID: ${stepDef.id}`,
    `Role: ${stepDef.role}`,
    `Goal: ${goal}`,
    promptScriptNote,
    executionAdapterNote,
    guidance,
    outputReq,
    handoffReq,
    `Absolute artifact output root: ${artifactAbs}`,
    "Write files under the artifact output root exactly with the required relative paths.",
    "Constraints:",
    "- Prefer small, reviewable changes.",
    "- Keep outputs deterministic and explicit.",
    "- Include concise validation evidence.",
  ].join("\n");
}

export function validatePromptScriptBinding({ stepDef, promptScriptRegistry, promptScript }) {
  const promptScriptId = String(stepDef?.prompt_script_id || "").trim();
  if (!promptScriptId) {
    return { ok: true, prompt_script_id: "" };
  }
  if (!promptScriptRegistry || !promptScriptRegistry.scripts) {
    return { ok: false, code: "PROMPT_SCRIPT_REGISTRY_MISSING", detail: "prompt script registry not loaded" };
  }
  if (!promptScript) {
    return { ok: false, code: "PROMPT_SCRIPT_NOT_FOUND", detail: `prompt script '${promptScriptId}' not found` };
  }
  if (String(promptScript.role || "") !== String(stepDef.role || "")) {
    return {
      ok: false,
      code: "PROMPT_SCRIPT_ROLE_MISMATCH",
      detail: `prompt script '${promptScriptId}' role '${promptScript.role}' does not match step role '${stepDef.role}'`,
    };
  }
  return { ok: true, prompt_script_id: promptScriptId };
}
