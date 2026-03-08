import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));

function isObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function getDefaultPromptScriptRegistryPath() {
  const candidates = [
    path.resolve(process.cwd(), "configs", "prompt_scripts", "registry.json"),
    path.resolve(process.cwd(), "..", "configs", "prompt_scripts", "registry.json"),
    path.resolve(MODULE_DIR, "..", "..", "configs", "prompt_scripts", "registry.json"),
  ];
  for (const item of candidates) {
    if (fs.existsSync(item)) return item;
  }
  return candidates[0];
}

export function loadPromptScriptRegistryOrThrow(registryPath) {
  const raw = fs.readFileSync(registryPath, "utf8");
  const parsed = JSON.parse(raw);
  const errors = [];

  if (!isObject(parsed)) errors.push("registry root must be an object");
  if (!String(parsed.version || "").trim()) errors.push("missing string field: version");
  if (!isObject(parsed.scripts)) errors.push("missing object field: scripts");
  if (errors.length > 0) {
    throw new Error(`prompt script registry invalid: ${errors.join("; ")}`);
  }

  for (const [scriptId, spec] of Object.entries(parsed.scripts || {})) {
    if (!isObject(spec)) {
      errors.push(`script '${scriptId}' must be an object`);
      continue;
    }
    if (!String(spec.role || "").trim()) errors.push(`script '${scriptId}' missing role`);
    if (Object.prototype.hasOwnProperty.call(spec, "model")) errors.push(`script '${scriptId}' must not contain deprecated model field`);
    if (!String(spec.llm_role || "").trim()) errors.push(`script '${scriptId}' missing llm_role`);
    if (!isObject(spec.input_schema)) errors.push(`script '${scriptId}' missing input_schema`);
    if (!isObject(spec.output_schema)) errors.push(`script '${scriptId}' missing output_schema`);
    if (!Array.isArray(spec.tool_permissions)) errors.push(`script '${scriptId}' missing tool_permissions[]`);
    if (!String(spec.artifact_type || "").trim()) errors.push(`script '${scriptId}' missing artifact_type`);
    if (!isObject(spec.validation)) errors.push(`script '${scriptId}' missing validation`);
  }

  if (errors.length > 0) {
    throw new Error(`prompt script registry invalid: ${errors.join("; ")}`);
  }

  return parsed;
}

export function getPromptScript(registry, scriptId) {
  if (!registry || !scriptId) return null;
  return registry.scripts?.[scriptId] || null;
}

export function validatePromptScriptsAgainstAgents({ promptRegistry, agentRegistry }) {
  const errors = [];
  const scripts = promptRegistry?.scripts || {};
  const agents = agentRegistry?.agents || {};

  for (const [scriptId, spec] of Object.entries(scripts)) {
    const agentId = String(spec.agent_id || "").trim();
    if (!agentId) {
      errors.push(`script '${scriptId}' missing agent_id`);
      continue;
    }
    const agent = agents[agentId];
    if (!agent) {
      errors.push(`script '${scriptId}' references missing agent '${agentId}'`);
      continue;
    }
    if (String(spec.role || "") !== String(agent.role || "")) {
      errors.push(`script '${scriptId}' role '${spec.role}' does not match agent '${agentId}' role '${agent.role}'`);
    }
    if (String(spec.llm_role || "").trim() !== String(spec.role || "").trim()) {
      errors.push(`script '${scriptId}' llm_role '${spec.llm_role}' does not match role '${spec.role}'`);
    }
    const scriptTools = Array.isArray(spec.tool_permissions) ? spec.tool_permissions : [];
    const agentTools = Array.isArray(agent.allowed_tools) ? agent.allowed_tools : [];
    for (const tool of scriptTools) {
      if (!agentTools.includes(tool)) {
        errors.push(`script '${scriptId}' tool '${tool}' not allowed by agent '${agentId}'`);
      }
    }
  }

  return { ok: errors.length === 0, errors };
}
