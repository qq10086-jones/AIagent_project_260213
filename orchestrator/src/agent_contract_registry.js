import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));

function isObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function getDefaultAgentRegistryDir() {
  const candidates = [
    path.resolve(process.cwd(), "configs", "agents"),
    path.resolve(process.cwd(), "..", "configs", "agents"),
    path.resolve(MODULE_DIR, "..", "..", "configs", "agents"),
  ];
  for (const item of candidates) {
    if (fs.existsSync(item)) return item;
  }
  return candidates[0];
}

export function loadAgentContractsOrThrow(agentDirPath) {
  if (!fs.existsSync(agentDirPath)) {
    throw new Error(`agent contract dir not found: ${agentDirPath}`);
  }
  const files = fs.readdirSync(agentDirPath).filter((item) => item.endsWith(".json"));
  const errors = [];
  const agents = {};

  for (const file of files) {
    const abs = path.join(agentDirPath, file);
    const parsed = JSON.parse(fs.readFileSync(abs, "utf8"));
    if (!isObject(parsed)) {
      errors.push(`${file}: agent contract must be an object`);
      continue;
    }
    if (!String(parsed.agent_id || "").trim()) errors.push(`${file}: missing agent_id`);
    if (!String(parsed.role || "").trim()) errors.push(`${file}: missing role`);
    if (!String(parsed.mission || "").trim()) errors.push(`${file}: missing mission`);
    if (!isObject(parsed.input_schema)) errors.push(`${file}: missing input_schema`);
    if (!isObject(parsed.output_schema)) errors.push(`${file}: missing output_schema`);
    if (!Array.isArray(parsed.allowed_tools)) errors.push(`${file}: missing allowed_tools[]`);
    if (!Array.isArray(parsed.forbidden_actions)) errors.push(`${file}: missing forbidden_actions[]`);
    if (!Array.isArray(parsed.SOP)) errors.push(`${file}: missing SOP[]`);
    if (String(parsed.agent_id || "").trim()) {
      agents[parsed.agent_id] = parsed;
    }
  }

  if (Object.keys(agents).length === 0) {
    errors.push("no agent contracts found");
  }
  if (errors.length > 0) {
    throw new Error(`agent contract registry invalid: ${errors.join("; ")}`);
  }
  return { version: "1.0.0", agents };
}
