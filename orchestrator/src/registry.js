import fs from "fs";
import path from "path";

function readJsonFile(filePath) {
  const raw = fs.readFileSync(filePath, "utf8");
  return JSON.parse(raw);
}

function isObject(v) {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

function valueType(v) {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  return typeof v;
}

function validateToolParams(schema, payload) {
  if (!isObject(schema)) return [];
  const errors = [];
  const required = Array.isArray(schema.required) ? schema.required : [];
  const properties = isObject(schema.properties) ? schema.properties : {};

  for (const field of required) {
    if (payload[field] === undefined || payload[field] === null || payload[field] === "") {
      errors.push(`missing required param '${field}'`);
    }
  }

  for (const [key, rule] of Object.entries(properties)) {
    if (!isObject(rule) || !rule.type || payload[key] === undefined || payload[key] === null) continue;
    const got = valueType(payload[key]);
    if (got !== rule.type) {
      errors.push(`param '${key}' type mismatch: expected '${rule.type}', got '${got}'`);
    }
  }
  return errors;
}

export function loadRegistryOrThrow(registryPath) {
  const reg = readJsonFile(registryPath);
  const errors = [];

  if (!isObject(reg)) errors.push("registry root must be an object");
  if (!reg.version || typeof reg.version !== "string") errors.push("missing string field: version");
  if (!isObject(reg.project_types)) errors.push("missing object field: project_types");
  if (!isObject(reg.roles)) errors.push("missing object field: roles");
  if (!isObject(reg.tools)) errors.push("missing object field: tools");
  if (!isObject(reg.workflows)) errors.push("missing object field: workflows");
  if (!isObject(reg.acceptance_suites)) errors.push("missing object field: acceptance_suites");

  if (errors.length) {
    throw new Error(`registry invalid: ${errors.join("; ")}`);
  }

  for (const [role, spec] of Object.entries(reg.roles)) {
    if (!Array.isArray(spec?.tools)) {
      errors.push(`role '${role}' must define tools[]`);
      continue;
    }
    for (const tool of spec.tools) {
      if (!reg.tools[tool]) errors.push(`role '${role}' references unknown tool '${tool}'`);
    }
  }

  for (const [workflowId, wf] of Object.entries(reg.workflows)) {
    if (!reg.project_types[wf?.project_type]) {
      errors.push(`workflow '${workflowId}' references unknown project_type '${wf?.project_type}'`);
    }
    if (!Array.isArray(wf?.steps) || wf.steps.length === 0) {
      errors.push(`workflow '${workflowId}' must define non-empty steps[]`);
      continue;
    }
    const stepIds = new Set();
    for (const step of wf.steps) {
      if (!String(step?.id || "").trim()) continue;
      if (stepIds.has(step.id)) {
        errors.push(`workflow '${workflowId}' contains duplicate step id '${step.id}'`);
      }
      stepIds.add(step.id);
    }
    for (const step of wf.steps) {
      if (!reg.roles[step.role]) errors.push(`workflow '${workflowId}' step '${step.id}' references unknown role '${step.role}'`);
      if (!reg.tools[step.tool]) errors.push(`workflow '${workflowId}' step '${step.id}' references unknown tool '${step.tool}'`);
      if (reg.roles[step.role] && !reg.roles[step.role].tools.includes(step.tool)) {
        errors.push(`workflow '${workflowId}' step '${step.id}' tool '${step.tool}' not allowed for role '${step.role}'`);
      }
      if (step.prompt_script_id && typeof step.prompt_script_id !== "string") {
        errors.push(`workflow '${workflowId}' step '${step.id}' prompt_script_id must be a string`);
      }
      if (step.depends_on !== undefined && !Array.isArray(step.depends_on)) {
        errors.push(`workflow '${workflowId}' step '${step.id}' depends_on must be an array`);
      }
      if (Array.isArray(step.depends_on)) {
        const seenDependencies = new Set();
        for (const dependencyId of step.depends_on) {
          if (typeof dependencyId !== "string" || dependencyId.trim() === "") {
            errors.push(`workflow '${workflowId}' step '${step.id}' depends_on must contain non-empty strings`);
            continue;
          }
          if (dependencyId === step.id) {
            errors.push(`workflow '${workflowId}' step '${step.id}' cannot depend on itself`);
          }
          if (seenDependencies.has(dependencyId)) {
            errors.push(`workflow '${workflowId}' step '${step.id}' has duplicate dependency '${dependencyId}'`);
          }
          seenDependencies.add(dependencyId);
          if (!stepIds.has(dependencyId)) {
            errors.push(`workflow '${workflowId}' step '${step.id}' depends_on unknown step '${dependencyId}'`);
          }
        }
      }
    }
  }

  for (const [projectType, pt] of Object.entries(reg.project_types)) {
    if (pt.default_workflow && !reg.workflows[pt.default_workflow]) {
      errors.push(`project_type '${projectType}' default_workflow '${pt.default_workflow}' not found`);
    }
    if (pt.acceptance_suite && !reg.acceptance_suites[pt.acceptance_suite]) {
      errors.push(`project_type '${projectType}' acceptance_suite '${pt.acceptance_suite}' not found`);
    }
  }

  if (errors.length) {
    throw new Error(`registry invalid: ${errors.join("; ")}`);
  }

  return reg;
}

export function getDefaultRegistryPath() {
  const candidates = [
    path.join(process.cwd(), "configs", "registry", "capability_registry.json"),
    path.join(process.cwd(), "..", "configs", "registry", "capability_registry.json"),
    path.join(process.cwd(), "configs", "capability_registry.json"),
    path.join(process.cwd(), "..", "configs", "capability_registry.json"),
  ];
  const normalized = Array.from(new Set(candidates.map((item) => path.resolve(item))));
  for (const p of normalized) {
    if (fs.existsSync(p)) return p;
  }
  return normalized[0];
}

export function validateTaskInputAgainstRegistry({ registry, tool_name, payload }) {
  const errors = [];
  const body = isObject(payload) ? payload : {};
  const projectType = body.project_type;
  const workflowId = body.workflow_id;
  const role = body.role || body.role_name;
  const stepId = body.step_id;
  const toolSpec = registry.tools?.[tool_name];

  if (!toolSpec) {
    errors.push(`tool '${tool_name}' not found in registry`);
    return { ok: false, errors };
  }

  if (projectType && !registry.project_types?.[projectType]) {
    errors.push(`project_type '${projectType}' not found`);
  }

  if (workflowId) {
    const wf = registry.workflows?.[workflowId];
    if (!wf) {
      errors.push(`workflow '${workflowId}' not found`);
    } else {
      if (projectType && wf.project_type && wf.project_type !== projectType) {
        errors.push(`workflow '${workflowId}' project_type mismatch: expected '${wf.project_type}'`);
      }
      if (role && !wf.steps.some((s) => s.role === role && s.tool === tool_name)) {
        errors.push(`workflow '${workflowId}' has no step for role '${role}' with tool '${tool_name}'`);
      }
      if (!role && !wf.steps.some((s) => s.tool === tool_name)) {
        errors.push(`workflow '${workflowId}' has no step using tool '${tool_name}'`);
      }
      if (stepId && !wf.steps.some((s) => s.id === stepId)) {
        errors.push(`workflow '${workflowId}' has no step '${stepId}'`);
      }
    }
  }

  if (role) {
    const r = registry.roles?.[role];
    if (!r) {
      errors.push(`role '${role}' not found`);
    } else if (!Array.isArray(r.tools) || !r.tools.includes(tool_name)) {
      errors.push(`tool '${tool_name}' is not allowed for role '${role}'`);
    }
  }

  const paramErrors = validateToolParams(toolSpec.params, body);
  errors.push(...paramErrors);
  return { ok: errors.length === 0, errors };
}
