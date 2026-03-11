import fs from "fs";
import path from "path";
import { loadWorkerCodingTaskClassesOrThrow } from "./worker_coding_task_classes.js";

const { taskClassSet: WORKER_CODING_TASK_CLASS_SET } = loadWorkerCodingTaskClassesOrThrow();

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

export function getDefaultWorkerCodingTemplateRegistryPath() {
  const candidates = [
    path.resolve(process.cwd(), "..", "configs", "registry", "worker_coding_beta_templates.json"),
    path.resolve(process.cwd(), "configs", "registry", "worker_coding_beta_templates.json"),
  ];
  for (const item of candidates) {
    if (fs.existsSync(item)) return item;
  }
  return candidates[0];
}

export function loadWorkerCodingTemplateRegistryOrThrow(filePath = getDefaultWorkerCodingTemplateRegistryPath()) {
  const registry = readJson(filePath);
  if (!registry || typeof registry !== "object" || Array.isArray(registry)) {
    throw new Error("worker coding template registry must be an object");
  }
  if (!Array.isArray(registry.templates)) {
    throw new Error("worker coding template registry must define templates[]");
  }
  const templatesById = new Map();
  for (const template of registry.templates) {
    const templateId = String(template?.template_id || "").trim();
    if (!templateId) {
      throw new Error("worker coding template missing template_id");
    }
    if (templatesById.has(templateId)) {
      throw new Error(`duplicate worker coding template_id '${templateId}'`);
    }
    templatesById.set(templateId, template);
  }
  return {
    path: filePath,
    version: String(registry.version || ""),
    templates: registry.templates,
    templatesById,
  };
}

export function applyWorkerCodingTemplateDefaults({ payload, templateRegistry, stepDef }) {
  if (!payload || typeof payload !== "object") return payload;
  if (String(stepDef?.tool || "") !== "coding.delegate") return payload;

  const templateId = String(payload.beta_template_id || "").trim();
  if (!templateId) return payload;

  const template = templateRegistry?.templatesById?.get(templateId);
  if (!template) {
    const err = new Error(`unknown worker coding beta template '${templateId}'`);
    err.code = "WORKER_CODING_BETA_TEMPLATE_UNKNOWN";
    throw err;
  }
  const templateTaskClass = String(template.task_class || "").trim().toLowerCase();
  if (!WORKER_CODING_TASK_CLASS_SET.has(templateTaskClass)) {
    const err = new Error(`worker coding beta template '${templateId}' has unknown task_class '${template.task_class}'`);
    err.code = "WORKER_CODING_BETA_TEMPLATE_INVALID_TASK_CLASS";
    throw err;
  }
  if (payload.task_class && String(payload.task_class).trim().toLowerCase() !== templateTaskClass) {
    const err = new Error(
      `worker coding beta template '${templateId}' task_class mismatch: payload='${payload.task_class}' template='${template.task_class}'`
    );
    err.code = "WORKER_CODING_BETA_TEMPLATE_TASK_CLASS_MISMATCH";
    throw err;
  }

  if (!payload.task_class) payload.task_class = templateTaskClass;
  if (!payload.context_envelope) payload.context_envelope = template.context_envelope || null;
  if (!Array.isArray(payload.target_path_hints) || payload.target_path_hints.length === 0) {
    payload.target_path_hints = Array.isArray(template.target_path_hints) ? template.target_path_hints : [];
  }
  if (!Array.isArray(payload.template_verification_tiers) || payload.template_verification_tiers.length === 0) {
    payload.template_verification_tiers = Array.isArray(template.verification_tiers) ? template.verification_tiers : [];
  }
  if (!Array.isArray(payload.human_acceptance_criteria) || payload.human_acceptance_criteria.length === 0) {
    payload.human_acceptance_criteria = Array.isArray(template.human_acceptance_criteria) ? template.human_acceptance_criteria : [];
  }
  if (!Array.isArray(payload.summary_expectations) || payload.summary_expectations.length === 0) {
    payload.summary_expectations = Array.isArray(template.summary_expectations) ? template.summary_expectations : [];
  }
  return payload;
}
