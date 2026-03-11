import fs from "fs";
import path from "path";
import { validateJsonSchemaLite } from "../src/schema_lite_validator.js";
import { loadWorkerCodingTaskClassesOrThrow } from "../src/worker_coding_task_classes.js";

const ROOT = path.resolve(process.cwd(), "..");
const taskContractSchemaPath = path.resolve(process.cwd(), "contracts", "worker_coding_task_contract.schema.json");
const betaTemplateSchemaPath = path.resolve(process.cwd(), "contracts", "worker_coding_beta_template_registry.schema.json");
const betaTemplateRegistryPath = path.resolve(ROOT, "configs", "registry", "worker_coding_beta_templates.json");

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function assertFile(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`file not found: ${filePath}`);
  }
}

function main() {
  try {
    assertFile(taskContractSchemaPath);
    assertFile(betaTemplateSchemaPath);
    assertFile(betaTemplateRegistryPath);

    const taskContractSchema = readJson(taskContractSchemaPath);
    const betaTemplateSchema = readJson(betaTemplateSchemaPath);
    const betaTemplateRegistry = readJson(betaTemplateRegistryPath);
    const { taskClasses: WORKER_CODING_TASK_CLASSES, taskClassSet: WORKER_CODING_TASK_CLASS_SET } = loadWorkerCodingTaskClassesOrThrow();

    const registryErrors = validateJsonSchemaLite(betaTemplateSchema, betaTemplateRegistry);
    if (registryErrors.length > 0) {
      throw new Error(`beta template registry invalid: ${registryErrors.join("; ")}`);
    }

    const seenTemplateIds = new Set();
    for (const template of betaTemplateRegistry.templates || []) {
      if (!WORKER_CODING_TASK_CLASS_SET.has(String(template.task_class || ""))) {
        throw new Error(`unknown task_class in template '${template.template_id}': ${template.task_class}`);
      }
      if (seenTemplateIds.has(template.template_id)) {
        throw new Error(`duplicate template_id: ${template.template_id}`);
      }
      seenTemplateIds.add(template.template_id);
      const contractErrors = validateJsonSchemaLite(taskContractSchema, {
        task_class: template.task_class,
        beta_template_id: template.template_id,
        context_envelope: template.context_envelope,
      });
      if (contractErrors.length > 0) {
        throw new Error(`template '${template.template_id}' task contract invalid: ${contractErrors.join("; ")}`);
      }
    }

    console.log(`[worker-coding-contract] valid: templates=${seenTemplateIds.size}`);
    console.log(`[worker-coding-contract] registry=${path.relative(process.cwd(), betaTemplateRegistryPath).replace(/\\/g, "/")}`);
    console.log(`[worker-coding-contract] task classes=${WORKER_CODING_TASK_CLASSES.join(",")}`);
  } catch (err) {
    console.error(`[worker-coding-contract] invalid: ${err.message}`);
    process.exit(1);
  }
}

main();
