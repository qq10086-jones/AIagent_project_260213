import fs from "fs";
import path from "path";
import { validateJsonSchemaLite } from "../src/schema_lite_validator.js";
import { loadWorkerCodingTaskClassesOrThrow } from "../src/worker_coding_task_classes.js";

const cohortPlanSchemaPath = path.resolve(process.cwd(), "contracts", "worker_coding_cohort_plan.schema.json");
const cohortPlanPath = path.resolve(process.cwd(), "..", "configs", "registry", "worker_coding_cohort_plan_v1.json");
const betaTemplateRegistryPath = path.resolve(process.cwd(), "..", "configs", "registry", "worker_coding_beta_templates.json");

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function main() {
  try {
    const schema = readJson(cohortPlanSchemaPath);
    const plan = readJson(cohortPlanPath);
    const templateRegistry = readJson(betaTemplateRegistryPath);
    const { taskClassSet: WORKER_CODING_TASK_CLASS_SET } = loadWorkerCodingTaskClassesOrThrow();
    const templatesById = new Map((templateRegistry.templates || []).map((item) => [String(item.template_id || ""), item]));

    const errors = validateJsonSchemaLite(schema, plan);
    if (errors.length > 0) {
      throw new Error(errors.join("; "));
    }

    const seenIds = new Set();
    for (const item of plan.tasks || []) {
      if (seenIds.has(item.cohort_task_id)) {
        throw new Error(`duplicate cohort_task_id: ${item.cohort_task_id}`);
      }
      seenIds.add(item.cohort_task_id);
      if (!WORKER_CODING_TASK_CLASS_SET.has(String(item.task_class || ""))) {
        throw new Error(`unknown task_class in cohort plan: ${item.task_class}`);
      }
      const template = templatesById.get(String(item.beta_template_id || ""));
      if (!template) {
        throw new Error(`unknown beta_template_id in cohort plan: ${item.beta_template_id}`);
      }
      if (String(template.task_class || "") !== String(item.task_class || "")) {
        throw new Error(
          `cohort task '${item.cohort_task_id}' task_class mismatch with template '${item.beta_template_id}': plan='${item.task_class}' template='${template.task_class}'`
        );
      }
    }

    console.log(`[worker-coding-cohort-plan] valid: tasks=${seenIds.size}`);
    console.log(`[worker-coding-cohort-plan] plan=${path.relative(process.cwd(), cohortPlanPath).replace(/\\/g, "/")}`);
  } catch (err) {
    console.error(`[worker-coding-cohort-plan] invalid: ${err.message}`);
    process.exit(1);
  }
}

main();
