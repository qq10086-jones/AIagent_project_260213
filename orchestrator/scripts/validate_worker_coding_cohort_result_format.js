import fs from "fs";
import path from "path";
import { validateJsonSchemaLite } from "../src/schema_lite_validator.js";

const schemaPath = path.resolve(process.cwd(), "contracts", "worker_coding_cohort_result.schema.json");

function main() {
  try {
    if (!fs.existsSync(schemaPath)) {
      throw new Error(`schema file not found: ${schemaPath}`);
    }
    const schema = JSON.parse(fs.readFileSync(schemaPath, "utf8"));
    const fixture = {
      cohort_run_id: "wc-cohort-001",
      generated_at: new Date().toISOString(),
      summary: {
        total_runs: 4,
        pass_count: 2,
        fail_count: 1,
        partial_count: 1,
      },
      results: [
        {
          cohort_id: "C-FE-01",
          task_class: "fe_create",
          beta_template_id: "wc.fe_create.v1",
          verification_tier_target: "lint + build",
          verification_tier_achieved: "lint + build",
          result: "pass",
          failure_attribution: "none",
          operator_note: "",
        },
      ],
    };
    const errors = validateJsonSchemaLite(schema, fixture);
    if (errors.length > 0) {
      throw new Error(errors.join("; "));
    }
    console.log("[worker-coding-cohort-result] valid schema and fixture");
  } catch (err) {
    console.error(`[worker-coding-cohort-result] invalid: ${err.message}`);
    process.exit(1);
  }
}

main();
