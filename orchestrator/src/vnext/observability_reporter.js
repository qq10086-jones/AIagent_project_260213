import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "../schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "..", "contracts", "observability");
const FAILURE_REPORT_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "failure_report_input.schema.json"), "utf8")
);

/**
 * Task 10-03: Formats a progress transition update for Discord.
 */
export function formatTransitionNotification({ previousStep, nextStep }) {
  if (!nextStep) {
    return `✅ Workflow completed successfully. Preparing final artifacts...`;
  }
  
  if (!previousStep) {
    return `🚀 Starting workflow execution. First step: **${nextStep.id}** (${nextStep.role})...`;
  }

  return `✅ Completed **${previousStep.id}**.\n⏳ Transitioning to **${nextStep.id}** (${nextStep.role})...`;
}

/**
 * Task 10-04: Formats a failure report blocking the workflow.
 */
export function formatFailureReport(input) {
  const validation = validateJsonSchemaLite(FAILURE_REPORT_SCHEMA, input, "$");
  if (validation.length > 0) {
    throw new Error(`Invalid failure report input: ${validation.join(", ")}`);
  }

  const { step_id, role, error_code, error_message, raw_logs } = input;

  let report = `❌ **Workflow Halted at Step:** \`${step_id}\`\n`;
  report += `👤 **Role:** \`${role}\`\n`;
  report += `⚠️ **Error Code:** \`${error_code}\`\n\n`;
  report += `**Details:**\n> ${error_message.split('\n').join('\n> ')}\n\n`;

  if (raw_logs) {
    // Sanitize basic secrets (very naive example, actual sanitization would be robust)
    let sanitizedLogs = String(raw_logs).replace(/([a-zA-Z0-9_-]{20,})/g, "***REDACTED***");
    report += `**Diagnostic Logs:**\n\`\`\`text\n${sanitizedLogs}\n\`\`\`\n`;
  }

  return report;
}
