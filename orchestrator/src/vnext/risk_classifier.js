import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "../schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "..", "contracts", "guardrails");
const RISK_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "risk_classification.schema.json"), "utf8")
);

const HIGH_RISK_KEYWORDS = [
  "rm -rf", "delete", "drop table", "truncate", "secrets", ".env", "aws_access_key",
  "production deployment", "publish", "market impact", "buy", "sell"
];

const MEDIUM_RISK_KEYWORDS = [
  "write file", "rewrite", "refactor all", "patch", "git commit", "git push"
];

/**
 * Classifies the risk of an intent, tool, or raw text string.
 * @param {Object} params
 * @param {string} params.intent The high level intent (e.g. 'coding', 'chat')
 * @param {string} params.tool_name The tool requested (e.g. 'bash.execute')
 * @param {string} params.raw_input The raw command or description
 * @returns {Object} RiskClassificationResult
 */
export function classifyRisk({ intent = "", tool_name = "", raw_input = "" }) {
  let riskLevel = "low";
  let requiresApproval = false;
  const triggers = [];

  const textToAnalyze = String(raw_input || "").toLowerCase();

  // Rule 1: High risk tool invocations
  if (tool_name === "bash.execute" || tool_name === "broker.trade" || tool_name === "system.config") {
    // If it's bash but just 'ls' or 'cat', it might be low, but we'll flag 'rm'
    for (const kw of HIGH_RISK_KEYWORDS) {
      if (textToAnalyze.includes(kw)) {
        riskLevel = "high";
        requiresApproval = true;
        triggers.push({
          rule_id: "HIGH_RISK_KEYWORD",
          description: `Detected high-risk keyword: ${kw}`
        });
      }
    }
  }

  // Rule 2: Quant Execution with market impact is always high risk
  if (intent === "quant" && tool_name === "broker.trade") {
    riskLevel = "high";
    requiresApproval = true;
    triggers.push({
      rule_id: "QUANT_TRADE_EXECUTION",
      description: "Financial actions strictly require human approval"
    });
  }

  // Rule 3: Medium Risk fallback
  if (riskLevel === "low") {
    for (const kw of MEDIUM_RISK_KEYWORDS) {
      if (textToAnalyze.includes(kw)) {
        riskLevel = "medium";
        // Medium risk does not strictly require approval in local mode, but is flagged
        triggers.push({
          rule_id: "MEDIUM_RISK_KEYWORD",
          description: `Detected medium-risk keyword: ${kw}`
        });
      }
    }
  }

  const result = {
    risk_level: riskLevel,
    requires_approval: requiresApproval,
    triggers
  };

  const validation = validateJsonSchemaLite(RISK_SCHEMA, result, "$");
  if (validation.length > 0) {
    throw new Error(`Invalid risk classification result: ${validation.join(", ")}`);
  }

  return result;
}
