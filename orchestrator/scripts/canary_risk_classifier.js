import fs from "fs";
import path from "path";
import { classifyRisk } from "../src/vnext/risk_classifier.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const lowRisk = classifyRisk({
    intent: "chat",
    tool_name: "none",
    raw_input: "Hello how are you?"
  });
  assertEqual(lowRisk.risk_level, "low", "chat risk");
  assertEqual(lowRisk.requires_approval, false, "chat approval");

  const mediumRisk = classifyRisk({
    intent: "coding",
    tool_name: "file.write",
    raw_input: "write file index.js"
  });
  assertEqual(mediumRisk.risk_level, "medium", "write file risk");
  assertEqual(mediumRisk.requires_approval, false, "write file approval");

  const highRiskDelete = classifyRisk({
    intent: "coding",
    tool_name: "bash.execute",
    raw_input: "rm -rf /"
  });
  assertEqual(highRiskDelete.risk_level, "high", "rm risk");
  assertEqual(highRiskDelete.requires_approval, true, "rm approval");

  const highRiskTrade = classifyRisk({
    intent: "quant",
    tool_name: "broker.trade",
    raw_input: "buy 100 shares of AAPL"
  });
  assertEqual(highRiskTrade.risk_level, "high", "trade risk");
  assertEqual(highRiskTrade.requires_approval, true, "trade approval");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "risk_classifier");
  fs.mkdirSync(outDir, { recursive: true });
  
  const reportPath = path.join(outDir, "risk_classifier_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    results: {
      lowRisk,
      mediumRisk,
      highRiskDelete,
      highRiskTrade
    }
  }, null, 2), "utf8");

  console.log("# Risk Classifier Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
