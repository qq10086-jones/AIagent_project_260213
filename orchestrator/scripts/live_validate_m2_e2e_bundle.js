import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const baseUrlArgIndex = process.argv.indexOf("--base-url");
const approvalTokenArgIndex = process.argv.indexOf("--approval-token");
const forwardedArgs = [];

if (baseUrlArgIndex >= 0 && process.argv[baseUrlArgIndex + 1]) {
  forwardedArgs.push("--base-url", process.argv[baseUrlArgIndex + 1]);
}
if (approvalTokenArgIndex >= 0 && process.argv[approvalTokenArgIndex + 1]) {
  forwardedArgs.push("--approval-token", process.argv[approvalTokenArgIndex + 1]);
}

const steps = [
  {
    name: "live_vnext_runtime",
    script: path.join(__dirname, "live_validate_vnext_runtime.js"),
  },
  {
    name: "live_workflow_runtime",
    script: path.join(__dirname, "live_validate_workflow_runtime.js"),
  },
];

let failed = false;

for (const step of steps) {
  console.log(`# Running ${step.name}`);
  const result = spawnSync(process.execPath, [step.script, ...forwardedArgs], {
    stdio: "inherit",
    env: process.env,
  });
  if ((result.status || 0) !== 0) {
    failed = true;
  }
}

if (failed) {
  process.exit(1);
}
