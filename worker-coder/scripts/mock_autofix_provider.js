import fs from "node:fs";
import path from "node:path";

const prompt = String(process.argv[2] || "");
const workspaceRoot = process.cwd();
const targetFile = path.join(workspaceRoot, "sandbox", "crm_site", "app.js");

fs.mkdirSync(path.dirname(targetFile), { recursive: true });

if (prompt.includes("[Auto-Fix Retry]")) {
  fs.writeFileSync(targetFile, "const status = 'fixed';\nmodule.exports = { status };\n", "utf8");
  console.log("mock autofix provider: repaired file");
  process.exit(0);
}

fs.writeFileSync(targetFile, "const status = ;\nmodule.exports = { status };\n", "utf8");
console.log("mock autofix provider: wrote broken file");
process.exit(0);
