import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

function checkSyntax(root, relPath) {
  try {
    execFileSync(process.execPath, ["--check", path.join(root, relPath)], {
      stdio: "pipe",
    });
    return true;
  } catch (err) {
    if (String(err?.code || "") === "EPERM" || String(err?.message || "").includes("EPERM")) {
      return false;
    }
    throw err;
  }
}

function main() {
  const root = process.cwd();
  const workerText = fs.readFileSync(path.join(root, "worker.js"), "utf8");
  const codingServiceText = fs.readFileSync(path.join(root, "coding_service.js"), "utf8");
  const providerPreflightText = fs.readFileSync(path.join(root, "provider_preflight.js"), "utf8");
  const entrypointText = fs.readFileSync(path.join(root, "entrypoint.sh"), "utf8");
  const opencodeTemplateText = fs.readFileSync(path.join(root, "opencode.json.tpl"), "utf8");

  assert.match(workerText, /import\s+\{\s*validateSafeCommand\s*\}\s+from\s+"\.\/verification_runner\.js"/);
  assert.doesNotMatch(workerText, /import\s+\{[^}]*validateSafeCommand[^}]*\}\s+from\s+"\.\/coding_service\.js"/);
  assert.match(workerText, /export\s+async\s+function\s+main\s*\(/);
  assert.match(workerText, /import\.meta\.url\s*===\s*pathToFileURL\(process\.argv\[1\]\)\.href/);
  assert.match(workerText, /startup-preflight/);

  assert.match(entrypointText, /OPENCODE_PLUGINS_JSON/);
  assert.match(entrypointText, /superpowers@git\+https:\/\/github\.com\/obra\/superpowers\.git/);
  assert.match(opencodeTemplateText, /"plugin"\s*:\s*\$\{OPENCODE_PLUGINS_JSON\}/);

  assert.match(codingServiceText, /from '\.\/scope_guard\.js'/);
  assert.match(codingServiceText, /from '\.\/step_artifact_contract\.js'/, "coding_service must import from step_artifact_contract");
  assert.ok(fs.existsSync(path.join(root, "scope_guard.js")));
  assert.ok(fs.existsSync(path.join(root, "step_artifact_contract.js")), "step_artifact_contract.js must exist");
  assert.match(providerPreflightText, /ALIBABA_AUTH_MISSING/);

  const syntaxChecks = [
    checkSyntax(root, "worker.js"),
    checkSyntax(root, "coding_service.js"),
    checkSyntax(root, "scope_guard.js"),
    checkSyntax(root, "provider_preflight.js"),
    checkSyntax(root, "step_artifact_contract.js"),
  ];
  if (syntaxChecks.some((item) => item === false)) {
    console.log("startup_smoke.test.js: syntax subprocess checks skipped due sandbox EPERM");
  }

  console.log("startup_smoke.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("startup_smoke.test.js: failed");
  console.error(err);
  process.exit(1);
}
