import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const RUNNER_SCRIPT = path.resolve(MODULE_DIR, "..", "scripts", "run_acceptance_test.mjs");

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-accept-test-"));
}

function writeFile(root, rel, content) {
  const abs = path.join(root, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, typeof content === "string" ? content : JSON.stringify(content, null, 2), "utf8");
}

function readJson(absPath) {
  return JSON.parse(fs.readFileSync(absPath, "utf8"));
}

function runAcceptanceTest(workspaceRoot, artifactRoot, { expectNonZero = false } = {}) {
  const relRoot = path.relative(workspaceRoot, path.resolve(workspaceRoot, artifactRoot));
  try {
    execSync(`node "${RUNNER_SCRIPT}" --artifact-root "${relRoot}"`, {
      cwd: workspaceRoot,
      timeout: 15000,
      stdio: "pipe",
    });
  } catch (err) {
    if (!expectNonZero) throw err;
    // runner exited non-zero (fail-closed), result file should still exist
  }
  return readJson(path.join(workspaceRoot, artifactRoot, "verify", "acceptance_result.json"));
}

test("acceptance test: all deterministic criteria pass", () => {
  const ws = makeWorkspace();
  const artRoot = "runtime/artifacts/release/test-run";
  writeFile(ws, `${artRoot}/plan/acceptance.json`, {
    criteria: [
      { id: "AC-1", description: "echo test passes", verify_command: "echo hello", verify_tier: "deterministic" },
      { id: "AC-2", description: "echo with pattern", verify_command: "echo foobar", expected_pattern: "foo", verify_tier: "deterministic" },
    ],
    artifacts: ["impl/be_changes/server.js"],
    owner: "pm",
    version: "1.0",
  });

  const result = runAcceptanceTest(ws, artRoot);
  assert.equal(result.verdict, "pass");
  assert.equal(result.deterministic_pass, 2);
  assert.equal(result.deterministic_fail, 0);
  assert.equal(result.criteria_results.length, 2);
  assert.equal(result.criteria_results[0].status, "pass");
  assert.equal(result.criteria_results[1].status, "pass");
});

test("acceptance test: deterministic criterion fails", () => {
  const ws = makeWorkspace();
  const artRoot = "runtime/artifacts/release/fail-run";
  writeFile(ws, `${artRoot}/plan/acceptance.json`, {
    criteria: [
      { id: "AC-1", description: "should fail", verify_command: "echo hello", expected_pattern: "^NOTFOUND$", verify_tier: "deterministic" },
    ],
    artifacts: ["x"],
    owner: "pm",
    version: "1.0",
  });

  const result = runAcceptanceTest(ws, artRoot);
  assert.equal(result.verdict, "fail");
  assert.equal(result.deterministic_fail, 1);
  assert.equal(result.criteria_results[0].status, "fail");
});

test("acceptance test: legacy string criteria are deferred to semantic", () => {
  const ws = makeWorkspace();
  const artRoot = "runtime/artifacts/release/legacy-run";
  writeFile(ws, `${artRoot}/plan/acceptance.json`, {
    criteria: ["User can create a customer", "API returns JSON"],
    artifacts: ["x"],
    owner: "pm",
    version: "1.0",
  });

  const result = runAcceptanceTest(ws, artRoot);
  assert.equal(result.verdict, "no_deterministic_criteria");
  assert.equal(result.semantic_pending, 2);
  assert.equal(result.criteria_results[0].verify_tier, "semantic");
  assert.equal(result.criteria_results[0].status, "skip");
});

test("acceptance test: mixed criteria — deterministic pass + semantic skip", () => {
  const ws = makeWorkspace();
  const artRoot = "runtime/artifacts/release/mixed-run";
  writeFile(ws, `${artRoot}/plan/acceptance.json`, {
    criteria: [
      { id: "AC-1", description: "echo works", verify_command: "echo ok", verify_tier: "deterministic" },
      { id: "AC-2", description: "code review needed", verify_tier: "semantic" },
    ],
    artifacts: ["x"],
    owner: "pm",
    version: "1.0",
  });

  const result = runAcceptanceTest(ws, artRoot);
  assert.equal(result.verdict, "pass");
  assert.equal(result.deterministic_pass, 1);
  assert.equal(result.semantic_pending, 1);
});

test("acceptance test: no acceptance.json fails closed", () => {
  const ws = makeWorkspace();
  const artRoot = "runtime/artifacts/release/empty-run";
  fs.mkdirSync(path.join(ws, artRoot), { recursive: true });

  const result = runAcceptanceTest(ws, artRoot, { expectNonZero: true });
  assert.equal(result.verdict, "fail");
  assert.match(result.runner_error, /not found|unreadable/);
  assert.equal(result.criteria_results.length, 0);
});

test("acceptance test: invalid expected_pattern fails closed (not auto-pass)", () => {
  const ws = makeWorkspace();
  const artRoot = "runtime/artifacts/release/badregex-run";
  writeFile(ws, `${artRoot}/plan/acceptance.json`, {
    criteria: [
      { id: "AC-1", description: "bad regex", verify_command: "echo ok", expected_pattern: "[invalid(regex", verify_tier: "deterministic" },
    ],
    artifacts: ["x"],
    owner: "pm",
    version: "1.0",
  });

  const result = runAcceptanceTest(ws, artRoot);
  assert.equal(result.verdict, "fail");
  assert.equal(result.deterministic_fail, 1);
  assert.equal(result.criteria_results[0].status, "error");
  assert.match(result.criteria_results[0].error, /invalid expected_pattern regex/);
});

test("acceptance test: blocked command produces error status", () => {
  const ws = makeWorkspace();
  const artRoot = "runtime/artifacts/release/blocked-run";
  writeFile(ws, `${artRoot}/plan/acceptance.json`, {
    criteria: [
      { id: "AC-1", description: "dangerous cmd", verify_command: "rm -rf /", verify_tier: "deterministic" },
    ],
    artifacts: ["x"],
    owner: "pm",
    version: "1.0",
  });

  const result = runAcceptanceTest(ws, artRoot);
  assert.equal(result.verdict, "fail");
  assert.equal(result.criteria_results[0].status, "error");
  assert.match(result.criteria_results[0].error, /blocked|not whitelisted/);
});
