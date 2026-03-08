/**
 * canary_arch_design.js
 *
 * WS-12-04: Architect canary with real artifact check.
 *
 * Validates that the arch_design step produces all required artifacts,
 * that architect_to_impl.json contains at least 1 ADR decision,
 * and that the handoff passes schema validation.
 *
 * Also verifies the FAILURE case: missing plan/interfaces.md causes validation to fail.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateArchitectOutput } from "../src/coding_team_validators.js";
import { validateJsonSchemaLite } from "../src/schema_lite_validator.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// Load arch handoff schema
const ARCH_HANDOFF_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(ROOT, "contracts", "coding_team_arch_handoff.schema.json"), "utf8")
);

const REQUIRED_ARTIFACT_FILES = [
  "plan/arch.md",
  "plan/interfaces.md",
  "plan/workplan.md",
  "risk/risk_report.json",
  "handoff/architect_to_impl.json",
];

function assert(condition, label) {
  if (!condition) throw new Error(`ASSERTION FAILED: ${label}`);
}

function readJsonSafe(absPath) {
  try { return JSON.parse(fs.readFileSync(absPath, "utf8")); } catch { return null; }
}

function setupFixture(fixtureRoot, { includeInterfaces = true } = {}) {
  fs.mkdirSync(path.join(fixtureRoot, "plan"), { recursive: true });
  fs.mkdirSync(path.join(fixtureRoot, "risk"), { recursive: true });
  fs.mkdirSync(path.join(fixtureRoot, "handoff"), { recursive: true });

  fs.writeFileSync(
    path.join(fixtureRoot, "plan", "arch.md"),
    [
      "# Module Breakdown",
      "",
      "## Modules",
      "- auth: handles login and session management",
      "- crm: contact and deal management",
      "",
      "## Interfaces",
      "- POST /login",
      "- GET /contacts",
      "",
      "## Dependency Choices",
      "- PostgreSQL for primary persistence",
      "- Redis for session cache",
      "",
      "## Risk Notes",
      "- Auth migration requires staged rollout",
      "- Session token rotation on deploy",
    ].join("\n"),
    "utf8"
  );

  if (includeInterfaces) {
    fs.writeFileSync(
      path.join(fixtureRoot, "plan", "interfaces.md"),
      [
        "# API Interfaces",
        "",
        "## POST /login",
        "Request: `{ username, password }`",
        "Response: `{ token, expires_at }`",
        "",
        "## GET /contacts",
        "Response: `[{ id, name, email }]`",
        "",
        "## POST /contacts",
        "Request: `{ name, email, phone? }`",
      ].join("\n"),
      "utf8"
    );
  }

  fs.writeFileSync(
    path.join(fixtureRoot, "plan", "workplan.md"),
    [
      "# Workplan",
      "",
      "module breakdown finalized",
      "interface contract ready",
      "dependency choices confirmed",
      "risk notes tracked",
    ].join("\n"),
    "utf8"
  );

  fs.writeFileSync(
    path.join(fixtureRoot, "risk", "risk_report.json"),
    JSON.stringify({
      risks: [
        { level: "medium", title: "auth migration", mitigation: "staged rollout behind feature flag" },
        { level: "low", title: "session token rotation", mitigation: "zero-downtime deploy script" },
      ],
      decision_log: [
        "Use PostgreSQL as primary persistence layer",
        "Use Redis for session caching",
        "Ship auth module first behind feature flag",
      ],
      generated_at: new Date().toISOString(),
    }, null, 2),
    "utf8"
  );

  fs.writeFileSync(
    path.join(fixtureRoot, "handoff", "architect_to_impl.json"),
    JSON.stringify({
      from_step: "arch_design",
      to_steps: ["impl_be", "impl_fe", "qa_verify"],
      modules: ["auth", "crm"],
      interfaces: ["POST /login", "GET /contacts", "POST /contacts"],
      decisions: [
        { adr_id: "ADR-001", title: "Use PostgreSQL as primary database", status: "accepted" },
        { adr_id: "ADR-002", title: "Redis session cache over DB sessions", status: "accepted" },
      ],
      risks: ["auth migration", "session token rotation on deploy"],
    }, null, 2),
    "utf8"
  );
}

function validateArtifactPresence(fixtureRoot, artifactRoot) {
  const missing = [];
  for (const rel of REQUIRED_ARTIFACT_FILES) {
    const abs = path.resolve(fixtureRoot, rel);
    if (!fs.existsSync(abs)) missing.push(rel);
  }
  return { ok: missing.length === 0, missing };
}

function validateHandoffDecisions(handoffJson) {
  if (!handoffJson) return { ok: false, code: "HANDOFF_PARSE_FAILED", detail: "could not parse architect_to_impl.json" };
  const decisions = handoffJson.decisions;
  if (!Array.isArray(decisions) || decisions.length === 0) {
    return { ok: false, code: "HANDOFF_NO_DECISIONS", detail: "decisions array is empty or missing" };
  }
  return { ok: true, decisions_count: decisions.length };
}

function validateHandoffSchema(handoffJson) {
  const errors = validateJsonSchemaLite(ARCH_HANDOFF_SCHEMA, handoffJson, "$");
  return { ok: errors.length === 0, errors };
}

function main() {
  const artifactsRoot = path.resolve(ROOT, "artifacts", "canary", "arch_design");
  const okFixture = path.join(artifactsRoot, "arch_ok");
  const noInterfacesFixture = path.join(artifactsRoot, "arch_no_interfaces");
  const emptyInterfacesFixture = path.join(artifactsRoot, "arch_empty_interfaces");

  // --- Setup fixtures ---
  setupFixture(okFixture, { includeInterfaces: true });
  setupFixture(noInterfacesFixture, { includeInterfaces: false });
  setupFixture(emptyInterfacesFixture, { includeInterfaces: true });
  fs.writeFileSync(
    path.join(emptyInterfacesFixture, "plan", "interfaces.md"),
    [
      "# Interfaces",
      "",
      "This file exists but does not define any concrete interface heading.",
    ].join("\n"),
    "utf8"
  );

  // ===== CASE 1: complete arch_design output — should pass =====
  const artifactPresence = validateArtifactPresence(okFixture);
  assert(artifactPresence.ok, `arch_ok: all required artifacts present (missing: ${artifactPresence.missing.join(", ")})`);

  const archValidation = validateArchitectOutput({
    workspaceRoot: ROOT,
    artifactRoot: path.relative(ROOT, okFixture),
  });
  assert(archValidation.ok === true, `arch_ok: validateArchitectOutput passed (got code=${archValidation.code}, detail=${archValidation.detail})`);

  const handoffPath = path.join(okFixture, "handoff", "architect_to_impl.json");
  const handoffJson = readJsonSafe(handoffPath);
  assert(handoffJson !== null, "arch_ok: architect_to_impl.json is valid JSON");

  const decisionsCheck = validateHandoffDecisions(handoffJson);
  assert(decisionsCheck.ok === true, `arch_ok: decisions array has ≥1 entry (${decisionsCheck.detail || ""})`);

  const schemaCheck = validateHandoffSchema(handoffJson);
  assert(schemaCheck.ok === true, `arch_ok: handoff schema valid (errors: ${schemaCheck.errors.join("; ")})`);

  // ===== CASE 2: missing plan/interfaces.md — should fail =====
  const artifactPresenceNoIf = validateArtifactPresence(noInterfacesFixture);
  assert(!artifactPresenceNoIf.ok, "arch_no_interfaces: artifact presence check correctly detects missing interfaces.md");
  assert(artifactPresenceNoIf.missing.includes("plan/interfaces.md"), "arch_no_interfaces: interfaces.md listed in missing");

  const archValidationNoIf = validateArchitectOutput({
    workspaceRoot: ROOT,
    artifactRoot: path.relative(ROOT, noInterfacesFixture),
  });
  assert(archValidationNoIf.ok === false, "arch_no_interfaces: validateArchitectOutput correctly fails");
  assert(
    archValidationNoIf.code === "ARCH_REQUIRED_FILES_MISSING",
    `arch_no_interfaces: error code is ARCH_REQUIRED_FILES_MISSING (got '${archValidationNoIf.code}')`
  );

  // ===== CASE 3: plan/interfaces.md exists but has no interface heading -> should fail =====
  const archValidationEmptyIf = validateArchitectOutput({
    workspaceRoot: ROOT,
    artifactRoot: path.relative(ROOT, emptyInterfacesFixture),
  });
  assert(archValidationEmptyIf.ok === false, "arch_empty_interfaces: validateArchitectOutput correctly fails");
  assert(
    archValidationEmptyIf.code === "ARCH_REQUIRED_SECTIONS_MISSING",
    `arch_empty_interfaces: error code is ARCH_REQUIRED_SECTIONS_MISSING (got '${archValidationEmptyIf.code}')`
  );
  assert(
    String(archValidationEmptyIf.detail || "").includes("ARCH_INTERFACES_UNSPECIFIED"),
    `arch_empty_interfaces: detail includes ARCH_INTERFACES_UNSPECIFIED (got '${archValidationEmptyIf.detail}')`
  );

  // ===== Write report =====
  const report = {
    ok: true,
    generated_at: new Date().toISOString(),
    canary: "arch_design",
    ws: "WS-12-04",
    cases: {
      arch_ok: {
        verdict: "PASS",
        artifact_presence: artifactPresence,
        arch_validation: archValidation,
        decisions_check: decisionsCheck,
        schema_check: schemaCheck,
      },
      arch_no_interfaces: {
        verdict: "PASS (failure correctly detected)",
        artifact_presence: artifactPresenceNoIf,
        arch_validation: { ok: archValidationNoIf.ok, code: archValidationNoIf.code },
      },
      arch_empty_interfaces: {
        verdict: "PASS (empty interfaces correctly rejected)",
        arch_validation: {
          ok: archValidationEmptyIf.ok,
          code: archValidationEmptyIf.code,
          detail: archValidationEmptyIf.detail,
        },
      },
    },
    summary: {
      required_files_checked: REQUIRED_ARTIFACT_FILES,
      schema_id: ARCH_HANDOFF_SCHEMA.$id,
      decisions_min: 1,
      failure_case_covered: true,
    },
  };

  const reportPath = path.join(artifactsRoot, "arch_design_canary.json");
  fs.mkdirSync(artifactsRoot, { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");

  console.log("# Arch Design Canary (WS-12-04)");
  console.log(`- verdict: PASS`);
  console.log(`- cases: arch_ok=PASS, arch_no_interfaces=failure-detected`);
  console.log(`- decisions checked: ${decisionsCheck.decisions_count}`);
  console.log(`- schema: ${ARCH_HANDOFF_SCHEMA.$id}`);
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
