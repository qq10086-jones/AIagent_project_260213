import fs from "fs";
import path from "path";
import { validatePmOutput, validateArchitectOutput } from "../src/coding_team_validators.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const fixturePath = path.resolve(process.cwd(), "canary_inputs", "coding_team_output_validators_min.json");
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));

  const tmpRoot = path.resolve(process.cwd(), "artifacts", "canary", "output_validators_fixture");
  fs.mkdirSync(tmpRoot, { recursive: true });
  fs.mkdirSync(path.join(tmpRoot, "pm_ok", "plan"), { recursive: true });
  fs.mkdirSync(path.join(tmpRoot, "arch_ok", "plan"), { recursive: true });
  fs.mkdirSync(path.join(tmpRoot, "arch_ok", "risk"), { recursive: true });

  fs.writeFileSync(
    path.join(tmpRoot, "pm_ok", "plan", "spec.md"),
    [
      "# Scope",
      "Build CRM MVP",
      "",
      "## User Stories",
      "- As a user, I can log in.",
      "",
      "## Acceptance Criteria",
      "- Login works.",
      "",
      "## Non-Goals",
      "- No billing.",
      "",
      "## Artifact List",
      "- spec.md",
    ].join("\n"),
    "utf8"
  );
  fs.writeFileSync(
    path.join(tmpRoot, "pm_ok", "plan", "milestones.md"),
    [
      "# Milestones",
      "scope confirmed",
      "user stories reviewed",
      "artifact list approved",
    ].join("\n"),
    "utf8"
  );
  fs.writeFileSync(
    path.join(tmpRoot, "pm_ok", "plan", "acceptance.json"),
    JSON.stringify({
      criteria: ["Login works"],
      artifacts: ["plan/spec.md", "plan/milestones.md"],
      owner: "pm_agent",
      version: "v1",
      generated_at: new Date().toISOString(),
      step_id: "pm_spec",
    }, null, 2),
    "utf8"
  );

  fs.writeFileSync(
    path.join(tmpRoot, "arch_ok", "plan", "arch.md"),
    [
      "# Module Breakdown",
      "module: auth",
      "",
      "## Interfaces",
      "interface: login API",
      "",
      "## Dependency Choices",
      "dependency: postgres",
      "",
      "## Risk Notes",
      "risk: auth integration",
    ].join("\n"),
    "utf8"
  );
  fs.writeFileSync(
    path.join(tmpRoot, "arch_ok", "plan", "workplan.md"),
    [
      "# Workplan",
      "module breakdown finalized",
      "interface contract ready",
      "dependency choices confirmed",
      "risk notes tracked",
    ].join("\n"),
    "utf8"
  );
  fs.writeFileSync(
    path.join(tmpRoot, "arch_ok", "risk", "risk_report.json"),
    JSON.stringify({
      risks: [{ level: "medium", title: "auth integration", mitigation: "add staged rollout" }],
      decision_log: ["Use postgres for primary persistence", "Ship auth first behind feature flag"],
      generated_at: new Date().toISOString(),
    }, null, 2),
    "utf8"
  );

  const pm = validatePmOutput({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/output_validators_fixture/pm_missing",
  });
  const arch = validateArchitectOutput({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/output_validators_fixture/arch_missing",
  });

  assertEqual(pm.ok, false, "pm.ok");
  assertEqual(pm.code, fixture.expected?.pm_error_code, "pm.code");
  assertEqual(arch.ok, false, "architect.ok");
  assertEqual(arch.code, fixture.expected?.architect_error_code, "architect.code");

  const pmOk = validatePmOutput({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/output_validators_fixture/pm_ok",
  });
  const archOk = validateArchitectOutput({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/output_validators_fixture/arch_ok",
  });
  assertEqual(pmOk.ok, true, "pmOk.ok");
  assertEqual(archOk.ok, true, "archOk.ok");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "coding_team_output_validators");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "coding_team_output_validators_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    pm,
    architect: arch,
    pm_ok: pmOk,
    architect_ok: archOk,
  }, null, 2), "utf8");
  console.log("# Coding Team Output Validators Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
