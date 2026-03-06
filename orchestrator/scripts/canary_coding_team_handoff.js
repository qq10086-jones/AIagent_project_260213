import fs from "fs";
import path from "path";
import {
  getDefaultHandoffContractPath,
  loadHandoffContractsOrThrow,
} from "../src/handoff_contract_registry.js";
import { validateCodingTeamHandoff } from "../src/coding_team_handoff_validators.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function main() {
  const fixturePath = path.resolve(process.cwd(), "canary_inputs", "coding_team_handoff_min.json");
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  const registry = loadHandoffContractsOrThrow(getDefaultHandoffContractPath());
  assertEqual(registry.workflow_id, fixture.expected?.workflow_id, "workflow_id");

  for (const [handoffId, spec] of Object.entries(fixture.expected?.handoffs || {})) {
    const handoff = registry.handoffs?.[handoffId];
    if (!handoff) throw new Error(`missing handoff '${handoffId}'`);
    assertEqual(String(handoff.from_step || ""), String(spec.from_step || ""), `${handoffId}.from_step`);
  }

  const tmpRoot = path.resolve(process.cwd(), "artifacts", "canary", "handoff_fixture");
  fs.mkdirSync(path.join(tmpRoot, "pm_ok", "plan"), { recursive: true });
  fs.mkdirSync(path.join(tmpRoot, "pm_ok", "handoff"), { recursive: true });
  fs.mkdirSync(path.join(tmpRoot, "arch_ok", "plan"), { recursive: true });
  fs.mkdirSync(path.join(tmpRoot, "arch_ok", "risk"), { recursive: true });
  fs.mkdirSync(path.join(tmpRoot, "arch_ok", "handoff"), { recursive: true });

  fs.writeFileSync(path.join(tmpRoot, "pm_ok", "plan", "spec.md"), "# Scope\n\n## User Stories\n\n## Acceptance Criteria\n\n## Non Goals\n\n## Artifact List\n", "utf8");
  fs.writeFileSync(path.join(tmpRoot, "pm_ok", "plan", "acceptance.json"), JSON.stringify({ criteria: ["ok"] }, null, 2), "utf8");
  fs.writeFileSync(path.join(tmpRoot, "pm_ok", "plan", "milestones.md"), "scope user_stories acceptance_criteria non_goals artifact_list", "utf8");
  fs.writeFileSync(path.join(tmpRoot, "pm_ok", "handoff", "pm_to_architect.json"), JSON.stringify({
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "CRM MVP",
    artifacts: ["plan/spec.md", "plan/acceptance.json"],
    acceptance: { criteria: ["Login works"] }
  }, null, 2), "utf8");

  fs.writeFileSync(path.join(tmpRoot, "arch_ok", "plan", "arch.md"), "# Module Breakdown\n\n## Interfaces\n\n## Dependency Choices\n\n## Risk Notes\n", "utf8");
  fs.writeFileSync(path.join(tmpRoot, "arch_ok", "plan", "workplan.md"), "module_breakdown interfaces dependency_choices risk_notes", "utf8");
  fs.writeFileSync(path.join(tmpRoot, "arch_ok", "risk", "risk_report.json"), JSON.stringify({ risks: [{ title: "auth", mitigation: "staged rollout" }] }, null, 2), "utf8");
  fs.writeFileSync(path.join(tmpRoot, "arch_ok", "handoff", "architect_to_impl.json"), JSON.stringify({
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe", "qa_verify"],
    modules: ["auth", "crm"],
    interfaces: ["POST /login"],
    decisions: ["postgres"],
    risks: ["auth migration"]
  }, null, 2), "utf8");

  const pmOk = validateCodingTeamHandoff({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/handoff_fixture/pm_ok",
    handoff: registry.handoffs.pm_to_architect,
  });
  const archOk = validateCodingTeamHandoff({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/handoff_fixture/arch_ok",
    handoff: registry.handoffs.architect_to_backend_frontend,
  });
  const pmMissingTyped = validateCodingTeamHandoff({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/output_validators_fixture/pm_ok",
    handoff: registry.handoffs.pm_to_architect,
  });

  assertEqual(pmOk.ok, true, "pmOk.ok");
  assertEqual(archOk.ok, true, "archOk.ok");
  assertEqual(pmMissingTyped.ok, false, "pmMissingTyped.ok");
  assertEqual(pmMissingTyped.code, "HANDOFF_ARTIFACTS_MISSING", "pmMissingTyped.code");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "coding_team_handoff");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "coding_team_handoff_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    handoff_count: Object.keys(registry.handoffs || {}).length,
    pm_ok: pmOk,
    arch_ok: archOk,
    pm_missing_typed: pmMissingTyped,
  }, null, 2), "utf8");
  console.log("# Coding Team Handoff Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
