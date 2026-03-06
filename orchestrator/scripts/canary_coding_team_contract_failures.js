import fs from "fs";
import path from "path";
import { validatePmOutput, validateArchitectOutput } from "../src/coding_team_validators.js";
import { validateCodingTeamHandoff } from "../src/coding_team_handoff_validators.js";
import {
  getDefaultHandoffContractPath,
  loadHandoffContractsOrThrow,
} from "../src/handoff_contract_registry.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected='${expected}' actual='${actual}'`);
  }
}

function writeJson(absPath, value) {
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, JSON.stringify(value, null, 2), "utf8");
}

function writeText(absPath, value) {
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, value, "utf8");
}

function validateStepLikeWorkflow({ workspaceRoot, artifactRoot, handoff, role }) {
  const roleResult = role === "pm"
    ? validatePmOutput({ workspaceRoot, artifactRoot })
    : validateArchitectOutput({ workspaceRoot, artifactRoot });
  if (!roleResult.ok) {
    return { stop_stage: "role_output_validation", code: roleResult.code, role_result: roleResult };
  }
  const handoffResult = validateCodingTeamHandoff({ workspaceRoot, artifactRoot, handoff });
  if (!handoffResult.ok) {
    return { stop_stage: "handoff_validation", code: handoffResult.code, handoff_result: handoffResult };
  }
  return { stop_stage: "passed", code: "OK", role_result: roleResult, handoff_result: handoffResult };
}

function main() {
  const fixturePath = path.resolve(process.cwd(), "canary_inputs", "coding_team_contract_failures_min.json");
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  const registry = loadHandoffContractsOrThrow(getDefaultHandoffContractPath());

  const tmpRoot = path.resolve(process.cwd(), "artifacts", "canary", "contract_failures_fixture");

  writeText(
    path.join(tmpRoot, "pm_invalid_shape", "plan", "spec.md"),
    "# Scope\n\n## User Stories\n\n## Acceptance Criteria\n\n## Non-Goals\n\n## Artifact List\n"
  );
  writeText(
    path.join(tmpRoot, "pm_invalid_shape", "plan", "milestones.md"),
    "scope user stories acceptance criteria non-goals artifact list"
  );
  writeJson(
    path.join(tmpRoot, "pm_invalid_shape", "plan", "acceptance.json"),
    { criteria: ["Login works"] }
  );

  writeText(
    path.join(tmpRoot, "arch_invalid_shape", "plan", "arch.md"),
    "# Module Breakdown\n\n## Interfaces\n\n## Dependency Choices\n\n## Risk Notes\n"
  );
  writeText(
    path.join(tmpRoot, "arch_invalid_shape", "plan", "workplan.md"),
    "module breakdown interfaces dependency choices risk notes"
  );
  writeJson(
    path.join(tmpRoot, "arch_invalid_shape", "risk", "risk_report.json"),
    { risks: [{ title: "auth" }] }
  );

  writeText(
    path.join(tmpRoot, "pm_handoff_missing_field", "plan", "spec.md"),
    "# Scope\n\n## User Stories\n\n## Acceptance Criteria\n\n## Non-Goals\n\n## Artifact List\n"
  );
  writeText(
    path.join(tmpRoot, "pm_handoff_missing_field", "plan", "milestones.md"),
    "scope user_stories acceptance_criteria non_goals artifact_list"
  );
  writeJson(
    path.join(tmpRoot, "pm_handoff_missing_field", "plan", "acceptance.json"),
    { criteria: ["Login works"], artifacts: ["plan/spec.md"], owner: "pm_agent", version: "v1" }
  );
  writeJson(
    path.join(tmpRoot, "pm_handoff_missing_field", "handoff", "pm_to_architect.json"),
    {
      from_step: "pm_spec",
      to_steps: ["arch_design"],
      artifacts: ["plan/spec.md"],
      acceptance: { criteria: ["Login works"] }
    }
  );

  writeText(
    path.join(tmpRoot, "arch_handoff_schema_invalid", "plan", "arch.md"),
    "# Module Breakdown\n\n## Interfaces\n\n## Dependency Choices\n\n## Risk Notes\n"
  );
  writeText(
    path.join(tmpRoot, "arch_handoff_schema_invalid", "plan", "workplan.md"),
    "module_breakdown interfaces dependency_choices risk_notes"
  );
  writeJson(
    path.join(tmpRoot, "arch_handoff_schema_invalid", "risk", "risk_report.json"),
    { risks: [{ title: "auth", mitigation: "staged rollout" }], decision_log: ["Use postgres"] }
  );
  writeJson(
    path.join(tmpRoot, "arch_handoff_schema_invalid", "handoff", "architect_to_impl.json"),
    {
      from_step: "arch_design",
      to_steps: ["impl_be"],
      modules: [""],
      interfaces: ["POST /login"],
      decisions: ["postgres"],
      risks: ["auth migration"]
    }
  );

  const pmInvalidShape = validateStepLikeWorkflow({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/contract_failures_fixture/pm_invalid_shape",
    handoff: registry.handoffs.pm_to_architect,
    role: "pm",
  });
  const archInvalidShape = validateStepLikeWorkflow({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/contract_failures_fixture/arch_invalid_shape",
    handoff: registry.handoffs.architect_to_backend_frontend,
    role: "arch",
  });
  const pmHandoffMissingField = validateStepLikeWorkflow({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/contract_failures_fixture/pm_handoff_missing_field",
    handoff: registry.handoffs.pm_to_architect,
    role: "pm",
  });
  const archHandoffSchemaInvalid = validateStepLikeWorkflow({
    workspaceRoot: process.cwd(),
    artifactRoot: "artifacts/canary/contract_failures_fixture/arch_handoff_schema_invalid",
    handoff: registry.handoffs.architect_to_backend_frontend,
    role: "arch",
  });

  assertEqual(pmInvalidShape.code, fixture.expected.pm_invalid_shape_code, "pmInvalidShape.code");
  assertEqual(archInvalidShape.code, fixture.expected.arch_invalid_shape_code, "archInvalidShape.code");
  assertEqual(pmHandoffMissingField.code, fixture.expected.handoff_missing_field_code, "pmHandoffMissingField.code");
  assertEqual(archHandoffSchemaInvalid.code, fixture.expected.handoff_schema_invalid_code, "archHandoffSchemaInvalid.code");
  assertEqual(pmInvalidShape.stop_stage, "role_output_validation", "pmInvalidShape.stop_stage");
  assertEqual(archInvalidShape.stop_stage, "role_output_validation", "archInvalidShape.stop_stage");
  assertEqual(pmHandoffMissingField.stop_stage, "handoff_validation", "pmHandoffMissingField.stop_stage");
  assertEqual(archHandoffSchemaInvalid.stop_stage, "handoff_validation", "archHandoffSchemaInvalid.stop_stage");

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "coding_team_contract_failures");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "coding_team_contract_failures_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    pm_invalid_shape: pmInvalidShape,
    arch_invalid_shape: archInvalidShape,
    pm_handoff_missing_field: pmHandoffMissingField,
    arch_handoff_schema_invalid: archHandoffSchemaInvalid,
  }, null, 2), "utf8");
  console.log("# Coding Team Contract Failures Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
