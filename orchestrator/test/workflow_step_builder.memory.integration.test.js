import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { createStepBuilder } from "../src/domain/workflow_step_builder.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-memory-step-"));
}

function makeStepBuilder() {
  return createStepBuilder({
    registry: {
      project_types: {
        webapp_crm: {
          acceptance_suite: "webapp_crm_v0",
        },
      },
      acceptance_suites: {
        webapp_crm_v0: {
          commands: ["node --version"],
          required_reports: ["qa_report.json", "run_manifest.json"],
        },
      },
    },
    promptScriptRegistry: {
      scripts: {
        "architect.system_spec.v2": {
          script_id: "architect.system_spec.v2",
          role: "architect",
          llm_role: "architect",
          artifact_type: "architect_spec",
          validation: {},
        },
      },
    },
    handoffContracts: { handoffs: {} },
  });
}

function makeRun(runId = "memory-run") {
  return {
    run_id: runId,
    workflow_run_id: `wf-${runId}`,
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    input_json: JSON.stringify({
      goal: "Design the CRM architecture",
    }),
  };
}

function makeStepDef() {
  return {
    id: "arch_design",
    role: "architect",
    tool: "coding.delegate",
    gate: "policy",
    prompt_script_id: "architect.system_spec.v2",
  };
}

test("arch_design prompt includes read-only memory context when memory files exist", () => {
  const memoryRoot = path.join(makeWorkspace(), "memory");
  const projectRoot = path.join(memoryRoot, "memory-run");
  fs.mkdirSync(path.join(projectRoot, "adrs"), { recursive: true });
  fs.writeFileSync(
    path.join(projectRoot, "project_context.json"),
    JSON.stringify({ product: "CRM", domain: "sales_ops" }, null, 2)
  );
  fs.writeFileSync(
    path.join(projectRoot, "adrs", "ADR-001.json"),
    JSON.stringify({ adr_id: "ADR-001", title: "Use Postgres", status: "accepted" }, null, 2)
  );
  fs.writeFileSync(
    path.join(projectRoot, "task_history.json"),
    JSON.stringify([{ step_id: "impl_be", status: "succeeded", summary: "backend contract completed" }], null, 2)
  );

  const previousMemoryRoot = process.env.MEMORY_ROOT;
  process.env.MEMORY_ROOT = memoryRoot;
  try {
    const { buildStepPayload } = makeStepBuilder();
    const payload = buildStepPayload({
      run: makeRun(),
      stepDef: makeStepDef(),
      stepIndex: 1,
    });

    assert.match(payload.task_prompt, /\[Project Memory Context - Read Only\]/);
    assert.match(payload.task_prompt, /Project Context:/);
    assert.match(payload.task_prompt, /- product: CRM/);
    assert.match(payload.task_prompt, /Prior ADR Summaries \(1\):/);
    assert.match(payload.task_prompt, /ADR-001 \| accepted \| Use Postgres/);
    assert.match(payload.task_prompt, /Recent Task History \(1\):/);
    assert.match(payload.task_prompt, /impl_be \| succeeded \| backend contract completed/);
  } finally {
    if (previousMemoryRoot === undefined) delete process.env.MEMORY_ROOT;
    else process.env.MEMORY_ROOT = previousMemoryRoot;
  }
});

test("arch_design prompt omits memory block when memory files do not exist", () => {
  const memoryRoot = path.join(makeWorkspace(), "memory");
  const previousMemoryRoot = process.env.MEMORY_ROOT;
  process.env.MEMORY_ROOT = memoryRoot;
  try {
    const { buildStepPayload } = makeStepBuilder();
    const payload = buildStepPayload({
      run: makeRun("no-memory-run"),
      stepDef: makeStepDef(),
      stepIndex: 1,
    });

    assert.doesNotMatch(payload.task_prompt, /\[Project Memory Context - Read Only\]/);
    assert.match(payload.task_prompt, /\[CodingTeam Step\] Architecture Design/);
  } finally {
    if (previousMemoryRoot === undefined) delete process.env.MEMORY_ROOT;
    else process.env.MEMORY_ROOT = previousMemoryRoot;
  }
});
