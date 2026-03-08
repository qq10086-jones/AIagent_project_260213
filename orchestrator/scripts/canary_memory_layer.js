import fs from "fs";
import os from "os";
import path from "path";

import { getPriorADRs, getProjectContext, getTaskHistory } from "../src/domain/memory_reader.js";
import { persistWorkflowMemory } from "../src/domain/memory_writer.js";
import { createStepBuilder } from "../src/domain/workflow_step_builder.js";
import { resolveOrchestratorArtifactPath } from "./_paths.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function writeJson(targetPath, value) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, JSON.stringify(value, null, 2), "utf8");
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

function main() {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "ocn-memory-canary-"));
  const memoryRoot = path.join(tempRoot, "memory");
  const releaseRoot = path.join(tempRoot, "artifacts", "release", "run-memory");
  fs.mkdirSync(path.join(releaseRoot, "plan", "adr"), { recursive: true });
  fs.writeFileSync(
    path.join(releaseRoot, "plan", "adr", "ADR-001.md"),
    "# Use Postgres\n\nStatus: accepted\nDecision: persist CRM state in Postgres.\n",
    "utf8"
  );
  writeJson(
    path.join(memoryRoot, "run-memory", "project_context.json"),
    { product: "CRM", domain: "sales_ops", region: "jp" }
  );

  const previousMemoryRoot = process.env.MEMORY_ROOT;
  process.env.MEMORY_ROOT = memoryRoot;

  try {
    const noContextProject = "missing-memory";
    assert(getProjectContext(noContextProject) === null, "memory reader should return null when project_context.json is missing");
    assert(getPriorADRs(noContextProject).length === 0, "memory reader should return [] when ADRs are missing");
    assert(getTaskHistory(noContextProject).length === 0, "memory reader should return [] when task history is missing");

    const persisted = persistWorkflowMemory({
      run: {
        workflow_run_id: "wf-memory",
        run_id: "run-memory",
        workflow_id: "coding_team_v0",
        project_type: "webapp_crm",
      },
      releaseRoot,
    });
    assert(persisted.ok === true, "persistWorkflowMemory should succeed");
    assert(persisted.copied_adr_paths.length === 1, "one ADR markdown file should be copied");

    const context = getProjectContext("run-memory");
    const adrs = getPriorADRs("run-memory");
    const history = getTaskHistory("run-memory");
    assert(context?.product === "CRM", "project context should be readable");
    assert(adrs.length === 1, "copied ADR should be readable");
    assert(adrs[0]?.adr_id === "ADR-001", "ADR id should be derived from markdown filename");
    assert(history.length === 1, "task history entry should be appended");
    assert(history[0]?.status === "succeeded", "task history status should be succeeded");

    const { buildStepPayload } = makeStepBuilder();
    const payload = buildStepPayload({
      run: {
        run_id: "run-memory",
        workflow_run_id: "wf-memory",
        workflow_id: "coding_team_v0",
        project_type: "webapp_crm",
        input_json: JSON.stringify({ goal: "Design the CRM architecture" }),
      },
      stepDef: {
        id: "arch_design",
        role: "architect",
        tool: "coding.delegate",
        gate: "policy",
        prompt_script_id: "architect.system_spec.v2",
      },
      stepIndex: 1,
    });

    assert(/\[Project Memory Context - Read Only\]/.test(payload.task_prompt), "architect prompt should include memory block");
    assert(/ADR-001 \| accepted \| Use Postgres/.test(payload.task_prompt), "architect prompt should include ADR summary");
    assert(
      /workflow_complete \| succeeded \| Workflow coding_team_v0 completed successfully/.test(payload.task_prompt),
      "architect prompt should include recent task history"
    );

    const outDir = resolveOrchestratorArtifactPath("canary", "memory_layer");
    fs.mkdirSync(outDir, { recursive: true });
    const reportPath = path.join(outDir, "memory_layer_canary.json");
    fs.writeFileSync(
      reportPath,
      JSON.stringify(
        {
          ok: true,
          generated_at: new Date().toISOString(),
          checks: {
            reader_missing_graceful: true,
            writer_created_history: true,
            writer_copied_adr_markdown: true,
            architect_prompt_includes_memory_block: true,
          },
          memory_root: memoryRoot.replace(/\\/g, "/"),
          copied_adr_paths: persisted.copied_adr_paths,
        },
        null,
        2
      ),
      "utf8"
    );

    console.log("# Memory Layer Canary");
    console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
  } finally {
    if (previousMemoryRoot === undefined) delete process.env.MEMORY_ROOT;
    else process.env.MEMORY_ROOT = previousMemoryRoot;
  }
}

main();
