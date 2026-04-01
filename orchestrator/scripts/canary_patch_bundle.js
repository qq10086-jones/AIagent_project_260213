import fs from "fs";
import os from "os";
import path from "path";

import { createPatchBundleService } from "../src/domain/patch_bundle_service.js";
import { createStepBuilder } from "../src/domain/workflow_step_builder.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-canary-patch-"));
}

function writeFile(workspaceRoot, relativePath, content) {
  const targetPath = path.join(workspaceRoot, relativePath);
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, content, "utf8");
  return targetPath;
}

function readFile(workspaceRoot, relativePath) {
  return fs.readFileSync(path.join(workspaceRoot, relativePath), "utf8");
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function buildMinimalStepBuilder({ workspaceRoot, diffFirstEnabled }) {
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
          required_reports: ["qa_report.json"],
        },
      },
    },
    promptScriptRegistry: {
      scripts: {
        "backend.impl.v1": {
          script_id: "backend.impl.v1",
          role: "backend",
          llm_role: "backend",
          artifact_type: "backend_impl",
          validation: {},
        },
        "backend.impl.v2": {
          script_id: "backend.impl.v2",
          role: "backend",
          llm_role: "backend",
          artifact_type: "backend_impl_patch_bundle",
          validation: {},
        },
      },
    },
    handoffContracts: { handoffs: {} },
    workspaceRoot,
    runtimeConfig: {
      execution: {
        diff_first_enabled: diffFirstEnabled,
      },
    },
  });
}

function main() {
  const workspaceRoot = makeWorkspace();
  writeFile(
    workspaceRoot,
    "src/server.js",
    [
      "const express = require('express');",
      "app.use(express.json());",
      "const oldBlockStart = true;",
      "const legacy = 'on';",
      "const oldBlockEnd = true;",
      "const removeStart = true;",
      "const removeMe = true;",
      "const removeEnd = true;",
      "",
    ].join("\n")
  );
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/app.js", "export const app = true;\n");

  const service = createPatchBundleService({ workspaceRoot });
  const checks = [];

  const success = service.applyPatchBundle({
    bundle_id: "bundle-ok",
    step_id: "impl_be",
    mode: "structured_patch",
    summary: "happy path",
    operations: [
      {
        type: "insert_after_anchor",
        target_file: "src/server.js",
        anchor: "app.use(express.json());",
        content: "\napp.use(requestLogger);",
      },
      {
        type: "replace_range",
        target_file: "src/server.js",
        anchor_start: "const oldBlockStart = true;",
        anchor_end: "const oldBlockEnd = true;",
        content: "const replacement = 'ok';",
      },
      {
        type: "delete_range",
        target_file: "src/server.js",
        anchor_start: "const removeStart = true;",
        anchor_end: "const removeEnd = true;",
      },
      {
        type: "create_file",
        target_file: "src/request_logger.js",
        file_content: "export function requestLogger(_req, _res, next) { next(); }\n",
      },
    ],
  });
  assert(success.ok === true, "structured patch success case failed");
  assert(/requestLogger/.test(readFile(workspaceRoot, "src/server.js")), "insert_after_anchor not applied");
  assert(/replacement/.test(readFile(workspaceRoot, "src/server.js")), "replace_range not applied");
  assert(!/removeMe/.test(readFile(workspaceRoot, "src/server.js")), "delete_range not applied");
  assert(/requestLogger/.test(readFile(workspaceRoot, "src/request_logger.js")), "create_file not applied");
  checks.push({ id: "patch_applies_successfully", ok: true });

  const sameFile = service.applyPatchBundle({
    bundle_id: "bundle-shift",
    step_id: "impl_fe",
    mode: "structured_patch",
    summary: "same file sequential anchors",
    operations: [
      {
        type: "insert_after_anchor",
        target_file: "workspace/sandbox/crm_site/app.js",
        anchor: "export const app = true;",
        content: "\nexport const markerB = true;",
      },
      {
        type: "insert_after_anchor",
        target_file: "workspace/sandbox/crm_site/app.js",
        anchor: "export const markerB = true;",
        content: "\nexport const markerC = true;",
      },
    ],
  });
  assert(sameFile.ok === true, "same-file multi-op patch failed");
  assert(/markerC/.test(readFile(workspaceRoot, "workspace/sandbox/crm_site/app.js")), "anchor shift case not applied");
  checks.push({ id: "same_file_anchor_shift", ok: true });

  let malformedError = null;
  try {
    service.applyPatchBundle({
      bundle_id: "bundle-bad",
      step_id: "impl_fe",
      mode: "structured_patch",
      summary: "malformed",
      operations: [
        {
          type: "insert_after_anchor",
          target_file: "workspace/sandbox/crm_site/app.js",
          content: "missing anchor",
        },
      ],
    });
  } catch (err) {
    malformedError = err;
  }
  assert(malformedError?.code === "PATCH_ANCHOR_INVALID", "malformed patch did not emit typed error");
  assert(malformedError?.operation_index === 0, "malformed patch missing operation index");
  checks.push({ id: "malformed_patch_typed_error", ok: true, error_code: malformedError.code });

  let shiftedAnchorError = null;
  try {
    service.applyPatchBundle({
      bundle_id: "bundle-anchor-fail",
      step_id: "impl_fe",
      mode: "structured_patch",
      summary: "anchor not found after prior op",
      operations: [
        {
          type: "replace_range",
          target_file: "workspace/sandbox/crm_site/app.js",
          anchor_start: "export const app = true;",
          anchor_end: "export const markerB = true;",
          content: "export const rewritten = true;",
        },
        {
          type: "insert_after_anchor",
          target_file: "workspace/sandbox/crm_site/app.js",
          anchor: "export const markerB = true;",
          content: "\nexport const shouldFail = true;",
        },
      ],
    });
  } catch (err) {
    shiftedAnchorError = err;
  }
  assert(shiftedAnchorError?.code === "PATCH_ANCHOR_NOT_FOUND", "anchor mismatch case did not fail correctly");
  assert(shiftedAnchorError?.operation_index === 1, "anchor mismatch case wrong operation index");
  checks.push({ id: "anchor_not_found_after_prior_op", ok: true, error_code: shiftedAnchorError.code });

  let traversalError = null;
  try {
    service.applyPatchBundle({
      bundle_id: "bundle-traversal",
      step_id: "impl_be",
      mode: "structured_patch",
      summary: "traversal",
      operations: [
        {
          type: "create_file",
          target_file: "../escape.js",
          file_content: "console.log('bad');\n",
        },
      ],
    });
  } catch (err) {
    traversalError = err;
  }
  assert(traversalError?.code === "PATCH_PATH_TRAVERSAL", "path traversal case did not fail correctly");
  checks.push({ id: "path_traversal_rejected", ok: true, error_code: traversalError.code });

  const fallbackBundlePath = writeFile(
    workspaceRoot,
    "artifacts/release/canary-run/impl/be_patch_bundle.json",
    JSON.stringify({
      bundle_id: "bundle-fallback",
      step_id: "impl_be",
      mode: "full_file_fallback",
      operations: [],
      target_files: ["src/server.js"],
      summary: "fallback observable",
    }, null, 2)
  );
  const fallbackResult = service.applyPatchBundleFromFile(fallbackBundlePath);
  assert(fallbackResult.mode === "full_file_fallback", "fallback mode not observable");
  checks.push({ id: "fallback_mode_observable", ok: true, mode: fallbackResult.mode });

  const fullFileBuilder = buildMinimalStepBuilder({ workspaceRoot, diffFirstEnabled: false });
  const payload = fullFileBuilder.buildStepPayload({
    run: {
      run_id: "canary-run",
      workflow_run_id: "wf-canary-run",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Implement backend" }),
    },
    stepDef: {
      id: "impl_be",
      role: "backend",
      tool: "coding.delegate",
      gate: "policy",
      prompt_script_id: "backend.impl.v1",
    },
    stepIndex: 2,
  });
  assert(payload.execution_mode_requested === "full_file_fallback", "feature gate disabled did not force full-file mode");
  checks.push({ id: "feature_gate_disabled_full_file_mode", ok: true, mode: payload.execution_mode_requested });

  const outDir = path.resolve(process.cwd(), "artifacts", "canary", "patch_bundle");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "patch_bundle_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    workspace_root: workspaceRoot.replace(/\\/g, "/"),
    checks,
  }, null, 2), "utf8");

  console.log("# Patch Bundle Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main();
