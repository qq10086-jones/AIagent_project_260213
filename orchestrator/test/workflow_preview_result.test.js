import test from "node:test";
import assert from "node:assert/strict";

import {
  buildReleaseNotesUrl,
  resolvePreviewResultFromSteps,
  resolveWorkflowResultUrl,
} from "../src/domain/workflow_preview_result.js";

test("resolvePreviewResultFromSteps prefers deploy_preview result url", () => {
  const result = resolvePreviewResultFromSteps([
    { step_id: "release_pack", result_json: "{}" },
    {
      step_id: "deploy_preview",
      result_json: JSON.stringify({
        preview_url: "https://preview.example.com/run-1",
        preview_status: "deployed",
        deployment_result_path: "preview/deployment_result.json",
      }),
    },
  ]);

  assert.equal(result.preview_url, "https://preview.example.com/run-1");
  assert.equal(result.preview_status, "deployed");
  assert.equal(result.deployment_result_path, "preview/deployment_result.json");
});

test("resolveWorkflowResultUrl falls back to release notes when preview is absent", () => {
  const run = { run_id: "run-1", workflow_run_id: "wf-1" };
  const fallback = buildReleaseNotesUrl({
    minio: { endpoint: "http://nexus-minio:9000" },
    minioBucket: "nexus-artifacts",
    run,
  });

  const result = resolveWorkflowResultUrl({
    steps: [{ step_id: "deploy_preview", result_json: JSON.stringify({ preview_status: "skipped" }) }],
    minio: { endpoint: "http://nexus-minio:9000" },
    minioBucket: "nexus-artifacts",
    run,
  });

  assert.equal(result.preview_url, null);
  assert.equal(result.result_url, fallback);
});
