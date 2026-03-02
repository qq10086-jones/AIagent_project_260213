import fs from "fs";

function isObj(v) {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

function safeReadJson(p) {
  try {
    const raw = fs.readFileSync(p, "utf-8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function validateArtifactPack({
  run,
  steps = [],
  checkpoints = [],
  manifestPath,
  summaryPath,
  registry,
}) {
  const reasons = [];
  if (!run || !isObj(run)) reasons.push("run object missing");
  if (!Array.isArray(steps) || steps.length === 0) reasons.push("steps missing");
  if (!manifestPath || !fs.existsSync(manifestPath)) reasons.push("run_manifest.json missing");
  if (!summaryPath || !fs.existsSync(summaryPath)) reasons.push("run_summary.md missing");

  const manifest = manifestPath ? safeReadJson(manifestPath) : null;
  if (!manifest || !isObj(manifest)) {
    reasons.push("run_manifest.json invalid json");
  } else {
    const requiredTop = [
      "workflow_run_id",
      "run_id",
      "workflow_id",
      "project_type",
      "status",
      "steps",
      "checkpoints",
      "step_artifacts",
    ];
    for (const k of requiredTop) {
      if (manifest[k] === undefined || manifest[k] === null) reasons.push(`manifest missing field: ${k}`);
    }
    if (String(manifest.workflow_run_id || "") !== String(run.workflow_run_id || "")) {
      reasons.push("manifest workflow_run_id mismatch");
    }
    if (String(manifest.run_id || "") !== String(run.run_id || "")) {
      reasons.push("manifest run_id mismatch");
    }
    if (!Array.isArray(manifest.steps) || manifest.steps.length === 0) reasons.push("manifest steps empty");
    if (!Array.isArray(manifest.checkpoints)) reasons.push("manifest checkpoints not array");
    if (!Array.isArray(manifest.step_artifacts)) reasons.push("manifest step_artifacts not array");
    if (Array.isArray(manifest.step_artifacts) && Array.isArray(manifest.steps) && manifest.step_artifacts.length !== manifest.steps.length) {
      reasons.push("manifest step_artifacts length mismatch");
    }
    if (String(manifest.status || "") !== "succeeded") reasons.push("manifest status must be succeeded");
  }

  const succeededSteps = steps.filter((s) => String(s.status || "") === "succeeded");
  if (succeededSteps.length !== steps.length) reasons.push("not all workflow steps are succeeded");
  if (Array.isArray(checkpoints) && checkpoints.length < succeededSteps.length) {
    reasons.push("checkpoint count lower than succeeded steps");
  }

  const requiredArtifacts = registry?.project_types?.[run?.project_type]?.required_artifacts || [];
  if (manifest && isObj(manifest.artifact_coverage)) {
    for (const item of requiredArtifacts) {
      if (!manifest.artifact_coverage[item]) reasons.push(`required artifact missing: ${item}`);
    }
  }

  return {
    ok: reasons.length === 0,
    reasons,
    manifest,
    summary_path: summaryPath || null,
    manifest_path: manifestPath || null,
  };
}
