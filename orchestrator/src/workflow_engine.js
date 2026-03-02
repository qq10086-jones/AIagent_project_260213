import crypto from "crypto";
import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";
import { analyzeTaskRisk } from "./policy.js";
import { validateArtifactPack } from "./artifact_pack_validator.js";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";

function parseJsonSafe(raw, fallback = {}) {
  if (!raw || typeof raw !== "string") return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function base64UrlEncode(input) {
  return Buffer.from(input).toString("base64url");
}

function base64UrlDecode(input) {
  return Buffer.from(String(input || ""), "base64url").toString("utf8");
}

function signResumePayload(payload, secret) {
  const body = base64UrlEncode(JSON.stringify(payload));
  const sig = crypto.createHmac("sha256", secret).update(body).digest("base64url");
  return `${body}.${sig}`;
}

function verifyResumePayload(token, secret) {
  const tokenText = String(token || "");
  const [body, sig] = tokenText.split(".");
  if (!body || !sig) return { ok: false, error: "RESUME_INVALID: malformed token" };
  const expected = crypto.createHmac("sha256", secret).update(body).digest("base64url");
  if (expected !== sig) return { ok: false, error: "RESUME_INVALID: bad signature" };
  let payload = null;
  try {
    payload = JSON.parse(base64UrlDecode(body));
  } catch {
    return { ok: false, error: "RESUME_INVALID: bad payload" };
  }
  const exp = Number(payload?.exp || 0);
  const now = Math.floor(Date.now() / 1000);
  if (!Number.isFinite(exp) || exp <= now) {
    return { ok: false, error: "RESUME_INVALID: expired token" };
  }
  return { ok: true, payload };
}

function buildWorkspaceHash({ workflow_run_id, step_index, task_id, status, artifacts }) {
  const raw = JSON.stringify({
    workflow_run_id,
    step_index,
    task_id,
    status,
    artifacts: Array.isArray(artifacts) ? artifacts : [],
  });
  return crypto.createHash("sha256").update(raw).digest("hex");
}

function normalizeStepStatus(status) {
  const s = String(status || "").toLowerCase();
  if (["pending", "queued", "waiting_approval", "running", "succeeded", "failed"].includes(s)) return s;
  return "pending";
}

const STEP_CONTRACTS = {
  pm_spec: {
    title: "PM Specification",
    required_artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    instructions: [
      "Define user stories, scope boundaries, and non-goals for a minimal CRM web app.",
      "Write measurable acceptance criteria in plan/acceptance.json.",
      "Create phased milestones in plan/milestones.md.",
    ],
  },
  arch_design: {
    title: "Architecture Design",
    required_artifacts: ["plan/arch.md", "risk/risk_report.json", "plan/workplan.md"],
    instructions: [
      "Provide architecture decisions with tradeoffs and module boundaries.",
      "Publish top risks and mitigations in risk/risk_report.json.",
      "Split implementation work for FE/BE/QA in plan/workplan.md.",
    ],
  },
  impl_fe: {
    title: "Frontend Implementation",
    required_artifacts: ["patch/diff.patch", "tests/frontend_test_report.md", "run/run_frontend.md"],
    instructions: [
      "Implement frontend changes as patch-level updates only.",
      "Record frontend checks and outcomes in tests/frontend_test_report.md.",
      "Document local run steps in run/run_frontend.md.",
    ],
  },
  impl_be: {
    title: "Backend Implementation",
    required_artifacts: ["patch/diff.patch", "tests/backend_test_report.md", "run/run_backend.md"],
    instructions: [
      "Implement backend/API/data layer updates with backward compatibility notes.",
      "Record backend unit/integration results in tests/backend_test_report.md.",
      "Document backend run steps in run/run_backend.md.",
    ],
  },
  qa_verify: {
    title: "QA Verification",
    required_artifacts: ["tests/test_plan.md", "qa/smoke_report.md", "qa/verification.json"],
    instructions: [
      "Verify acceptance criteria against generated implementation artifacts.",
      "Publish pass/fail mapping in qa/verification.json.",
      "Provide smoke evidence in qa/smoke_report.md.",
    ],
  },
  release_pack: {
    title: "Release Pack",
    required_artifacts: ["summary/run_summary.md", "meta/run_manifest.json"],
    instructions: [
      "Assemble final release summary and execution manifest.",
      "Ensure artifact references are complete and traceable.",
    ],
  },
};

function buildStepPrompt({ run, stepDef, input, payload }) {
  const c = STEP_CONTRACTS[stepDef.id] || null;
  const goal = String(input.goal || input.task_prompt || input.prompt || "Build a minimal CRM web app").trim();
  const title = c?.title || stepDef.id;
  const required = Array.isArray(c?.required_artifacts) ? c.required_artifacts : [];
  const lines = Array.isArray(c?.instructions) ? c.instructions : [];
  const artifactRoot = String(payload.artifact_root || "");

  const outputReq = required.length > 0
    ? `Required artifacts (relative to ${artifactRoot}):\n- ${required.join("\n- ")}`
    : "Required artifacts: follow workflow step contract.";
  const guidance = lines.length > 0
    ? `Execution requirements:\n- ${lines.join("\n- ")}`
    : "Execution requirements: complete this step with verifiable outputs.";
  return [
    `[CodingTeam Step] ${title}`,
    `Workflow: ${run.workflow_id}`,
    `Project Type: ${run.project_type}`,
    `Step ID: ${stepDef.id}`,
    `Role: ${stepDef.role}`,
    `Goal: ${goal}`,
    guidance,
    outputReq,
    "Constraints:",
    "- Prefer small, reviewable changes.",
    "- Keep outputs deterministic and explicit.",
    "- Include concise validation evidence.",
  ].join("\n");
}

export function createWorkflowEngine({
  pool,
  registry,
  enqueueTask,
  recordEvent,
  makeIdempotencyKey,
  resumeTokenSecret = "dev-resume-secret",
  resumeTokenTtlSec = 86400,
  workspaceRoot = "/workspace",
  minio = null,
}) {
  const minioCfg = minio || {};
  const minioEnabled = Boolean(minioCfg.enabled);
  const minioBucket = String(minioCfg.bucket || "nexus-artifacts");
  const s3 = minioEnabled
    ? new S3Client({
        endpoint: String(minioCfg.endpoint || "http://nexus-minio:9000"),
        credentials: {
          accessKeyId: String(minioCfg.accessKey || "nexus"),
          secretAccessKey: String(minioCfg.secretKey || "nexuspassword"),
        },
        region: "us-east-1",
        forcePathStyle: true,
      })
    : null;

  async function getRun(workflow_run_id) {
    const row = await pool.query("SELECT * FROM workflow_runs WHERE workflow_run_id=$1", [workflow_run_id]);
    return row.rows[0] || null;
  }

  async function getSteps(workflow_run_id) {
    const row = await pool.query(
      "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 ORDER BY step_index ASC",
      [workflow_run_id]
    );
    return row.rows || [];
  }

  function buildStepPayload({ run, stepDef, stepIndex }) {
    const input = parseJsonSafe(run.input_json, {});
    const artifactRoot = pathForRunArtifacts(run.run_id);
    const contract = STEP_CONTRACTS[stepDef.id] || null;
    const payload = {
      ...(input.step_payloads?.[stepDef.id] || {}),
      ...(input.default_payload || {}),
      project_type: run.project_type,
      workflow_id: run.workflow_id,
      workflow_run_id: run.workflow_run_id,
      role: stepDef.role,
      step_id: stepDef.id,
      step_index: stepIndex,
      run_id: run.run_id,
      artifact_root: artifactRoot,
      expected_artifacts: contract?.required_artifacts || [],
    };

    if (stepDef.tool === "coding.delegate") {
      payload.task_prompt = payload.task_prompt || buildStepPrompt({ run, stepDef, input, payload });
      if (!payload.prompt) payload.prompt = payload.task_prompt;
      if (input.provider && !payload.provider) payload.provider = input.provider;
      if (input.model && !payload.model) payload.model = input.model;
    }

    if (stepDef.gate === "acceptance") {
      const suiteId = registry.project_types?.[run.project_type]?.acceptance_suite;
      const suite = suiteId ? registry.acceptance_suites?.[suiteId] : null;
      const commands = Array.isArray(suite?.commands) ? suite.commands.filter(Boolean) : [];
      if (!payload.command && commands.length > 0) {
        payload.command = commands.join(" && ");
      }
      payload.acceptance_suite_id = suiteId || payload.acceptance_suite_id || "";
      payload.required_reports = payload.required_reports || suite?.required_reports || [];
      payload.acceptance_context = {
        step_id: stepDef.id,
        role: stepDef.role,
        goal: String(input.goal || ""),
        required_artifacts: payload.expected_artifacts || [],
      };
    }
    return payload;
  }

  function pathForRunArtifacts(run_id) {
    return `artifacts/release/${run_id || "unknown-run"}`;
  }

  async function failWorkflowRun({ run, stepDef, stepIndex, error_code, error_message }) {
    await pool.query(
      `UPDATE workflow_runs
       SET status='failed', error_code=$2, error_message=$3, updated_at=NOW()
       WHERE workflow_run_id=$1`,
      [run.workflow_run_id, String(error_code || "WORKFLOW_FAILED"), String(error_message || "workflow failed")]
    );
    if (Number.isInteger(stepIndex)) {
      await pool.query(
        `UPDATE workflow_steps
         SET status='failed', error_code=$3, ended_at=NOW(), updated_at=NOW()
         WHERE workflow_run_id=$1 AND step_index=$2 AND status <> 'succeeded'`,
        [run.workflow_run_id, stepIndex, String(error_code || "WORKFLOW_FAILED")]
      );
    }
    if (run.run_id) {
      await pool.query("UPDATE runs SET status='failed' WHERE run_id=$1", [run.run_id]).catch(() => {});
    }
    await recordEvent(run.workflow_run_id, "workflow.failed", {
      workflow_run_id: run.workflow_run_id,
      step_id: stepDef?.id || null,
      step_index: Number.isInteger(stepIndex) ? stepIndex : null,
      error_code: String(error_code || "WORKFLOW_FAILED"),
      error: String(error_message || "workflow failed"),
    });
  }

  async function succeedWorkflowRun(run) {
    const pack = await generateArtifactPack(run);
    if (!pack.ok) {
      await pool.query(
        "UPDATE workflow_runs SET status='failed', error_code=$2, error_message=$3, updated_at=NOW() WHERE workflow_run_id=$1",
        [run.workflow_run_id, "ARTIFACT_INCOMPLETE", pack.error || "artifact pack incomplete"]
      );
      if (run.run_id) {
        await pool.query("UPDATE runs SET status='failed' WHERE run_id=$1", [run.run_id]).catch(() => {});
      }
      await recordEvent(run.workflow_run_id, "artifact.pack.failed", {
        workflow_run_id: run.workflow_run_id,
        error_code: "ARTIFACT_INCOMPLETE",
        reasons: pack.reasons || [],
      });
      return;
    }

    await pool.query(
      "UPDATE workflow_runs SET status='succeeded', updated_at=NOW() WHERE workflow_run_id=$1",
      [run.workflow_run_id]
    );
    if (run.run_id) {
      await pool.query("UPDATE runs SET status='completed' WHERE run_id=$1", [run.run_id]).catch(() => {});
    }
    await recordEvent(run.workflow_run_id, "workflow.succeeded", { workflow_run_id: run.workflow_run_id });
    await recordEvent(run.workflow_run_id, "artifact.pack.generated", {
      workflow_run_id: run.workflow_run_id,
      run_manifest: pack.run_manifest_path,
      release_summary: pack.summary_path,
    });
  }

  function inferProjectArtifactCoverage(steps) {
    const byId = Object.fromEntries((steps || []).map((s) => [String(s.step_id || ""), s]));
    const has = {
      spec: !!byId.pm_spec,
      arch: !!byId.arch_design,
      diff: !!byId.impl_fe || !!byId.impl_be,
      verification: !!byId.qa_verify,
      run_summary: !!byId.release_pack,
      run_manifest: true,
    };
    return has;
  }

  function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath, { recursive: true });
  }

  function writeJsonFile(targetPath, obj) {
    ensureDir(path.dirname(targetPath));
    fs.writeFileSync(targetPath, JSON.stringify(obj, null, 2), "utf8");
  }

  async function archiveReleasePackToMinio({ run, manifestPath, summaryPath }) {
    if (!s3) return [];
    const out = [];
    const files = [manifestPath, summaryPath].filter((p) => p && fs.existsSync(p));
    for (const filePath of files) {
      try {
        const data = fs.readFileSync(filePath);
        const ext = path.extname(filePath).replace(/^\./, "") || "bin";
        const key = `release/${String(run.run_id || run.workflow_run_id)}/${path.basename(filePath, path.extname(filePath))}.${ext}`;
        await s3.send(
          new PutObjectCommand({
            Bucket: minioBucket,
            Key: key,
            Body: data,
            ContentType: filePath.endsWith(".json") ? "application/json" : "text/markdown",
          })
        );
        const sha256 = crypto.createHash("sha256").update(data).digest("hex");
        out.push({
          object_key: key,
          bucket: minioBucket,
          sha256,
          mime_type: filePath.endsWith(".json") ? "application/json" : "text/markdown",
          file_size: Number(data.length || 0),
          local_path: filePath.replace(/\\/g, "/"),
        });
      } catch (err) {
        await recordEvent(run.workflow_run_id, "artifact.pack.minio.archive_failed", {
          file: filePath,
          error: err.message || String(err),
        });
      }
    }
    return out;
  }

  async function indexReleasePackToDb({ run, manifestPath, summaryPath, stepArtifacts, minioArchived = [] }) {
    const files = [manifestPath, summaryPath].filter((p) => p && fs.existsSync(p));
    for (const filePath of files) {
      try {
        const buf = fs.readFileSync(filePath);
        const sha256 = crypto.createHash("sha256").update(buf).digest("hex");
        const stat = fs.statSync(filePath);
        const relPath = path.relative(workspaceRoot, filePath).replace(/\\/g, "/");
        await pool.query(
          `INSERT INTO assets(task_id, object_key, sha256, mime_type, file_size, metadata_json)
           VALUES ($1,$2,$3,$4,$5,$6)`,
          [
            `workflow_run:${run.workflow_run_id}`,
            relPath,
            sha256,
            filePath.endsWith(".json") ? "application/json" : "text/markdown",
            Number(stat.size || 0),
            JSON.stringify({
              run_id: run.run_id,
              workflow_run_id: run.workflow_run_id,
              source: "release_pack_local",
            }),
          ]
        );
      } catch {}
    }

    for (const item of stepArtifacts || []) {
      for (const art of item.artifacts || []) {
        if (!art || (!art.object_key && !art.name)) continue;
        try {
          await pool.query(
            `INSERT INTO assets(task_id, object_key, sha256, mime_type, file_size, metadata_json)
             VALUES ($1,$2,$3,$4,$5,$6)`,
            [
              `workflow_run:${run.workflow_run_id}`,
              String(art.object_key || art.name || "unknown"),
              String(art.sha256 || ""),
              String(art.mime || "application/octet-stream"),
              Number(art.file_size || 0),
              JSON.stringify({
                run_id: run.run_id,
                workflow_run_id: run.workflow_run_id,
                source: "step_artifact_ref",
                step_index: Number(item.step_index),
                step_id: item.step_id,
                bucket: art.bucket || null,
              }),
            ]
          );
        } catch {}
      }
    }

    for (const art of minioArchived || []) {
      try {
        await pool.query(
          `INSERT INTO assets(task_id, object_key, sha256, mime_type, file_size, metadata_json)
           VALUES ($1,$2,$3,$4,$5,$6)`,
          [
            `workflow_run:${run.workflow_run_id}`,
            String(art.object_key || ""),
            String(art.sha256 || ""),
            String(art.mime_type || "application/octet-stream"),
            Number(art.file_size || 0),
            JSON.stringify({
              run_id: run.run_id,
              workflow_run_id: run.workflow_run_id,
              source: "release_pack_minio",
              bucket: art.bucket || minioBucket,
              local_path: art.local_path || null,
            }),
          ]
        );
      } catch {}
    }
  }

  async function generateArtifactPack(run) {
    const steps = await getSteps(run.workflow_run_id);
    const checkpointsRes = await pool.query(
      "SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, created_at FROM workflow_checkpoints WHERE workflow_run_id=$1 ORDER BY step_index ASC",
      [run.workflow_run_id]
    );
    const checkpoints = checkpointsRes.rows || [];
    const reasons = [];

    if (!Array.isArray(steps) || steps.length === 0) reasons.push("no workflow steps found");
    if (steps.some((s) => String(s.status) !== "succeeded")) reasons.push("not all steps are succeeded");
    if (checkpoints.length === 0) reasons.push("no checkpoints generated");
    if (checkpoints.length < steps.length) reasons.push("checkpoint count lower than step count");

    const required = registry.project_types?.[run.project_type]?.required_artifacts || [];
    const coverage = inferProjectArtifactCoverage(steps);
    for (const req of required) {
      if (!coverage[req]) reasons.push(`required artifact missing: ${req}`);
    }

    const cpArtifactByStep = new Map();
    for (const cp of checkpoints) {
      const key = `${Number(cp.step_index)}:${String(cp.step_id || "")}`;
      cpArtifactByStep.set(key, parseJsonSafe(cp.artifact_refs_json, []));
    }
    const stepArtifacts = steps.map((s) => {
      const key = `${Number(s.step_index)}:${String(s.step_id || "")}`;
      return {
        step_index: Number(s.step_index),
        step_id: s.step_id,
        task_id: s.task_id || "",
        artifacts: cpArtifactByStep.get(key) || [],
      };
    });

    const releaseRoot = path.join(workspaceRoot, "artifacts", "release", String(run.run_id || run.workflow_run_id));
    const manifestPath = path.join(releaseRoot, "meta", "run_manifest.json");
    const summaryPath = path.join(releaseRoot, "summary", "run_summary.md");
    const manifest = {
      workflow_run_id: run.workflow_run_id,
      run_id: run.run_id,
      workflow_id: run.workflow_id,
      project_type: run.project_type,
      status: reasons.length === 0 ? "succeeded" : "failed",
      generated_at: new Date().toISOString(),
      required_artifacts: required,
      artifact_coverage: coverage,
      step_artifacts: stepArtifacts,
      steps: steps.map((s) => ({
        step_index: Number(s.step_index),
        step_id: s.step_id,
        role_name: s.role_name,
        tool_name: s.tool_name,
        gate_name: s.gate_name,
        task_id: s.task_id,
        status: s.status,
        checkpoint_id: s.checkpoint_id,
        error_code: s.error_code || null,
      })),
      checkpoints: checkpoints.map((c) => ({
        checkpoint_id: c.checkpoint_id,
        step_index: Number(c.step_index),
        step_id: c.step_id,
        task_id: c.task_id,
        workspace_hash: c.workspace_hash,
      })),
      reasons,
    };

    try {
      writeJsonFile(manifestPath, manifest);
      ensureDir(path.dirname(summaryPath));
      const summaryLines = [
        `# Run Summary`,
        ``,
        `- run_id: ${run.run_id}`,
        `- workflow_run_id: ${run.workflow_run_id}`,
        `- workflow_id: ${run.workflow_id}`,
        `- project_type: ${run.project_type}`,
        `- status: ${manifest.status}`,
        `- generated_at: ${manifest.generated_at}`,
        ``,
        `## Steps`,
        ...manifest.steps.map((s) => `- [${s.status === "succeeded" ? "OK" : "FAIL"}] ${s.step_index}:${s.step_id} (${s.tool_name})`),
      ];
      fs.writeFileSync(summaryPath, summaryLines.join("\n"), "utf8");
      const minioArchived = await archiveReleasePackToMinio({
        run,
        manifestPath,
        summaryPath,
      });
      await indexReleasePackToDb({
        run,
        manifestPath,
        summaryPath,
        stepArtifacts,
        minioArchived,
      });
    } catch (err) {
      reasons.push(`artifact write failed: ${err.message}`);
    }

    const validator = validateArtifactPack({
      run,
      steps,
      checkpoints,
      manifestPath,
      summaryPath,
      registry,
    });
    const allReasons = [...reasons, ...(validator.reasons || [])];
    if (allReasons.length > 0) {
      return {
        ok: false,
        error: "artifact pack incomplete",
        reasons: [...new Set(allReasons)],
        run_manifest_path: manifestPath,
        summary_path: summaryPath,
        validator,
      };
    }
    return { ok: true, run_manifest_path: manifestPath, summary_path: summaryPath, validator };
  }

  async function validateRunArtifactPack(workflow_run_id) {
    const run = await getRun(workflow_run_id);
    if (!run) {
      const err = new Error(`workflow_run '${workflow_run_id}' not found`);
      err.code = "WORKFLOW_RUN_NOT_FOUND";
      throw err;
    }
    const steps = await getSteps(workflow_run_id);
    const cps = await pool.query(
      "SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, created_at FROM workflow_checkpoints WHERE workflow_run_id=$1 ORDER BY step_index ASC",
      [workflow_run_id]
    );
    const releaseRoot = path.join(workspaceRoot, "artifacts", "release", String(run.run_id || run.workflow_run_id));
    const manifestPath = path.join(releaseRoot, "meta", "run_manifest.json");
    const summaryPath = path.join(releaseRoot, "summary", "run_summary.md");
    return validateArtifactPack({
      run,
      steps,
      checkpoints: cps.rows || [],
      manifestPath,
      summaryPath,
      registry,
    });
  }

  async function archiveRunArtifactPack(workflow_run_id) {
    const run = await getRun(workflow_run_id);
    if (!run) {
      const err = new Error(`workflow_run '${workflow_run_id}' not found`);
      err.code = "WORKFLOW_RUN_NOT_FOUND";
      throw err;
    }
    const releaseRoot = path.join(workspaceRoot, "artifacts", "release", String(run.run_id || run.workflow_run_id));
    const manifestPath = path.join(releaseRoot, "meta", "run_manifest.json");
    const summaryPath = path.join(releaseRoot, "summary", "run_summary.md");
    if (!fs.existsSync(manifestPath) || !fs.existsSync(summaryPath)) {
      const err = new Error("ARTIFACT_INCOMPLETE: release pack files missing");
      err.code = "ARTIFACT_INCOMPLETE";
      throw err;
    }
    const steps = await getSteps(workflow_run_id);
    const cps = await pool.query(
      "SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, created_at FROM workflow_checkpoints WHERE workflow_run_id=$1 ORDER BY step_index ASC",
      [workflow_run_id]
    );
    const cpArtifactByStep = new Map();
    for (const cp of cps.rows || []) {
      const key = `${Number(cp.step_index)}:${String(cp.step_id || "")}`;
      cpArtifactByStep.set(key, parseJsonSafe(cp.artifact_refs_json, []));
    }
    const stepArtifacts = (steps || []).map((s) => {
      const key = `${Number(s.step_index)}:${String(s.step_id || "")}`;
      return {
        step_index: Number(s.step_index),
        step_id: s.step_id,
        task_id: s.task_id || "",
        artifacts: cpArtifactByStep.get(key) || [],
      };
    });
    const archived = await archiveReleasePackToMinio({ run, manifestPath, summaryPath });
    await indexReleasePackToDb({
      run,
      manifestPath,
      summaryPath,
      stepArtifacts,
      minioArchived: archived,
    });
    await recordEvent(workflow_run_id, "artifact.pack.minio.archived", {
      workflow_run_id,
      count: archived.length,
      bucket: minioBucket,
    });
    return { ok: true, count: archived.length, objects: archived };
  }

  async function dispatchStepByIndex(workflow_run_id, stepIndex, context = null) {
    const run = await getRun(workflow_run_id);
    if (!run) throw new Error(`workflow_run '${workflow_run_id}' not found`);
    if (run.status === "failed" || run.status === "succeeded") return { skipped: true, reason: `run status ${run.status}` };

    const wf = registry.workflows?.[run.workflow_id];
    if (!wf || !Array.isArray(wf.steps)) {
      await failWorkflowRun({
        run,
        stepDef: null,
        stepIndex,
        error_code: "WORKFLOW_DEF_MISSING",
        error_message: `workflow '${run.workflow_id}' not found in registry`,
      });
      return { failed: true, error_code: "WORKFLOW_DEF_MISSING" };
    }

    const stepDef = wf.steps[stepIndex];
    if (!stepDef) {
      await succeedWorkflowRun(run);
      return { completed: true };
    }

    const stepRowRes = await pool.query(
      "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2",
      [workflow_run_id, stepIndex]
    );
    const stepRow = stepRowRes.rows[0];
    if (!stepRow) {
      await failWorkflowRun({
        run,
        stepDef,
        stepIndex,
        error_code: "STEP_STATE_MISSING",
        error_message: `step state missing for ${stepDef.id}`,
      });
      return { failed: true, error_code: "STEP_STATE_MISSING" };
    }

    const stepStatus = normalizeStepStatus(stepRow.status);
    if (!["pending", "failed"].includes(stepStatus)) {
      return { skipped: true, reason: `step status ${stepStatus}` };
    }

    const payload = buildStepPayload({ run, stepDef, stepIndex });
    const risk = analyzeTaskRisk(stepDef.tool, payload);
    await recordEvent(workflow_run_id, "policy.gate.checked", {
      workflow_run_id,
      step_id: stepDef.id,
      step_index: stepIndex,
      tool_name: stepDef.tool,
      risk_level: risk.risk_level,
      requires_approval: Boolean(risk.requires_approval),
      reasons: risk.reasons || [],
    });

    try {
      const enq = await enqueueTask({
        tool_name: stepDef.tool,
        payload,
        run_id: run.run_id,
        risk_level: risk.risk_level,
        idempotency_key: makeIdempotencyKey(run.run_id, stepDef.tool, {
          workflow_run_id,
          step_id: stepDef.id,
          step_index: stepIndex,
          payload,
        }),
        context,
      });

      await pool.query(
        `UPDATE workflow_steps
         SET status=$3,
             task_id=$4,
             risk_level=$5,
             approval_required=$6,
             approval_reasons_json=$7,
             started_at=COALESCE(started_at, NOW()),
             updated_at=NOW()
         WHERE workflow_run_id=$1 AND step_index=$2`,
        [
          workflow_run_id,
          stepIndex,
          enq.waiting_approval ? "waiting_approval" : "queued",
          enq.task_id,
          risk.risk_level || "low",
          Boolean(enq.waiting_approval),
          JSON.stringify(risk.reasons || []),
        ]
      );
      await pool.query(
        "UPDATE workflow_runs SET current_step_index=$2, status='running', updated_at=NOW() WHERE workflow_run_id=$1",
        [workflow_run_id, stepIndex]
      );
      await recordEvent(enq.task_id, "workflow.step.dispatched", {
        workflow_run_id,
        step_id: stepDef.id,
        step_index: stepIndex,
        waiting_approval: Boolean(enq.waiting_approval),
      });
      return { ok: true, task_id: enq.task_id, waiting_approval: Boolean(enq.waiting_approval), step_id: stepDef.id };
    } catch (err) {
      const code = String(err?.code || "");
      await failWorkflowRun({
        run,
        stepDef,
        stepIndex,
        error_code: code || "STEP_DISPATCH_FAILED",
        error_message: err?.message || "step dispatch failed",
      });
      return { failed: true, error_code: code || "STEP_DISPATCH_FAILED", error: err?.message || "step dispatch failed" };
    }
  }

  async function createCheckpoint({ workflow_run_id, stepIndex, step_id, task_id, status, output }) {
    const artifacts = Array.isArray(output?.artifacts)
      ? output.artifacts.map((a) => ({
          bucket: a?.bucket || null,
          object_key: a?.object_key || null,
          name: a?.name || null,
          sha256: a?.sha256 || null,
          mime: a?.mime || null,
        }))
      : [];
    const workspace_hash = buildWorkspaceHash({
      workflow_run_id,
      step_index: stepIndex,
      task_id,
      status,
      artifacts,
    });
    const checkpoint_id = uuidv4();
    await pool.query(
      `INSERT INTO workflow_checkpoints(checkpoint_id, workflow_run_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, checkpoint_json)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
      [
        checkpoint_id,
        workflow_run_id,
        stepIndex,
        step_id,
        task_id || "",
        workspace_hash,
        JSON.stringify(artifacts),
        JSON.stringify({
          workflow_run_id,
          step_index: stepIndex,
          step_id,
          task_id,
          status,
          artifacts,
        }),
      ]
    );
    await pool.query(
      `UPDATE workflow_steps
       SET checkpoint_id=$3, updated_at=NOW()
       WHERE workflow_run_id=$1 AND step_index=$2`,
      [workflow_run_id, stepIndex, checkpoint_id]
    );
    await pool.query(
      "UPDATE workflow_runs SET last_checkpoint_id=$2, updated_at=NOW() WHERE workflow_run_id=$1",
      [workflow_run_id, checkpoint_id]
    );
    return { checkpoint_id, workspace_hash, artifacts };
  }

  async function startWorkflowRun({ workflow_id, project_type, run_id, input = {}, context = null }) {
    const wf = registry.workflows?.[workflow_id];
    if (!wf) {
      const err = new Error(`workflow '${workflow_id}' not found`);
      err.code = "WORKFLOW_NOT_FOUND";
      throw err;
    }
    const resolvedProjectType = String(project_type || wf.project_type || "");
    if (!registry.project_types?.[resolvedProjectType]) {
      const err = new Error(`project_type '${resolvedProjectType}' not found`);
      err.code = "PROJECT_TYPE_NOT_FOUND";
      throw err;
    }
    if (wf.project_type && wf.project_type !== resolvedProjectType) {
      const err = new Error(
        `workflow '${workflow_id}' project_type mismatch: expected '${wf.project_type}', got '${resolvedProjectType}'`
      );
      err.code = "WORKFLOW_PROJECT_TYPE_MISMATCH";
      throw err;
    }
    const steps = Array.isArray(wf.steps) ? wf.steps : [];
    if (steps.length === 0) {
      const err = new Error(`workflow '${workflow_id}' has no steps`);
      err.code = "WORKFLOW_EMPTY";
      throw err;
    }

    const workflow_run_id = uuidv4();
    await pool.query(
      `INSERT INTO workflow_runs(workflow_run_id, run_id, workflow_id, project_type, status, current_step_index, input_json)
       VALUES ($1,$2,$3,$4,'running',0,$5)`,
      [workflow_run_id, run_id, workflow_id, resolvedProjectType, JSON.stringify(input || {})]
    );

    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      await pool.query(
        `INSERT INTO workflow_steps(workflow_run_id, step_index, step_id, role_name, tool_name, gate_name, status)
         VALUES ($1,$2,$3,$4,$5,$6,'pending')`,
        [workflow_run_id, i, String(step.id || `step_${i}`), String(step.role || ""), String(step.tool || ""), String(step.gate || "")]
      );
    }

    await recordEvent(workflow_run_id, "workflow.started", {
      workflow_run_id,
      workflow_id,
      project_type: resolvedProjectType,
      run_id,
      steps: steps.map((s, idx) => ({ step_index: idx, step_id: s.id, role: s.role, tool: s.tool, gate: s.gate })),
    });

    const first = await dispatchStepByIndex(workflow_run_id, 0, context);
    return { workflow_run_id, run_id, workflow_id, project_type: resolvedProjectType, first_step: first };
  }

  async function handleTaskClaimed(task_id) {
    const row = await pool.query("SELECT payload_json FROM tasks WHERE task_id=$1", [task_id]);
    if (row.rows.length === 0) return { handled: false };
    const payload = parseJsonSafe(row.rows[0].payload_json, {});
    const workflow_run_id = payload.workflow_run_id;
    const step_index = Number(payload.step_index);
    if (!workflow_run_id || !Number.isInteger(step_index)) return { handled: false };
    await pool.query(
      `UPDATE workflow_steps
       SET status='running', started_at=COALESCE(started_at, NOW()), updated_at=NOW()
       WHERE workflow_run_id=$1 AND step_index=$2`,
      [workflow_run_id, step_index]
    );
    return { handled: true, workflow_run_id, step_index };
  }

  async function handleTaskApproved(task_id) {
    const row = await pool.query("SELECT payload_json FROM tasks WHERE task_id=$1", [task_id]);
    if (row.rows.length === 0) return { handled: false };
    const payload = parseJsonSafe(row.rows[0].payload_json, {});
    const workflow_run_id = payload.workflow_run_id;
    const step_index = Number(payload.step_index);
    if (!workflow_run_id || !Number.isInteger(step_index)) return { handled: false };
    await pool.query(
      `UPDATE workflow_steps
       SET status='queued', updated_at=NOW()
       WHERE workflow_run_id=$1 AND step_index=$2`,
      [workflow_run_id, step_index]
    );
    await recordEvent(task_id, "workflow.step.approval.approved", { workflow_run_id, step_index });
    return { handled: true, workflow_run_id, step_index };
  }

  async function handleTaskRejected(task_id, reason = "") {
    const taskRes = await pool.query("SELECT task_id, run_id, payload_json FROM tasks WHERE task_id=$1", [task_id]);
    if (taskRes.rows.length === 0) return { handled: false };
    const task = taskRes.rows[0];
    const payload = parseJsonSafe(task.payload_json, {});
    const workflow_run_id = payload.workflow_run_id;
    const step_index = Number(payload.step_index);
    const step_id = String(payload.step_id || "");
    if (!workflow_run_id || !Number.isInteger(step_index)) return { handled: false };

    await pool.query(
      `UPDATE workflow_steps
       SET status='failed',
           error_code='APPROVAL_REJECTED',
           result_json=$3,
           ended_at=NOW(),
           updated_at=NOW()
       WHERE workflow_run_id=$1 AND step_index=$2`,
      [workflow_run_id, step_index, JSON.stringify({ rejected: true, reason: String(reason || "") })]
    );
    const run = await getRun(workflow_run_id);
    if (run) {
      await failWorkflowRun({
        run,
        stepDef: { id: step_id },
        stepIndex: step_index,
        error_code: "APPROVAL_REJECTED",
        error_message: String(reason || "approval rejected"),
      });
    }
    await recordEvent(task_id, "workflow.step.approval.rejected", { workflow_run_id, step_index, reason: String(reason || "") });
    return { handled: true, workflow_run_id, step_index };
  }

  async function handleTaskTerminal({ task_id, status, output, error_code }) {
    const row = await pool.query(
      "SELECT task_id, run_id, payload_json FROM tasks WHERE task_id=$1",
      [task_id]
    );
    if (row.rows.length === 0) return { handled: false };
    const task = row.rows[0];
    const payload = parseJsonSafe(task.payload_json, {});
    const workflow_run_id = payload.workflow_run_id;
    const step_index = Number(payload.step_index);
    const step_id = String(payload.step_id || "");
    if (!workflow_run_id || !Number.isInteger(step_index)) return { handled: false };

    const run = await getRun(workflow_run_id);
    if (!run) return { handled: false };
    const stepRow = await pool.query(
      "SELECT gate_name FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2 LIMIT 1",
      [workflow_run_id, step_index]
    );
    const gateName = String(stepRow.rows[0]?.gate_name || "");
    if (status === "succeeded") {
      const checkpoint = await createCheckpoint({
        workflow_run_id,
        stepIndex: step_index,
        step_id,
        task_id,
        status,
        output: output || {},
      });

      await pool.query(
        `UPDATE workflow_steps
         SET status='succeeded',
             result_json=$3,
             error_code=NULL,
             ended_at=NOW(),
             checkpoint_id=$4,
             updated_at=NOW()
         WHERE workflow_run_id=$1 AND step_index=$2`,
        [workflow_run_id, step_index, JSON.stringify(output || {}), checkpoint.checkpoint_id]
      );

      const nextResult = await dispatchStepByIndex(workflow_run_id, step_index + 1);
      return {
        handled: true,
        workflow_run_id,
        step_index,
        checkpoint_id: checkpoint.checkpoint_id,
        next: nextResult,
      };
    }

    await pool.query(
      `UPDATE workflow_steps
       SET status='failed',
           result_json=$3,
           error_code=$4,
           ended_at=NOW(),
           updated_at=NOW()
       WHERE workflow_run_id=$1 AND step_index=$2`,
      [
        workflow_run_id,
        step_index,
        JSON.stringify(output || {}),
        String(error_code || (gateName === "acceptance" ? "ACCEPTANCE_FAILED" : "STEP_FAILED")),
      ]
    );
    await failWorkflowRun({
      run,
      stepDef: { id: step_id },
      stepIndex: step_index,
      error_code: String(error_code || (gateName === "acceptance" ? "ACCEPTANCE_FAILED" : "STEP_FAILED")),
      error_message: String(error_code || (gateName === "acceptance" ? "acceptance gate failed" : "step failed")),
    });
    return { handled: true, workflow_run_id, step_index };
  }

  async function issueResumeToken(workflow_run_id) {
    const run = await getRun(workflow_run_id);
    if (!run) {
      const err = new Error(`workflow_run '${workflow_run_id}' not found`);
      err.code = "WORKFLOW_RUN_NOT_FOUND";
      throw err;
    }
    if (!run.last_checkpoint_id) {
      const err = new Error("RESUME_INVALID: no checkpoint available");
      err.code = "RESUME_INVALID";
      throw err;
    }
    const cpRow = await pool.query(
      "SELECT checkpoint_id, step_index, workspace_hash FROM workflow_checkpoints WHERE checkpoint_id=$1",
      [run.last_checkpoint_id]
    );
    const cp = cpRow.rows[0];
    if (!cp) {
      const err = new Error("RESUME_INVALID: checkpoint not found");
      err.code = "RESUME_INVALID";
      throw err;
    }
    const nowSec = Math.floor(Date.now() / 1000);
    const payload = {
      workflow_run_id,
      checkpoint_id: cp.checkpoint_id,
      step_index: Number(cp.step_index),
      workspace_hash: cp.workspace_hash,
      iat: nowSec,
      exp: nowSec + Math.max(300, Number(resumeTokenTtlSec || 86400)),
    };
    const token = signResumePayload(payload, resumeTokenSecret);
    await pool.query(
      "UPDATE workflow_runs SET resume_token=$2, updated_at=NOW() WHERE workflow_run_id=$1",
      [workflow_run_id, token]
    );
    return { resume_token: token, expires_at: payload.exp, checkpoint_id: cp.checkpoint_id, step_index: payload.step_index };
  }

  async function resumeFromToken(workflow_run_id, resume_token, context = null) {
    const checked = verifyResumePayload(resume_token, resumeTokenSecret);
    if (!checked.ok) {
      const err = new Error(checked.error);
      err.code = "RESUME_INVALID";
      throw err;
    }
    const payload = checked.payload;
    if (payload.workflow_run_id !== workflow_run_id) {
      const err = new Error("RESUME_INVALID: workflow_run mismatch");
      err.code = "RESUME_INVALID";
      throw err;
    }
    const run = await getRun(workflow_run_id);
    if (!run) {
      const err = new Error(`workflow_run '${workflow_run_id}' not found`);
      err.code = "WORKFLOW_RUN_NOT_FOUND";
      throw err;
    }
    const cpRow = await pool.query(
      "SELECT * FROM workflow_checkpoints WHERE checkpoint_id=$1 AND workflow_run_id=$2",
      [payload.checkpoint_id, workflow_run_id]
    );
    const cp = cpRow.rows[0];
    if (!cp || cp.workspace_hash !== payload.workspace_hash) {
      const err = new Error("RESUME_INVALID: checkpoint mismatch");
      err.code = "RESUME_INVALID";
      throw err;
    }

    const steps = await getSteps(workflow_run_id);
    const nextStep = steps.find((s) => Number(s.step_index) > Number(cp.step_index) && normalizeStepStatus(s.status) !== "succeeded");
    if (!nextStep) {
      const err = new Error("RESUME_INVALID: no resumable step");
      err.code = "RESUME_INVALID";
      throw err;
    }
    await pool.query(
      `UPDATE workflow_steps
       SET status='pending', task_id=NULL, updated_at=NOW()
       WHERE workflow_run_id=$1 AND step_index=$2`,
      [workflow_run_id, Number(nextStep.step_index)]
    );
    await pool.query(
      "UPDATE workflow_runs SET status='running', error_code=NULL, error_message=NULL, updated_at=NOW() WHERE workflow_run_id=$1",
      [workflow_run_id]
    );
    const dispatch = await dispatchStepByIndex(workflow_run_id, Number(nextStep.step_index), context);
    await recordEvent(workflow_run_id, "workflow.resumed", {
      workflow_run_id,
      step_index: Number(nextStep.step_index),
      checkpoint_id: cp.checkpoint_id,
    });
    return { ok: true, workflow_run_id, resumed_step_index: Number(nextStep.step_index), dispatch };
  }

  async function getWorkflowRunStatus(workflow_run_id) {
    const run = await getRun(workflow_run_id);
    if (!run) return null;
    const steps = await getSteps(workflow_run_id);
    const cps = await pool.query(
      "SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash, created_at FROM workflow_checkpoints WHERE workflow_run_id=$1 ORDER BY created_at ASC",
      [workflow_run_id]
    );
    return {
      run,
      steps,
      checkpoints: cps.rows || [],
    };
  }

  return {
    startWorkflowRun,
    handleTaskClaimed,
    handleTaskApproved,
    handleTaskRejected,
    handleTaskTerminal,
    issueResumeToken,
    resumeFromToken,
    getWorkflowRunStatus,
    validateRunArtifactPack,
    archiveRunArtifactPack,
  };
}
