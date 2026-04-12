/**
 * delete_semantics.mjs — verify DELETE endpoints return 404 for missing rows.
 *
 * Strategy:
 *   1. Parse plan/interfaces.md for `## DELETE /path` headings.
 *   2. Start the BE server on a test port (or reuse the one smoke_test started).
 *   3. For each DELETE endpoint, issue DELETE with an obviously-nonexistent id.
 *   4. Expect 404 (or 400 with validation failure). Anything 2xx is a fail.
 *
 * If no DELETE endpoints are declared, the scanner skips with status=pass.
 * If the server is not reachable, the scanner returns status=warning with reason.
 */

import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { httpRequest, waitForServer } from "../lib/http_client.mjs";

export async function run({ workspaceRoot, artifactRoot, port = 13101 }) {
  const started = Date.now();
  const findings = [];
  const interfacesPath = path.resolve(workspaceRoot, artifactRoot, "plan/interfaces.md");

  if (!fs.existsSync(interfacesPath)) {
    return skipResult("plan/interfaces.md not found", started);
  }

  const deleteEndpoints = parseDeleteEndpoints(fs.readFileSync(interfacesPath, "utf8"));
  if (deleteEndpoints.length === 0) {
    return {
      scanner_id: "delete_semantics",
      status: "pass",
      findings: [],
      summary: { critical: 0, high: 0, medium: 0, low: 0, total: 0 },
      scanned_endpoints: [],
      note: "No DELETE endpoints declared in interfaces.md",
      duration_ms: Date.now() - started,
    };
  }

  const serverDir = path.resolve(workspaceRoot, artifactRoot, "impl/be_changes");
  const serverFile = path.resolve(serverDir, "server.js");
  if (!fs.existsSync(serverFile)) {
    return skipResult("impl/be_changes/server.js not found", started, "warning");
  }

  let serverProc = null;
  try {
    serverProc = await startServer(serverDir, port);
    const ready = await waitForServer(`http://127.0.0.1:${port}/`, { timeoutMs: 8000 });
    if (!ready) {
      return skipResult(`server did not become ready on port ${port} within 8s`, started, "warning");
    }

    for (const ep of deleteEndpoints) {
      const probeUrl = buildProbeUrl(port, ep.path);
      const res = await httpRequest({ method: "DELETE", url: probeUrl, timeoutMs: 3000 });
      if (res.status === 404 || res.status === 400) {
        // correct behaviour
        continue;
      }
      if (res.status >= 200 && res.status < 300) {
        findings.push({
          severity: "high",
          code: "DELETE_MISSING_404",
          endpoint: `DELETE ${ep.path}`,
          actual_status: res.status,
          expected_status: 404,
          probe_url: probeUrl,
          detail: `DELETE ${ep.path} returned ${res.status} for a non-existent id; must return 404.`,
          fix_hint: "Check existence before delete: if (!findById(id)) return res.status(404).json({error:'not_found'});",
        });
      } else if (res.status === 500) {
        findings.push({
          severity: "medium",
          code: "DELETE_CRASHES_ON_MISSING",
          endpoint: `DELETE ${ep.path}`,
          actual_status: res.status,
          probe_url: probeUrl,
          detail: `DELETE ${ep.path} returned 500 (server crash) for missing id; should return 404 gracefully.`,
        });
      } else if (res.status === 0) {
        findings.push({
          severity: "low",
          code: "DELETE_PROBE_FAILED",
          endpoint: `DELETE ${ep.path}`,
          probe_url: probeUrl,
          detail: `Could not probe DELETE ${ep.path}: ${res.error || "unknown error"}`,
        });
      }
      // 405, 401, 403 and others are ambiguous; report as low
      else if ([401, 403, 405].includes(res.status)) {
        findings.push({
          severity: "low",
          code: "DELETE_UNEXPECTED_STATUS",
          endpoint: `DELETE ${ep.path}`,
          actual_status: res.status,
          detail: `DELETE ${ep.path} returned ${res.status}; expected 404 for missing id.`,
        });
      }
    }
  } finally {
    if (serverProc) {
      await stopServer(serverProc);
    }
  }

  const critical = findings.filter((f) => f.severity === "critical").length;
  const high = findings.filter((f) => f.severity === "high").length;
  const medium = findings.filter((f) => f.severity === "medium").length;
  const low = findings.filter((f) => f.severity === "low").length;

  let status;
  if (critical > 0 || high > 0) status = "fail";
  else if (medium > 0) status = "pass_with_warnings";
  else status = "pass";

  return {
    scanner_id: "delete_semantics",
    status,
    findings,
    summary: { critical, high, medium, low, total: findings.length },
    scanned_endpoints: deleteEndpoints.map((e) => e.path),
    duration_ms: Date.now() - started,
  };
}

function skipResult(reason, started, status = "pass") {
  return {
    scanner_id: "delete_semantics",
    status,
    findings: [],
    summary: { critical: 0, high: 0, medium: 0, low: 0, total: 0 },
    skipped: true,
    reason,
    duration_ms: Date.now() - started,
  };
}

function parseDeleteEndpoints(md) {
  const endpoints = [];
  // v3.7.1 (codex #5): also accept ###, Endpoint: prefix, backticks, lowercase.
  const re = /^(?:#{2,4}\s+|Endpoint\s*:\s*)\s*`?(DELETE|delete)`?\s+`?(\/[\w/\-:{}.]+)`?/gm;
  let m;
  const seen = new Set();
  while ((m = re.exec(md)) !== null) {
    const p = m[2].trim();
    if (seen.has(p)) continue;
    seen.add(p);
    endpoints.push({ method: "DELETE", path: p });
  }
  return endpoints;
}

function buildProbeUrl(port, rawPath) {
  // Replace :param and {param} with a clearly-fake id
  const cleaned = rawPath
    .replace(/:([a-zA-Z_]\w*)/g, "nonexistent-00000000")
    .replace(/\{([a-zA-Z_]\w*)\}/g, "nonexistent-00000000");
  return `http://127.0.0.1:${port}${cleaned}`;
}

function startServer(serverDir, port) {
  return new Promise((resolve) => {
    const proc = spawn("node", ["server.js"], {
      cwd: serverDir,
      env: { ...process.env, PORT: String(port) },
      stdio: ["ignore", "pipe", "pipe"],
      detached: false,
    });
    proc.stdout.on("data", () => {});
    proc.stderr.on("data", () => {});
    proc.on("error", () => {});
    // return immediately; waitForServer handles readiness
    setTimeout(() => resolve(proc), 100);
  });
}

/**
 * v3.7.1 (codex #4): wait for server process to actually exit, not just signal it.
 * Prevents the next scanner from racing a still-exiting server on an overlapping port.
 */
async function stopServer(proc, timeoutMs = 3000) {
  return new Promise((resolve) => {
    if (!proc || proc.exitCode !== null) return resolve();
    const timer = setTimeout(() => {
      try { proc.kill("SIGKILL"); } catch { /* ignore */ }
      resolve();
    }, timeoutMs);
    proc.once("exit", () => { clearTimeout(timer); resolve(); });
    try { proc.kill("SIGTERM"); } catch { /* ignore */ }
  });
}
