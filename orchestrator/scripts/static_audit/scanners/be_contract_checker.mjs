/**
 * be_contract_checker.mjs — verify BE input validation on POST/PUT endpoints.
 *
 * Strategy:
 *   1. Parse plan/interfaces.md for ## METHOD /path headings + extract request fields.
 *   2. Start server.js on test port.
 *   3. For each POST/PUT endpoint:
 *      - Send empty body → expect 400
 *      - Send body with known-bad enum value if enum fields detected → expect 400
 *   4. Failures = missing input validation.
 *
 * This scanner shares server-start logic with delete_semantics, but uses a
 * separate port so they can run independently.
 */

import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { httpRequest, waitForServer } from "../lib/http_client.mjs";

export async function run({ workspaceRoot, artifactRoot, port = 13102 }) {
  const started = Date.now();
  const findings = [];
  const interfacesPath = path.resolve(workspaceRoot, artifactRoot, "plan/interfaces.md");

  if (!fs.existsSync(interfacesPath)) {
    return skipResult("plan/interfaces.md not found", started);
  }

  const md = fs.readFileSync(interfacesPath, "utf8");
  const writeEndpoints = parseWriteEndpoints(md);
  if (writeEndpoints.length === 0) {
    return {
      scanner_id: "be_contract_checker",
      status: "pass",
      findings: [],
      summary: { critical: 0, high: 0, medium: 0, low: 0, total: 0 },
      note: "No POST/PUT endpoints declared",
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
    serverProc = startServer(serverDir, port);
    const ready = await waitForServer(`http://127.0.0.1:${port}/`, { timeoutMs: 8000 });
    if (!ready) {
      return skipResult(`server did not become ready on port ${port} within 8s`, started, "warning");
    }

    for (const ep of writeEndpoints) {
      const url = buildProbeUrl(port, ep.path);

      // Test 1: empty body → expect 400 (validation should reject)
      const emptyRes = await httpRequest({ method: ep.method, url, body: {}, timeoutMs: 3000 });
      if (emptyRes.status >= 200 && emptyRes.status < 300) {
        findings.push({
          severity: "high",
          code: "BE_MISSING_REQUIRED_VALIDATION",
          endpoint: `${ep.method} ${ep.path}`,
          probe: "empty_body",
          actual_status: emptyRes.status,
          expected_status: 400,
          probe_url: url,
          detail: `${ep.method} ${ep.path} accepted an empty body with status ${emptyRes.status}; should return 400 when required fields are missing.`,
          fix_hint: "Add input validation: check required fields, return 400 {error, field, message} if missing.",
        });
      }

      // Test 2: If the endpoint has enum fields, try an obviously-invalid enum value
      if (ep.enumFields.length > 0 && ep.requiredFields.length > 0) {
        const body = {};
        for (const f of ep.requiredFields) body[f] = `valid-${f}`;
        for (const { field } of ep.enumFields) body[field] = "___INVALID_ENUM_VALUE___";
        const enumRes = await httpRequest({ method: ep.method, url, body, timeoutMs: 3000 });
        if (enumRes.status >= 200 && enumRes.status < 300) {
          findings.push({
            severity: "medium",
            code: "BE_MISSING_ENUM_VALIDATION",
            endpoint: `${ep.method} ${ep.path}`,
            probe: "invalid_enum_value",
            actual_status: enumRes.status,
            expected_status: 400,
            probe_url: url,
            detail: `${ep.method} ${ep.path} accepted an invalid enum value for ${ep.enumFields.map((e) => e.field).join(",")} with status ${enumRes.status}; should reject with 400.`,
            fix_hint: `Whitelist-check enum values: if (!ALLOWED_STATUS.includes(status)) return res.status(400).json({...});`,
          });
        }
      }
    }
  } finally {
    if (serverProc) {
      try { serverProc.kill("SIGTERM"); } catch { /* ignore */ }
    }
  }

  const critical = findings.filter((f) => f.severity === "critical").length;
  const high = findings.filter((f) => f.severity === "high").length;
  const medium = findings.filter((f) => f.severity === "medium").length;
  const low = findings.filter((f) => f.severity === "low").length;
  const status = critical > 0 || high > 0 ? "fail" : medium > 0 ? "pass_with_warnings" : "pass";

  return {
    scanner_id: "be_contract_checker",
    status,
    findings,
    summary: { critical, high, medium, low, total: findings.length },
    scanned_endpoints: writeEndpoints.map((e) => `${e.method} ${e.path}`),
    duration_ms: Date.now() - started,
  };
}

function skipResult(reason, started, status = "pass") {
  return {
    scanner_id: "be_contract_checker",
    status,
    findings: [],
    summary: { critical: 0, high: 0, medium: 0, low: 0, total: 0 },
    skipped: true,
    reason,
    duration_ms: Date.now() - started,
  };
}

/**
 * Extract POST/PUT endpoints from interfaces.md, along with any required fields
 * and enum-looking fields (status/priority/type) noted in the section body.
 */
function parseWriteEndpoints(md) {
  const endpoints = [];
  const headerRe = /^##\s+(POST|PUT)\s+(\/[\w/\-:{}.]+)/gm;
  let m;
  const matches = [];
  while ((m = headerRe.exec(md)) !== null) {
    matches.push({ method: m[1], path: m[2].trim(), startIdx: m.index, headerLen: m[0].length });
  }
  for (let i = 0; i < matches.length; i++) {
    const cur = matches[i];
    const nextStart = i + 1 < matches.length ? matches[i + 1].startIdx : md.length;
    const body = md.slice(cur.startIdx + cur.headerLen, nextStart);

    const requiredFields = [];
    // Look for "required: name, email" or "* name (required)" patterns
    const reqListRe = /required[:\s]+([\w,\s]+)/gi;
    let rm;
    while ((rm = reqListRe.exec(body)) !== null) {
      const fields = rm[1].split(/[,\s]+/).map((s) => s.trim()).filter(Boolean).filter((s) => /^[a-z_]\w*$/i.test(s));
      for (const f of fields) if (!requiredFields.includes(f)) requiredFields.push(f);
    }
    // Look for "- name (string, required)" patterns
    const bulletRe = /[-*]\s+`?([a-z_]\w*)`?\s*\([^)]*required[^)]*\)/gi;
    while ((rm = bulletRe.exec(body)) !== null) {
      if (!requiredFields.includes(rm[1])) requiredFields.push(rm[1]);
    }

    // Enum fields: look for "status: open | closed | ..." or fields with listed values
    const enumFields = [];
    const enumRe = /\b(status|priority|type|kind|state|category|role)\b\s*[:：]\s*([\w\s|,\/]+)/gi;
    while ((rm = enumRe.exec(body)) !== null) {
      const field = rm[1].toLowerCase();
      const values = rm[2].split(/[|,/]/).map((s) => s.trim()).filter((s) => /^[a-z_-]+$/i.test(s));
      if (values.length >= 2 && values.length <= 10) {
        enumFields.push({ field, values });
      }
    }

    endpoints.push({
      method: cur.method,
      path: cur.path,
      requiredFields,
      enumFields,
    });
  }
  return endpoints;
}

function buildProbeUrl(port, rawPath) {
  const cleaned = rawPath
    .replace(/:([a-zA-Z_]\w*)/g, "nonexistent-00000000")
    .replace(/\{([a-zA-Z_]\w*)\}/g, "nonexistent-00000000");
  return `http://127.0.0.1:${port}${cleaned}`;
}

function startServer(serverDir, port) {
  const proc = spawn("node", ["server.js"], {
    cwd: serverDir,
    env: { ...process.env, PORT: String(port) },
    stdio: ["ignore", "pipe", "pipe"],
    detached: false,
  });
  proc.stdout.on("data", () => {});
  proc.stderr.on("data", () => {});
  proc.on("error", () => {});
  return proc;
}
