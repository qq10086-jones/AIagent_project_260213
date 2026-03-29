import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import http from "http";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key.startsWith("--")) continue;
    out[key.slice(2)] = value;
    i += 1;
  }
  return out;
}

function ensureDir(absPath) {
  fs.mkdirSync(absPath, { recursive: true });
}

function readJson(absPath) {
  try {
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

function copyDirRecursive(src, dest) {
  if (!fs.existsSync(src) || !fs.statSync(src).isDirectory()) return false;
  ensureDir(dest);
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else if (entry.isFile()) {
      ensureDir(path.dirname(destPath));
      fs.copyFileSync(srcPath, destPath);
    }
  }
  return true;
}

function runProcess(command, args, options = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"], ...options });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk || "");
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk || "");
    });
    child.on("close", (code) => resolve({ code: Number(code || 0), stdout, stderr }));
    child.on("error", (error) => resolve({ code: 1, stdout, stderr: String(error?.message || error) }));
  });
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function httpRequest({ port, pathname }) {
  return new Promise((resolve) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port,
        path: pathname,
        method: "GET",
      },
      (res) => {
        let body = "";
        res.on("data", (chunk) => {
          body += String(chunk || "");
        });
        res.on("end", () => {
          resolve({
            status: Number(res.statusCode || 0),
            headers: res.headers || {},
            body,
          });
        });
      }
    );
    req.on("error", (error) => {
      resolve({
        status: 0,
        headers: {},
        body: "",
        error: String(error?.message || error),
      });
    });
    req.end();
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const artifactRoot = String(args["artifact-root"] || "").trim();
  const port = Number(args.port || 13099);
  const apiEndpoint = String(args["api-endpoint"] || "").trim();
  const rootAbs = path.resolve(process.cwd(), artifactRoot);
  const smokeDir = path.join(rootAbs, "smoke");
  const smokePath = path.join(smokeDir, "smoke_result.json");
  ensureDir(smokeDir);

  const result = {
    install_ok: false,
    server_started: false,
    root_check: {
      status: 0,
      content_type: "",
      passed: false
    },
    api_check: {
      endpoint: apiEndpoint || "",
      status: 0,
      response_sample: "",
      passed: false,
      skipped: !apiEndpoint
    },
    errors: [],
    verdict: "fail",
    evidence_level: "l1_only"
  };

  const beDir = path.join(rootAbs, "impl", "be_changes");
  const fePublicDir = path.join(rootAbs, "impl", "fe_changes", "public");
  const bePublicDir = path.join(beDir, "public");

  try {
    copyDirRecursive(fePublicDir, bePublicDir);

    const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";
    const install = await runProcess(npmCmd, ["install", "--silent"], { cwd: beDir, env: process.env });
    result.install_ok = install.code === 0;
    if (install.code !== 0) {
      result.errors.push(`npm install failed: ${String(install.stderr || install.stdout || "").trim().slice(0, 500)}`);
    }

    const server = spawn(process.execPath, ["server.js"], {
      cwd: beDir,
      env: {
        ...process.env,
        PORT: String(port),
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let serverStdout = "";
    let serverStderr = "";
    let exited = false;
    server.stdout.on("data", (chunk) => {
      serverStdout += String(chunk || "");
    });
    server.stderr.on("data", (chunk) => {
      serverStderr += String(chunk || "");
    });
    server.on("close", () => {
      exited = true;
    });
    server.on("error", (error) => {
      exited = true;
      result.errors.push(`server start error: ${String(error?.message || error)}`);
    });

    await wait(3000);
    result.server_started = !exited;
    if (!result.server_started) {
      result.errors.push(`server exited early: ${String(serverStderr || serverStdout).trim().slice(0, 500)}`);
    }

    const rootResponse = await httpRequest({ port, pathname: "/" });
    const rootType = String(rootResponse.headers?.["content-type"] || "");
    result.root_check.status = rootResponse.status;
    result.root_check.content_type = rootType;
    result.root_check.passed =
      rootResponse.status >= 200 &&
      rootResponse.status < 400 &&
      (rootType.includes("text/html") || /<html/i.test(rootResponse.body || ""));
    if (!result.root_check.passed) {
      result.errors.push(`root check failed: status=${rootResponse.status}`);
    }

    if (apiEndpoint) {
      const apiResponse = await httpRequest({ port, pathname: apiEndpoint });
      result.api_check.endpoint = apiEndpoint;
      result.api_check.status = apiResponse.status;
      result.api_check.response_sample = String(apiResponse.body || "").slice(0, 240);
      result.api_check.passed = apiResponse.status >= 200 && apiResponse.status < 400;
      result.api_check.skipped = false;
      if (!result.api_check.passed) {
        result.errors.push(`api check failed: ${apiEndpoint} status=${apiResponse.status}`);
      }
    }

    try {
      server.kill("SIGTERM");
    } catch {
      // ignore cleanup error
    }

    result.evidence_level = result.api_check.skipped ? "l1_only" : "l1_l2";
    result.verdict = result.root_check.passed && (result.api_check.skipped || result.api_check.passed) ? "pass" : "fail";
  } catch (error) {
    result.errors.push(String(error?.message || error));
  }

  fs.writeFileSync(smokePath, JSON.stringify(result, null, 2));
  process.exit(0);
}

main().catch((error) => {
  const fallback = {
    install_ok: false,
    server_started: false,
    root_check: { status: 0, content_type: "", passed: false },
    api_check: { endpoint: "", status: 0, response_sample: "", passed: false, skipped: true },
    errors: [String(error?.message || error)],
    verdict: "fail",
    evidence_level: "l1_only"
  };
  const args = parseArgs(process.argv.slice(2));
  const artifactRoot = String(args["artifact-root"] || "").trim();
  const rootAbs = path.resolve(process.cwd(), artifactRoot);
  const smokeDir = path.join(rootAbs, "smoke");
  ensureDir(smokeDir);
  fs.writeFileSync(path.join(smokeDir, "smoke_result.json"), JSON.stringify(fallback, null, 2));
  process.exit(0);
});
