import fs from "fs";
import path from "path";
import { spawn, spawnSync } from "child_process";
import { fileURLToPath } from "url";
import vm from "vm";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const mode = String(process.argv[2] || "lint").trim().toLowerCase();
const smokeDir = path.join(repoRoot, "smoke");
const smokeResultPath = path.join(smokeDir, "smoke_result.json");

function readText(relPath) {
  return fs.readFileSync(path.join(repoRoot, relPath), "utf8");
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeSmokeResult(result) {
  ensureDir(smokeDir);
  fs.writeFileSync(smokeResultPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
}

function checkJsSyntax(relPath) {
  const absPath = path.join(repoRoot, relPath);
  const isModuleFile = /\.m?js$/i.test(absPath);
  if (isModuleFile) {
    const result = spawnSync(process.execPath, ["--check", absPath], { encoding: "utf8" });
    if (result.status !== 0) {
      throw new Error((result.stderr || result.stdout || `syntax check failed for ${relPath}`).trim());
    }
    return;
  }
  const code = fs.readFileSync(absPath, "utf8");
  new vm.Script(code, { filename: absPath });
}

function runLint() {
  checkJsSyntax("app.js");
  checkJsSyntax("server.js");
  checkJsSyntax("impl/be_changes/server.js");
  const html = readText("index.html");
  const css = readText("styles.css");
  const implHtml = readText("impl/be_changes/public/index.html");
  assert(html.includes("<script type=\"module\" src=\"./app.js\"></script>"), "index.html must load app.js as module");
  assert(css.includes(":root"), "styles.css must define :root variables");
  assert(/Customer CRM/.test(implHtml), "impl/be_changes/public/index.html must render the CRM app shell");
}

function runTypeCheck() {
  const appText = readText("app.js");
  const serverText = readText("server.js");
  const implServerText = readText("impl/be_changes/server.js");
  assert(/document\.getElementById|fetchJson|boot\(/.test(appText), "app.js must expose the customer page boot logic");
  assert(/module\.exports|app\.listen|const app = express/.test(serverText), "server.js must define an express app");
  assert(/app\.post\('\/api\/auth\/login'/.test(implServerText), "impl/be_changes/server.js must expose login route");
  assert(/app\.get\('\/api\/customers'/.test(implServerText), "impl/be_changes/server.js must expose list customers route");
}

async function waitForServer(url, timeoutMs) {
  const started = Date.now();
  let lastError = "server did not become ready";
  while (Date.now() - started < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
      lastError = `health endpoint returned ${res.status}`;
    } catch (err) {
      lastError = err.message || String(err);
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(lastError);
}

function parseSetCookie(headers) {
  const setCookie = headers.get("set-cookie");
  if (!setCookie) throw new Error("missing session cookie");
  return setCookie.split(",")[0].split(";")[0];
}

async function fetchJson(url, { method = "GET", headers = {}, body } = {}) {
  const res = await fetch(url, {
    method,
    headers,
    body,
  });
  const text = await res.text();
  let json = null;
  if (text) {
    try {
      json = JSON.parse(text);
    } catch (err) {
      throw new Error(`invalid JSON from ${url}: ${err.message}`);
    }
  }
  return { res, json };
}

async function runRuntimeSmoke() {
  const port = 3310;
  const baseUrl = `http://127.0.0.1:${port}`;
  const serverPath = path.join(repoRoot, "impl", "be_changes", "server.js");
  const child = spawn(process.execPath, [serverPath], {
    cwd: path.dirname(serverPath),
    env: { ...process.env, PORT: String(port) },
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += String(chunk);
  });
  child.stderr.on("data", (chunk) => {
    stderr += String(chunk);
  });

  const result = {
    install_ok: true,
    server_started: false,
    root_check: { status: 0, content_type: "", passed: false },
    api_check: { endpoint: "/api/customers", status: 0, response_sample: "", passed: false, skipped: false },
    auth_check: { endpoint: "/api/auth/login", status: 0, passed: false },
    create_check: { endpoint: "/api/customers", status: 0, passed: false },
    errors: [],
    verdict: "fail",
    evidence_level: "l1_l2",
  };

  try {
    await waitForServer(`${baseUrl}/`, 10000);
    result.server_started = true;

    const rootRes = await fetch(`${baseUrl}/`);
    const rootHtml = await rootRes.text();
    result.root_check = {
      status: rootRes.status,
      content_type: rootRes.headers.get("content-type") || "",
      passed: rootRes.status === 200 && /Customer CRM/.test(rootHtml),
    };
    if (!result.root_check.passed) {
      result.errors.push("root page failed readiness check");
    }

    const loginResp = await fetchJson(`${baseUrl}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: "admin", password: "admin123" }),
    });
    const sessionCookie = parseSetCookie(loginResp.res.headers);
    result.auth_check = {
      endpoint: "/api/auth/login",
      status: loginResp.res.status,
      passed: loginResp.res.status === 200 && loginResp.json?.success === true,
    };
    if (!result.auth_check.passed) {
      result.errors.push("login failed");
    }

    const csrfResp = await fetchJson(`${baseUrl}/api/csrf-token`, {
      headers: { Cookie: sessionCookie },
    });
    const csrfToken = csrfResp.json?.csrfToken || "";
    if (!csrfToken) {
      result.errors.push("csrf token fetch failed");
    }

    const listResp = await fetchJson(`${baseUrl}/api/customers?limit=100`, {
      headers: { Cookie: sessionCookie },
    });
    result.api_check = {
      endpoint: "/api/customers",
      status: listResp.res.status,
      response_sample: JSON.stringify(listResp.json).slice(0, 200),
      passed: listResp.res.status === 200 && Array.isArray(listResp.json?.data),
      skipped: false,
    };
    if (!result.api_check.passed) {
      result.errors.push("customer list API failed");
    }

    const createResp = await fetchJson(`${baseUrl}/api/customers`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Cookie: sessionCookie,
        "CSRF-Token": csrfToken,
      },
      body: JSON.stringify({
        name: "Smoke Test User",
        email: "smoke@example.com",
        phone: "10086",
        company: "Nexus QA",
        notes: "created by verify_crm_site runtime smoke",
      }),
    });
    result.create_check = {
      endpoint: "/api/customers",
      status: createResp.res.status,
      passed: createResp.res.status === 201 && createResp.json?.success === true,
    };
    if (!result.create_check.passed) {
      result.errors.push("customer create API failed");
    }

    result.verdict = result.errors.length === 0 ? "pass" : "fail";
    writeSmokeResult(result);
    if (result.errors.length > 0) {
      throw new Error(result.errors.join("; "));
    }
  } finally {
    child.kill("SIGTERM");
    await new Promise((resolve) => {
      child.once("exit", () => resolve());
      setTimeout(() => {
        if (!child.killed) child.kill("SIGKILL");
        resolve();
      }, 1500);
    });
    if (stderr.trim() && !result.server_started) {
      result.errors.push(`server stderr: ${stderr.trim()}`);
      writeSmokeResult(result);
    }
  }
}

async function runTest() {
  const readme = readText("README.md");
  const html = readText("index.html");
  assert(readme.includes("Document Release Hub"), "README.md must describe the document release hub");
  assert(/Document Release Hub/.test(html), "index.html must contain the Document Release Hub title");
  assert(/Discord Intake Inbox/.test(html), "index.html must include Discord Intake Inbox");
  await runRuntimeSmoke();
}

async function runBuild() {
  runLint();
  runTypeCheck();
  const html = readText("index.html");
  assert(html.includes("styles.css"), "index.html must reference styles.css");
  assert(html.includes("generatorForm"), "index.html must include generatorForm");
}

async function main() {
  if (mode === "lint") runLint();
  else if (mode === "typecheck") runTypeCheck();
  else if (mode === "test") await runTest();
  else if (mode === "build") await runBuild();
  else throw new Error(`unknown mode '${mode}'`);
  console.log(`[verify_crm_site] ${mode}: pass`);
}

main().catch((err) => {
  console.error(`[verify_crm_site] ${mode}: fail`);
  console.error(err.message || String(err));
  process.exit(1);
});
