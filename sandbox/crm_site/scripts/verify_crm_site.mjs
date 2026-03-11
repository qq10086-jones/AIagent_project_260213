import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import vm from "vm";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const mode = String(process.argv[2] || "lint").trim().toLowerCase();

function readText(relPath) {
  return fs.readFileSync(path.join(repoRoot, relPath), "utf8");
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function checkJsSyntax(relPath) {
  const absPath = path.join(repoRoot, relPath);
  const code = fs.readFileSync(absPath, "utf8");
  if (typeof vm.SourceTextModule === "function") {
    new vm.SourceTextModule(code, { identifier: absPath });
    return;
  }
  new vm.Script(code, { filename: absPath });
}

function runLint() {
  checkJsSyntax("app.js");
  checkJsSyntax("server.js");
  const html = readText("index.html");
  const css = readText("styles.css");
  assert(html.includes("<script src=\"./app.js\"></script>"), "index.html must load app.js");
  assert(css.includes(":root"), "styles.css must define :root variables");
}

function runTypeCheck() {
  const appText = readText("app.js");
  const serverText = readText("server.js");
  assert(/module\.exports|export\s+/.test(appText), "app.js must expose module or export syntax");
  assert(/module\.exports|export\s+/.test(serverText), "server.js must expose module or export syntax");
}

function runTest() {
  const readme = readText("README.md");
  const html = readText("index.html");
  assert(readme.includes("CRM Pro Demo Site"), "README.md must describe the CRM sandbox");
  assert(/Nova CRM Pro/.test(html), "index.html must contain the Nova CRM Pro title");
}

function runBuild() {
  runLint();
  runTypeCheck();
  const html = readText("index.html");
  assert(html.includes("styles.css"), "index.html must reference styles.css");
  assert(html.includes("authView"), "index.html must include authView");
}

try {
  if (mode === "lint") runLint();
  else if (mode === "typecheck") runTypeCheck();
  else if (mode === "test") runTest();
  else if (mode === "build") runBuild();
  else {
    throw new Error(`unknown mode '${mode}'`);
  }
  console.log(`[verify_crm_site] ${mode}: pass`);
} catch (err) {
  console.error(`[verify_crm_site] ${mode}: fail`);
  console.error(err.message || String(err));
  process.exit(1);
}
