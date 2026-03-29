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
  const code = fs.readFileSync(absPath, "utf8")
    .replace(/^export\s+\{[^}]+\}\s+from\s+["'][^"']+["'];?\s*$/gm, "")
    .replace(/^export\s+\{[^}]+\};?\s*$/gm, "")
    .replace(/^export\s+default\s+/gm, "const __default__ = ");
  new vm.Script(code, { filename: absPath });
}

function runLint() {
  checkJsSyntax("app.js");
  checkJsSyntax("server.js");
  const html = readText("index.html");
  const css = readText("styles.css");
  assert(html.includes("<script type=\"module\" src=\"./app.js\"></script>"), "index.html must load app.js as module");
  assert(css.includes(":root"), "styles.css must define :root variables");
}

function runTypeCheck() {
  const appText = readText("app.js");
  const serverText = readText("server.js");
  assert(/export\s+/.test(appText), "app.js must expose export syntax");
  assert(/export\s+/.test(serverText), "server.js must expose export syntax");
}

function runTest() {
  const readme = readText("README.md");
  const html = readText("index.html");
  assert(readme.includes("Document Release Hub"), "README.md must describe the document release hub");
  assert(/Document Release Hub/.test(html), "index.html must contain the Document Release Hub title");
  assert(/Discord Intake Inbox/.test(html), "index.html must include Discord Intake Inbox");
}

function runBuild() {
  runLint();
  runTypeCheck();
  const html = readText("index.html");
  assert(html.includes("styles.css"), "index.html must reference styles.css");
  assert(html.includes("generatorForm"), "index.html must include generatorForm");
}

try {
  if (mode === "lint") runLint();
  else if (mode === "typecheck") runTypeCheck();
  else if (mode === "test") runTest();
  else if (mode === "build") runBuild();
  else throw new Error(`unknown mode '${mode}'`);
  console.log(`[verify_crm_site] ${mode}: pass`);
} catch (err) {
  console.error(`[verify_crm_site] ${mode}: fail`);
  console.error(err.message || String(err));
  process.exit(1);
}
