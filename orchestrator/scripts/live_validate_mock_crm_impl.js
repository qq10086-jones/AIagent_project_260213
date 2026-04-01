import fs from "fs";
import path from "path";

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeText(root, relPath, content) {
  const abs = path.join(root, ...relPath.split("/"));
  ensureDir(path.dirname(abs));
  fs.writeFileSync(abs, content, "utf8");
}

function writeJson(root, relPath, value) {
  writeText(root, relPath, `${JSON.stringify(value, null, 2)}\n`);
}

function parseArtifactRoot(prompt) {
  const match = String(prompt || "").match(/Absolute artifact output root:\s*([^\r\n]+)/i);
  return match ? String(match[1]).trim() : "";
}

function buildBackendSource() {
  return [
    "const express = require('express');",
    "const path = require('path');",
    "",
    "const app = express();",
    "app.use(express.json());",
    "app.use(express.static(path.join(__dirname, 'public')));",
    "",
    "const customers = [",
    "  { id: 'c1', name: 'Ada Lovelace', email: 'ada@example.com' },",
    "  { id: 'c2', name: 'Grace Hopper', email: 'grace@example.com' },",
    "];",
    "",
    "function findCustomer(id) {",
    "  return customers.find((customerRecord) => customerRecord.id === id);",
    "}",
    "",
    "app.get('/api/customers', (_req, res) => {",
    "  res.json({ customers });",
    "});",
    "",
    "app.get('/api/customers/:id', (req, res) => {",
    "  const customer = findCustomer(req.params.id);",
    "  if (!customer) return res.status(404).json({ error: 'not_found' });",
    "  return res.json(customer);",
    "});",
    "",
    "app.post('/api/customers', (req, res) => {",
    "  const next = {",
    "    id: `c${customers.length + 1}`,",
    "    name: String(req.body?.name || 'New Customer'),",
    "    email: String(req.body?.email || 'new@example.com'),",
    "  };",
    "  customers.push(next);",
    "  res.status(201).json(next);",
    "});",
    "",
    "app.put('/api/customers/:id', (req, res) => {",
    "  const customer = findCustomer(req.params.id);",
    "  if (!customer) return res.status(404).json({ error: 'not_found' });",
    "  customer.name = String(req.body?.name || customer.name);",
    "  customer.email = String(req.body?.email || customer.email);",
    "  return res.json(customer);",
    "});",
    "",
    "app.get('/', (_req, res) => {",
    "  res.sendFile(path.join(__dirname, 'public', 'index.html'));",
    "});",
    "",
    "const port = Number(process.env.PORT || 3000);",
    "app.listen(port, () => {",
    "  console.log(`crm-live-ready:${port}`);",
    "});",
    "",
  ].join("\n");
}

function buildFrontendSource() {
  return [
    "async function fetchJson(url, options) {",
    "  const response = await fetch(url, options);",
    "  return response.json();",
    "}",
    "",
    "function renderCustomers(customers) {",
    "  const list = document.getElementById('customer-list');",
    "  list.innerHTML = customers.map((customer) => `<li data-id=\"${customer.id}\">${customer.name} <span>${customer.email}</span></li>`).join('');",
    "}",
    "",
    "function renderDetail(customer) {",
    "  const detail = document.getElementById('customer-detail');",
    "  detail.textContent = `${customer.name} <${customer.email}>`;",
    "}",
    "",
    "async function boot() {",
    "  const payload = await fetchJson('/api/customers');",
    "  const customers = Array.isArray(payload.customers) ? payload.customers : [];",
    "  renderCustomers(customers);",
    "  if (customers[0]) renderDetail(customers[0]);",
    "}",
    "",
    "document.getElementById('customer-form').addEventListener('submit', async (event) => {",
    "  event.preventDefault();",
    "  const form = event.currentTarget;",
    "  const name = form.elements.name.value.trim();",
    "  const email = form.elements.email.value.trim();",
    "  const created = await fetchJson('/api/customers', {",
    "    method: 'POST',",
    "    headers: { 'Content-Type': 'application/json' },",
    "    body: JSON.stringify({ name, email }),",
    "  });",
    "  renderDetail(created);",
    "  const payload = await fetchJson('/api/customers');",
    "  renderCustomers(payload.customers || []);",
    "  form.reset();",
    "});",
    "",
    "boot();",
    "",
  ].join("\n");
}

function buildIndexHtml() {
  return [
    "<!doctype html>",
    "<html>",
    "<head><meta charset=\"utf-8\"><title>CRM Live Validation</title></head>",
    "<body>",
    "  <main>",
    "    <h1>CRM Live Validation</h1>",
    "    <ul id=\"customer-list\"></ul>",
    "    <section id=\"customer-detail\"></section>",
    "    <form id=\"customer-form\">",
    "      <input name=\"name\" placeholder=\"Name\" required />",
    "      <input name=\"email\" placeholder=\"Email\" required />",
    "      <button type=\"submit\">Save customer</button>",
    "    </form>",
    "  </main>",
    "  <script type=\"module\" src=\"./app.js\"></script>",
    "</body>",
    "</html>",
    "",
  ].join("\n");
}

function main() {
  const stepId = String(process.argv[2] || "").trim();
  const prompt = String(process.argv[3] || "");
  const artifactRoot = parseArtifactRoot(prompt);
  if (!stepId) throw new Error("step id required");
  if (!artifactRoot) throw new Error("artifact root not found in prompt");

  const cwd = process.cwd();
  const sandboxRoot = path.join(cwd, "sandbox", "crm_site");
  const artifactAbs = path.resolve(cwd, artifactRoot);
  ensureDir(sandboxRoot);
  ensureDir(artifactAbs);

  if (stepId === "impl_be") {
    const serverSource = buildBackendSource();
    writeText(cwd, "workspace/sandbox/crm_site/server.js", serverSource);
    writeText(artifactAbs, "impl/be_changes/server.js", serverSource);
    writeJson(artifactAbs, "impl/be_changes/package.json", {
      name: "crm-live-validation",
      version: "1.0.0",
      private: true,
      main: "server.js",
      dependencies: {
        express: "^4.21.2"
      }
    });
    writeText(artifactAbs, "impl/be_notes.md", [
      "# Backend Notes",
      "",
      "## API Contracts",
      "",
      "- GET /api/customers",
      "- GET /api/customers/:id",
      "- POST /api/customers",
      "- PUT /api/customers/:id",
      "",
      "## Shared Types",
      "",
      "- Customer: id, name, email",
      "",
      "## Scope Constraints",
      "",
      "- Minimal in-memory CRM for live validation.",
      "",
      "## Run Instructions",
      "",
      "1. cd impl/be_changes",
      "2. npm install",
      "3. node server.js",
      "",
    ].join("\n"));
    writeJson(artifactAbs, "handoff/be_to_fe.json", {
      from_step: "impl_be",
      to_step: "impl_fe",
      be_changes_path: "impl/be_changes",
      api_contracts: [
        { name: "List Customers", method: "GET", path: "/api/customers", response_shape: "{ customers: Customer[] }" },
        { name: "Get Customer", method: "GET", path: "/api/customers/:id", response_shape: "Customer" },
        { name: "Create Customer", method: "POST", path: "/api/customers", response_shape: "Customer" },
        { name: "Update Customer", method: "PUT", path: "/api/customers/:id", response_shape: "Customer" }
      ],
      shared_types: [
        { name: "Customer", description: "CRM customer record.", fields: ["id", "name", "email"] }
      ],
      scope_constraints: ["Only workspace/sandbox/crm_site/server.js is modified."]
    });
    console.log("live_validate_mock_crm_impl: backend artifacts written");
    return;
  }

  if (stepId === "impl_fe") {
    const appSource = buildFrontendSource();
    writeText(cwd, "workspace/sandbox/crm_site/app.js", appSource);
    writeText(artifactAbs, "impl/fe_changes/app.js", appSource);
    writeText(artifactAbs, "impl/fe_changes/public/index.html", buildIndexHtml());
    writeText(artifactAbs, "impl/fe_changes/public/app.js", appSource);
    writeText(artifactAbs, "impl/fe_notes.md", [
      "# Frontend Notes",
      "",
      "## UI Scope",
      "",
      "- Customer list",
      "- Customer detail",
      "- Customer add form",
      "",
      "## API Consumption",
      "",
      "- GET /api/customers",
      "- POST /api/customers",
      "",
      "## Run Instructions",
      "",
      "1. Open / after backend starts",
      "",
    ].join("\n"));
    writeJson(artifactAbs, "handoff/impl_to_qa.json", {
      from_steps: ["impl_be", "impl_fe"],
      to_step: "qa_verify",
      be_changes_path: "impl/be_changes",
      fe_changes_path: "impl/fe_changes",
      run_instructions: "Start backend and verify customer list, detail, and add flow.",
      known_limitations: ["Persistence is in-memory only"],
      api_contracts_path: "handoff/be_to_fe.json"
    });
    console.log("live_validate_mock_crm_impl: frontend artifacts written");
    return;
  }

  throw new Error(`unsupported step id: ${stepId}`);
}

main();
