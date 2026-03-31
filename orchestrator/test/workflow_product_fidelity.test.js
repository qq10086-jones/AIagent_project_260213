import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import {
  buildGoNoGoResult,
  buildProductFidelityReport,
  buildPreviewValidationReport,
} from "../src/domain/workflow_artifact_audit.js";
import { buildFinalResultPackage } from "../src/final_result_packager.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-product-fidelity-"));
}

function writeText(targetPath, value) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, value, "utf8");
}

function writeJson(targetPath, value) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, JSON.stringify(value, null, 2), "utf8");
}

test("buildProductFidelityReport flags shallow scaffold artifacts with warning", () => {
  const releaseRoot = makeWorkspace();
  writeText(path.join(releaseRoot, "impl", "fe_changes", "app.js"), [
    "export function renderItemList(items) {",
    '  return items.map((item) => item.name).join(", ");',
    "}",
    "",
    "export function submitItemForm(payload) {",
    '  return { method: "POST", path: "/api/items", body: payload };',
    "}",
    "",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "be_changes", "server.js"), [
    "export function listItemsHandler() {",
    '  return [{ id: "item-001", name: "Sample Item" }];',
    "}",
    "",
    "export function createItemHandler(input) {",
    '  return { id: "item-new", ...input };',
    "}",
    "",
  ].join("\n"));
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass_with_warnings",
    checks: [
      {
        check_id: "qa-1",
        layer: "semantic",
        description: "Acceptance A1 coverage review",
        status: "warning",
        detail: "Auto-generated QA scaffold pending human review for A1.",
      },
    ],
    verified_artifacts: ["A1"],
    source: "worker-coder artifact scaffold",
  });
  writeJson(path.join(releaseRoot, "preview", "preview_runtime.json"), {
    project_root: "/workspace/sandbox/crm_site",
  });

  const result = buildProductFidelityReport({
    run: {
      workflow_run_id: "wf-1",
      run_id: "run-1",
      workflow_id: "coding_team_v0",
      project_type: "generic_app",
    },
    releaseRoot,
  });

  assert.equal(result.classification, "preview_mismatch");
  assert.equal(result.should_warn, true);
  assert.equal(result.perceptual_quality.score, "low");
  assert.ok(result.reasoning.some((item) => item.criterion === "placeholder_free" && item.pass === false));
});

test("buildPreviewValidationReport flags shared sandbox mismatch for non-CRM project types", () => {
  const releaseRoot = makeWorkspace();
  writeJson(path.join(releaseRoot, "preview", "preview_runtime.json"), {
    project_root: "/workspace/sandbox/crm_site",
    preview_url: "https://preview.example.com/run-1",
  });

  const result = buildPreviewValidationReport({
    run: {
      workflow_run_id: "wf-preview-1",
      run_id: "run-preview-1",
      workflow_id: "coding_team_v0",
      project_type: "generic_app",
    },
    releaseRoot,
  });

  assert.equal(result.classification, "preview_mismatch");
  assert.equal(result.preview_source, "shared_crm_sandbox");
  assert.equal(result.should_warn, true);
});

test("buildGoNoGoResult carries product fidelity warning without changing verdict", () => {
  const result = buildGoNoGoResult({
    run: {
      workflow_run_id: "wf-2",
      run_id: "run-2",
      workflow_id: "coding_team_v0",
      project_type: "generic_app",
    },
    manifest: { status: "succeeded" },
    steps: [{ step_id: "qa_verify", gate_name: "acceptance", status: "succeeded" }],
    validator: { ok: true, reasons: [] },
    canaryReport: { verdict: "pass", totals: { missing_artifacts_total: 0 } },
    productFidelityReport: {
      classification: "artifact_complete_but_shallow",
      warning: "product_fidelity_warning:artifact_complete_but_shallow",
      should_warn: true,
    },
    productFidelityReportRelPath: "artifacts/release/run-2/qa/product_fidelity_report.json",
    previewValidationReport: {
      classification: "preview_mismatch",
      warning: "preview_validation_warning:preview_mismatch",
      should_warn: true,
    },
    previewValidationReportRelPath: "artifacts/release/run-2/qa/preview_validation_report.json",
    expectedSteps: 1,
    strict: true,
  });

  assert.equal(result.verdict, "GO");
  assert.deepEqual(result.product_fidelity_warning, {
    classification: "artifact_complete_but_shallow",
    warning: "product_fidelity_warning:artifact_complete_but_shallow",
    report_path: "artifacts/release/run-2/qa/product_fidelity_report.json",
  });
  assert.deepEqual(result.preview_validation_warning, {
    classification: "preview_mismatch",
    warning: "preview_validation_warning:preview_mismatch",
    report_path: "artifacts/release/run-2/qa/preview_validation_report.json",
  });
});

test("buildProductFidelityReport classifies visually_incomplete when functional checks pass but perceptual quality is low", () => {
  const releaseRoot = makeWorkspace();
  // FE passes depth (>= 8 lines, >= 180 bytes) but has NO UI signals — perceptual score stays low
  writeText(path.join(releaseRoot, "impl", "fe_changes", "app.js"), [
    "export function formatPrice(amount) {",
    "  return `$${Number(amount).toFixed(2)}`;",
    "}",
    "export function sortByPrice(products) {",
    "  return products.slice().sort((a, b) => a.price - b.price);",
    "}",
    "export function filterByCategory(products, category) {",
    "  return products.filter(p => p.category === category);",
    "}",
    "export function computeCartTotal(entries) {",
    "  return entries.reduce((sum, entry) => sum + entry.price * entry.quantity, 0);",
    "}",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "be_changes", "server.js"), [
    "export function listProductsHandler() {",
    "  return [{ id: 'p1', name: 'Widget', price: 9.99 }];",
    "}",
    "export function addCartHandler(input) {",
    "  return { cart_id: 'cart-1', items: [input] };",
    "}",
    "export function getCartHandler() {",
    "  return { cart_id: 'cart-1', items: [], total: 0 };",
    "}",
  ].join("\n"));
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [{ check_id: "qa-1", layer: "semantic", description: "Product journey", status: "pass", detail: "products list and cart endpoints verified" }],
    verified_artifacts: ["A1"],
  });
  // No preview file — preview check passes vacuously

  const result = buildProductFidelityReport({
    run: {
      workflow_run_id: "wf-vi-1",
      run_id: "run-vi-1",
      workflow_id: "coding_team_v0",
      project_type: "generic_app",
    },
    releaseRoot,
  });

  assert.equal(result.classification, "visually_incomplete");
  assert.equal(result.should_warn, true);
  assert.equal(result.perceptual_quality.score, "low");
  const pqCheck = result.reasoning.find((r) => r.criterion === "perceptual_quality_minimum");
  assert.ok(pqCheck, "perceptual_quality_minimum criterion should be present");
  assert.equal(pqCheck.pass, false);
});

test("buildProductFidelityReport classifies demo_usable when perceptual quality is mid or high", () => {
  const releaseRoot = makeWorkspace();
  // Rich FE with UI signals and many lines — perceptual score will be high
  writeText(path.join(releaseRoot, "impl", "fe_changes", "app.js"), [
    "import React, { useState } from 'react';",
    "export function ProductGrid({ products }) {",
    "  return (",
    "    <div className='grid'>",
    "      {products.map(p => (",
    "        <div key={p.id} className='card'>",
    "          <h3>{p.name}</h3>",
    "          <span>{p.price}</span>",
    "          <button onClick={() => addToCart(p.id)}>Add to Cart</button>",
    "        </div>",
    "      ))}",
    "    </div>",
    "  );",
    "}",
    "export function CartPanel({ cart }) {",
    "  return <div className='cart'>{cart.items.length} items</div>;",
    "}",
    "export function addToCart(id) {",
    "  return fetch('/api/cart', { method: 'POST', body: JSON.stringify({ id }) });",
    "}",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "be_changes", "server.js"), [
    "export function listProductsHandler() {",
    "  return [{ id: 'p1', name: 'Widget', price: 9.99 }];",
    "}",
    "export function addCartHandler(input) {",
    "  return { cart_id: 'cart-1', items: [input] };",
    "}",
    "export function getCartHandler() {",
    "  return { cart_id: 'cart-1', items: [], total: 0 };",
    "}",
  ].join("\n"));
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [{ check_id: "qa-1", layer: "semantic", description: "Product grid journey", status: "pass", detail: "product grid and cart verified" }],
    verified_artifacts: ["A1"],
  });

  const result = buildProductFidelityReport({
    run: {
      workflow_run_id: "wf-du-1",
      run_id: "run-du-1",
      workflow_id: "coding_team_v0",
      project_type: "generic_app",
    },
    releaseRoot,
  });

  assert.ok(["demo_usable", "domain_misaligned"].includes(result.classification),
    `expected demo_usable or domain_misaligned, got ${result.classification}`);
  assert.ok(["mid", "high"].includes(result.perceptual_quality.score),
    `expected mid or high perceptual quality, got ${result.perceptual_quality.score}`);
  const pqCheck = result.reasoning.find((r) => r.criterion === "perceptual_quality_minimum");
  assert.equal(pqCheck.pass, true);
});

test("buildProductFidelityReport uses static HTML surface for single_file_html perceptual quality", () => {
  const releaseRoot = makeWorkspace();
  writeText(path.join(releaseRoot, "impl", "fe_changes", "app.js"), [
    "document.addEventListener('DOMContentLoaded', () => {",
    "  const cta = document.getElementById('cta-button');",
    "  if (cta) cta.addEventListener('click', () => alert('Hello'));",
    "});",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "fe_changes", "public", "app.js"), [
    "const faqButtons = document.querySelectorAll('.faq-question');",
    "faqButtons.forEach((button) => {",
    "  button.addEventListener('click', () => button.classList.toggle('open'));",
    "});",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "fe_changes", "public", "index.html"), [
    "<!DOCTYPE html>",
    "<html>",
    "<body>",
    "  <main>",
    "    <section class='hero'>",
    "      <h1>Discover Your Next Great Read</h1>",
    "      <button id='cta-button'>Browse Collection</button>",
    "    </section>",
    "    <section class='faq'>",
    "      <button class='faq-question'>What is this?</button>",
    "    </section>",
    "  </main>",
    "</body>",
    "</html>",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "be_changes", "server.js"), [
    "export function startServer() {",
    "  return { ok: true, port: 3000 };",
    "}",
    "export function healthCheck() {",
    "  return { status: 'ok' };",
    "}",
    "export function serveIndex() {",
    "  return '<html></html>';",
    "}",
  ].join("\n"));
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [{ check_id: "qa-1", layer: "semantic", description: "Landing page journey", status: "pass", detail: "hero, CTA, and FAQ interaction verified" }],
    verified_artifacts: ["A1"],
  });

  const result = buildProductFidelityReport({
    run: {
      workflow_run_id: "wf-html-1",
      run_id: "run-html-1",
      workflow_id: "coding_team_v0",
      project_type: "single_file_html",
    },
    releaseRoot,
  });

  assert.equal(result.classification, "demo_usable");
  assert.ok(["mid", "high"].includes(result.perceptual_quality.score));
  assert.equal(result.perceptual_quality.interactive_affordances_visible, true);
  assert.match(result.reasoning.find((item) => item.criterion === "frontend_depth")?.evidence || "", /public\/index\.html/);
});

test("buildProductFidelityReport applies domain pack checks for webapp_crm project type", () => {
  const releaseRoot = makeWorkspace();
  writeText(path.join(releaseRoot, "impl", "fe_changes", "app.js"), [
    "export function renderCustomerList(customers) {",
    "  return customers.map((c) => `<tr>${c.name} ${c.email}</tr>`).join('');",
    "}",
    "export function openCustomerDetail(id) {",
    "  return fetch(`/api/customers/${id}`).then((r) => r.json());",
    "}",
    "export function createCustomerForm(payload) {",
    "  return fetch('/api/customers', { method: 'POST', body: JSON.stringify(payload) });",
    "}",
    "export function renderContactList(contacts) {",
    "  return contacts.map((c) => `<tr>${c.name}</tr>`).join('');",
    "}",
    "",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "be_changes", "server.js"), [
    "export function listCustomersHandler() {",
    "  return [{ id: 'cust-001', name: 'Acme Corp', email: 'acme@example.com', status: 'active' }];",
    "}",
    "export function createCustomerHandler(input) {",
    "  return { id: 'cust-new', ...input };",
    "}",
    "",
  ].join("\n"));
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [{ check_id: "qa-1", layer: "semantic", description: "Customer list journey", status: "pass", detail: "customer list renders correctly" }],
    verified_artifacts: ["A1"],
  });
  writeJson(path.join(releaseRoot, "preview", "preview_runtime.json"), {
    project_root: "/workspace/sandbox/crm_site",
  });

  const result = buildProductFidelityReport({
    run: {
      workflow_run_id: "wf-crm-1",
      run_id: "run-crm-1",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
    },
    releaseRoot,
  });

  assert.ok(result.domain_pack !== null, "domain_pack should be set for webapp_crm");
  assert.equal(result.domain_pack.domain, "crm");
  const domainNounCheck = result.reasoning.find((r) => r.criterion === "domain_noun_present");
  assert.ok(domainNounCheck, "domain_noun_present criterion should be present");
  assert.equal(domainNounCheck.pass, true, "domain nouns should be found in crm impl");
});

test("buildProductFidelityReport prefers published frontend surface when public assets exist", () => {
  const releaseRoot = makeWorkspace();
  writeText(path.join(releaseRoot, "impl", "fe_changes", "app.js"), [
    "export function showReservationPlaceholder() {",
    "  return 'placeholder';",
    "}",
    "export function renderMenuItem(item) {",
    "  return item.name;",
    "}",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "fe_changes", "public", "index.html"), [
    "<!doctype html>",
    "<html>",
    "<body>",
    "  <main>",
    "    <h1>Customer Workspace</h1>",
    "    <button id='addCustomerBtn'>Add Customer</button>",
    "    <form id='customerForm'></form>",
    "  </main>",
    "  <script src='app.js'></script>",
    "</body>",
    "</html>",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "fe_changes", "public", "app.js"), [
    "async function loadCustomers() {",
    "  const res = await fetch('/api/customers');",
    "  return res.json();",
    "}",
    "document.getElementById('addCustomerBtn')?.addEventListener('click', loadCustomers);",
  ].join("\n"));
  writeText(path.join(releaseRoot, "impl", "be_changes", "server.js"), [
    "export function listCustomersHandler() {",
    "  return [{ id: 'cust-001', name: 'Acme Corp' }];",
    "}",
    "export function createCustomerHandler(input) {",
    "  return { id: 'cust-new', ...input };",
    "}",
  ].join("\n"));
  writeJson(path.join(releaseRoot, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [{ check_id: "qa-1", layer: "semantic", description: "Customer journey", status: "pass", detail: "customer list and add flow verified" }],
    verified_artifacts: ["A1"],
  });
  writeJson(path.join(releaseRoot, "preview", "preview_runtime.json"), {
    project_root: "/workspace/sandbox/crm_site",
  });

  const result = buildProductFidelityReport({
    run: {
      workflow_run_id: "wf-crm-public-1",
      run_id: "run-crm-public-1",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
    },
    releaseRoot,
  });

  assert.equal(result.reasoning.find((item) => item.criterion === "placeholder_free")?.pass, true);
  assert.equal(result.reasoning.find((item) => item.criterion === "domain_not_generic_crud")?.pass, true);
  assert.ok(["mid", "high"].includes(result.perceptual_quality.score));
});

test("buildGoNoGoResult blocks GO when fidelityGateMode=blocking and fidelity warns", () => {
  const result = buildGoNoGoResult({
    run: {
      workflow_run_id: "wf-block-1",
      run_id: "run-block-1",
      workflow_id: "coding_team_v0",
      project_type: "generic_app",
    },
    manifest: { status: "succeeded" },
    steps: [{ step_id: "qa_verify", gate_name: "acceptance", status: "succeeded" }],
    validator: { ok: true, reasons: [] },
    canaryReport: { verdict: "pass", totals: { missing_artifacts_total: 0 } },
    productFidelityReport: {
      classification: "artifact_complete_but_shallow",
      warning: "product_fidelity_warning:artifact_complete_but_shallow",
      should_warn: true,
    },
    productFidelityReportRelPath: "artifacts/release/run-block-1/qa/product_fidelity_report.json",
    previewValidationReport: null,
    previewValidationReportRelPath: "",
    expectedSteps: 1,
    strict: true,
    fidelityGateMode: "blocking",
  });

  assert.equal(result.verdict, "NO_GO");
  assert.ok(result.reasons.some((r) => r.startsWith("product_fidelity_gate:")));
});

test("buildGoNoGoResult does not block GO when fidelityGateMode=warning (default)", () => {
  const result = buildGoNoGoResult({
    run: {
      workflow_run_id: "wf-warn-1",
      run_id: "run-warn-1",
      workflow_id: "coding_team_v0",
      project_type: "generic_app",
    },
    manifest: { status: "succeeded" },
    steps: [{ step_id: "qa_verify", gate_name: "acceptance", status: "succeeded" }],
    validator: { ok: true, reasons: [] },
    canaryReport: { verdict: "pass", totals: { missing_artifacts_total: 0 } },
    productFidelityReport: {
      classification: "artifact_complete_but_shallow",
      warning: "product_fidelity_warning:artifact_complete_but_shallow",
      should_warn: true,
    },
    productFidelityReportRelPath: "artifacts/release/run-warn-1/qa/product_fidelity_report.json",
    previewValidationReport: null,
    previewValidationReportRelPath: "",
    expectedSteps: 1,
    strict: true,
    fidelityGateMode: "warning",
  });

  assert.equal(result.verdict, "GO");
  assert.ok(result.product_fidelity_warning !== null);
});

test("buildFinalResultPackage includes preview and fidelity artifacts when present", () => {
  const pkg = buildFinalResultPackage({
    workflowRunId: "wf-3",
    runId: "run-3",
    status: "succeeded",
    summaryPath: "/tmp/run_summary.md",
    manifestPath: "/tmp/run_manifest.json",
    goNoGoResultPath: "/tmp/go_no_go_result.json",
    strictCanaryReportPath: "/tmp/strict_canary_report.md",
    strictCanaryJsonPath: "/tmp/strict_canary_report.json",
    previewValidationReportPath: "/tmp/preview_validation_report.json",
    productFidelityReportPath: "/tmp/product_fidelity_report.json",
    goNoGoVerdict: "GO",
    strictCanaryVerdict: "pass",
  });

  assert.ok(pkg.artifacts.some((item) => item.label === "preview_validation_report" && item.path === "/tmp/preview_validation_report.json"));
  assert.ok(pkg.artifacts.some((item) => item.label === "product_fidelity_report" && item.path === "/tmp/product_fidelity_report.json"));
});
