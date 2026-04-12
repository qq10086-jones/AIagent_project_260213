/**
 * workflow_state.js
 *
 * Step status normalisation, step contract definitions, and prompt-building
 * helpers.  Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import path from "path";

export function normalizeStepStatus(status) {
  const s = String(status || "").toLowerCase();
  if (["pending", "queued", "waiting_approval", "running", "succeeded", "failed", "partial_failure"].includes(s)) return s;
  return "pending";
}

export const STEP_CONTRACTS = {
  pm_spec: {
    title: "PM Specification",
    required_artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    instructions: [
      "Define user stories, scope boundaries, and non-goals for the actual system described in the Goal field above.",
      "FEATURE ENUMERATION (mandatory): Before writing spec.md, identify EVERY distinct feature or module in the Goal (e.g. 'customer management', 'complaint tickets', 'file upload', 'dashboard'). Each feature MUST get its own ## section in Scope, at least 2 dedicated User Stories (## User Stories section, formatted as '- **US-N:** As a user, I want to...'), and at least 1 acceptance criterion. If the Goal mentions N features, produce at least N*2 user stories. Omitting ANY feature from the Goal is a validation failure.",
      "Non-Goals MUST ONLY list items NOT mentioned in the Goal. NEVER move a Goal feature into Non-Goals. If the Goal says 'dashboard analytics', then analytics is a GOAL. Putting a Goal feature in Non-Goals will fail validation.",
      "CRITICAL — plan/acceptance.json: Extract EVERY numbered requirement and feature from the Goal into a SEPARATE acceptance criterion. Do NOT summarize or generalize. Each criterion object must have: id (AC-1, AC-2, ...), description (the specific feature), verify_command (a concrete shell command like 'grep', 'curl', or 'node -e' that can deterministically check), expected_pattern (regex to match output), and verify_tier: 'deterministic'. If no command can verify it, set verify_tier: 'semantic'.",
      "plan/spec.md must reproduce ALL core requirements from the Goal verbatim in the Scope section — do not omit, truncate, or paraphrase them.",
      "Create phased milestones in plan/milestones.md with at least one milestone per Goal feature.",
      "DESIGN QUALITY (mandatory): For EVERY module, the spec MUST include: (a) full CRUD — create, read-list, read-detail, update, DELETE with confirmation dialog; (b) operation feedback — toast or inline message after every write; (c) empty state — guidance text + action button on empty lists; (d) form validation — inline errors on required fields, loading/disabled submit button; (e) loading indicator on every async fetch; (f) error handling — retry on network error, back-nav on not-found. plan/acceptance.json MUST contain at least one criterion per design quality rule with verify_command using grep to check for toast/dialog/empty-state patterns in frontend source.",
    ],
  },
  arch_design: {
    title: "Architecture Design",
    required_artifacts: ["plan/arch.md", "plan/interfaces.md", "risk/risk_report.json", "plan/workplan.md", "plan/workplan.json"],
    instructions: [
      "Provide architecture decisions with tradeoffs and module boundaries.",
      "Publish top risks and mitigations in risk/risk_report.json.",
      "Split implementation work for FE/BE/QA in plan/workplan.md using the structured task list format: '## BE Tasks' and '## FE Tasks' sections, each task as '- [ ] T-BE-N: <description> | verify: <concrete assertion>', max 8 tasks per team, ordered by dependency.",
      "Also write plan/workplan.json with structured be_tasks and fe_tasks arrays so implementation steps can consume the workplan without markdown parsing.",
      "Write handoff/architect_to_impl.json with from_step, to_steps, modules, interfaces, decisions, risks, and a top-level workplan object containing be_tasks and fe_tasks arrays that mirror plan/workplan.json.",
      "For minimal or reviewable CRM goals, do not include responsive design, mobile polish, breakpoints, or device-specific styling tasks unless the goal explicitly requests them.",
      "CRITICAL: plan/interfaces.md MUST list EVERY API endpoint as a '## METHOD /path' heading (e.g. '## GET /api/books', '## DELETE /api/books/:id'). Include ALL CRUD methods (GET/POST/PUT/DELETE) for each resource mentioned in the spec. If the spec mentions delete functionality, you MUST include DELETE endpoints. Omitting endpoints will fail validation.",
      "Under each plan/interfaces.md heading, include request shape, response shape or payload shape, and auth requirement.",
      "Keep the Interfaces section in plan/arch.md brief and point it to plan/interfaces.md for the concrete contract list.",
      "Document the application boot strategy so preview deployment can infer an entrypoint or manifest.",
      "FE TASK STRUCTURE (v3.5): fe_tasks in workplan.json MUST follow skeleton-first order: first 2-3 tasks create app shell (navigation between ALL modules, shared components like toast/dialog/loading/empty-state), then per-module tasks. Every spec module MUST have at least 2 fe_tasks (list+CRUD view, forms+validation). If spec has N modules, emit at least N*2+3 fe_tasks.",
      "DELETE ENDPOINTS: Every entity MUST have a DELETE endpoint in interfaces.md with error response shapes (400, 404, 500).",
    ],
  },
  impl_fe: {
    title: "Frontend Implementation",
    required_artifacts: ["impl/fe_changes/public/index.html", "impl/fe_changes/public/app.js", "impl/fe_notes.md"],
    instructions: [
      "If plan/workplan.md exists, read the '## FE Tasks' section and execute tasks in listed order, checking each task's verify condition before proceeding to the next.",
      "Prefer plan/workplan.json when present; use its structured FE task list before falling back to plan/workplan.md.",
      "Implement frontend outputs as complete files under impl/fe_changes/public/.",
      "Consume handoff/be_to_fe.json as the backend contract source for API usage.",
      "Use same-origin relative API paths only. Do not hardcode localhost URLs.",
      "Write impl/fe_notes.md with UI decisions, assumptions, and run instructions.",
      "No placeholders: every component and handler must be complete. After each file, self-review that all API calls match handoff/be_to_fe.json and all DOM IDs are consistent.",
      "Verification before completion: confirm every user journey in plan/acceptance.json is wired up end-to-end before finishing.",
      "TWO-PHASE FE (v3.5 mandatory): Phase 1 — Skeleton: implement app shell first (nav/sidebar listing ALL modules, shared components: showToast, showConfirmDialog, renderLoading, renderEmptyState, client-side routing). Phase 2 — Per-Module: for EACH module implement list view (with empty state + loading), create/edit forms (with validation + submit loading), delete (with confirmation dialog), success/error toast after every write. Do NOT stop after one module — implement ALL modules from the spec.",
    ],
  },
  impl_fe_skeleton: {
    title: "Frontend Skeleton",
    required_artifacts: ["impl/fe_changes/public/index.html", "impl/fe_changes/public/app.js"],
    instructions: [
      "This step builds ONLY the app shell. Do NOT implement any module's CRUD views yet.",
      "Read plan/workplan.json and identify ALL modules from the spec (e.g. Customer Management, Tickets, Dashboard).",
      "Consume handoff/be_to_fe.json as the backend contract source for API usage.",
      "Build index.html with: (a) full page layout with sidebar/nav, (b) nav items for EVERY module from the spec, (c) CSS styles for layout, cards, tables, forms, buttons, modals, toasts, loading spinners, empty states, badges.",
      "Build app.js with: (a) showToast(message, type) function, (b) showConfirmDialog(message, onConfirm) function, (c) renderLoading(container) function, (d) renderEmptyState(container, message, actionLabel, onAction) function, (e) escapeHtml(text) function, (f) apiFetch(url, options) wrapper with error handling, (g) client-side routing via switchView() that calls render functions for each module, (h) setupNavigation() to wire nav clicks, (i) STUB render function for each module: renderDashboardView(), renderCustomersView(), renderTicketsView(), etc. — each stub should call renderLoading() then show a placeholder.",
      "Use same-origin relative API paths only. Do not hardcode localhost URLs.",
      "The skeleton must be a working app that shows navigation between all modules with loading placeholders.",
    ],
  },
  impl_fe_modules: {
    title: "Frontend Module Implementation",
    required_artifacts: ["impl/fe_changes/public/app.js", "impl/fe_notes.md"],
    instructions: [
      "This step fills in ALL module views in the existing app.js skeleton created by the previous step.",
      "Read the existing impl/fe_changes/public/app.js — it already has the app shell, shared components (showToast, showConfirmDialog, renderLoading, renderEmptyState, escapeHtml, apiFetch), navigation, and stub render functions.",
      "Read plan/workplan.json for the FE task list. Read handoff/be_to_fe.json for API contracts.",
      "For EACH module, replace the stub render function with a full implementation: (a) list view with table, empty state when no data, loading indicator during fetch; (b) create form in a modal with required field validation and inline errors; (c) edit form pre-populated with existing data; (d) delete with showConfirmDialog confirmation; (e) showToast after every successful create/update/delete; (f) error handling with showToast on failure.",
      "CRITICAL: You MUST implement ALL modules — not just the first one. If the spec has Customer, Tickets, Dashboard — implement all three. Count the modules and verify you implemented each one before finishing.",
      "Keep ALL existing shared components and navigation code intact. Only replace the stub render functions with full implementations.",
      "Write impl/fe_notes.md with UI decisions and module coverage summary.",
      "Verification: count the number of render*View functions that are fully implemented (not stubs). This count MUST equal the number of modules from the spec.",
    ],
  },
  impl_be: {
    title: "Backend Implementation",
    required_artifacts: ["impl/be_changes/server.js", "impl/be_changes/package.json", "impl/be_notes.md", "handoff/be_to_fe.json"],
    instructions: [
      "If plan/workplan.md exists, read the '## BE Tasks' section and execute tasks in listed order, checking each task's verify condition before proceeding to the next.",
      "Prefer plan/workplan.json when present; use its structured BE task list before falling back to plan/workplan.md.",
      "Implement backend/API/data layer outputs as complete files under impl/be_changes/.",
      "Write impl/be_changes/package.json with all runtime dependencies required by server.js.",
      "Configure server.js to serve static files from impl/be_changes/public/ and return public/index.html from GET /.",
      "server.js must respect process.env.PORT when provided.",
      "Write impl/be_notes.md with implementation decisions, assumptions, and run instructions.",
      "Emit handoff/be_to_fe.json with api_contracts, shared_types, and explicit scope_constraints.",
      "No placeholders: every function must be complete. After each file, self-review that all imports resolve and no stubs remain.",
      "Verification before completion: confirm every endpoint in plan/interfaces.md is implemented before writing handoff/be_to_fe.json.",
    ],
  },
  smoke_test: {
    title: "Smoke Test",
    required_artifacts: ["smoke/smoke_result.json"],
    instructions: [
      "Install backend dependencies from impl/be_changes/package.json.",
      "Assemble impl/fe_changes/public/* into impl/be_changes/public/ before starting the server.",
      "Start the backend on test port 13099 using process.env.PORT.",
      "Run L1 validation: confirm GET / responds with HTML or HTTP 200.",
      "Run L2 validation: if a primary API endpoint can be identified, capture its HTTP status and response sample.",
      "Always write smoke/smoke_result.json, even when checks fail."
    ]
  },
  qa_verify: {
    title: "QA Verification",
    required_artifacts: ["verify/qa_report.json"],
    instructions: [
      "Read plan/acceptance.json to get the acceptance criteria for this project.",
      "For each acceptance criterion, evaluate whether impl/fe_changes/* and impl/be_changes/* satisfy it — cite specific file names or code as evidence.",
      "Run deterministic checks: confirm required files exist (plan/acceptance.json, impl/be_changes/, impl/fe_changes/). Note any missing files as fail.",
      "If smoke/smoke_result.json exists, cite its root_check and api_check fields as deterministic evidence using exact status codes and response samples.",
      "Run semantic checks: does the implementation address the stated goal? Does UI match acceptance criteria?",
      "Write verify/qa_report.json: checks array items use fields check_id, layer (deterministic|semantic), description, status (pass|fail|warning), detail. journey_checks items use journey_id, description, status, evidence (string array). rubric_citations use term, criterion, evidence (string), pass (boolean).",
      "Every check detail field must quote a specific observation — never use 'pending human review' or placeholder text.",
      "UX QUALITY GATE (v3.5): In addition to acceptance criteria, check these UX dimensions and include them in rubric_citations. FAIL_IF_MISSING items must set overall_status to 'fail' if not found: (1) CRUD Delete — grep delete/remove in server.js+app.js, (2) Operation Feedback — grep toast/notification in app.js, (3) Empty State — grep empty/no-found in app.js, (4) Navigation — grep nav/sidebar in index.html, (5) Module Coverage — count spec modules vs app.js views. WARN_IF_MISSING: form validation, loading states.",
    ],
  },
  release_pack: {
    title: "Release Pack",
    required_artifacts: ["release/release_notes.md", "release/artifact_manifest.json", "release/README.md", "release/start.sh"],
    instructions: [
      "Assemble release/release_notes.md as the human-readable package summary.",
      "CRITICAL: release_notes.md MUST describe ONLY features actually implemented and verified. Read verify/qa_report.json and smoke/smoke_result.json to determine which features passed. Do NOT copy the Goal verbatim as 'what was built'. If a Goal feature was not implemented or failed QA, state it explicitly as 'Not included in this release'. Overstating delivered functionality is a release governance violation.",
      "Assemble release/artifact_manifest.json as the machine-readable package manifest.",
      "Assemble impl/fe_changes/public/* into impl/be_changes/public/ before writing run instructions.",
      "Write release/README.md using real commands inferred from impl/be_changes/package.json, impl/be_changes/server.js, and smoke/smoke_result.json when present.",
      "Write release/start.sh as an executable script that runs the application from impl/be_changes.",
      "Ensure artifact references are complete and traceable.",
    ],
  },
  deploy_preview: {
    title: "Deploy Preview",
    required_artifacts: ["preview/deployment_result.json"],
    instructions: [
      "Validate preview eligibility using static dependency scanning before any remote deployment call.",
      "Attempt preview deployment only when required credentials and deployment metadata are available.",
      "Persist preview/deployment_result.json with preview_url or an explicit fallback reason.",
    ],
  },
};

export function buildStepPrompt({ run, stepDef, input, payload, promptScript = null }) {
  const c = STEP_CONTRACTS[stepDef.id] || null;
  const goal = String(input.goal || input.task_prompt || input.prompt || "Build a minimal web app").trim();
  const title = c?.title || stepDef.id;
  const required = Array.isArray(c?.required_artifacts) ? c.required_artifacts : [];
  const lines = Array.isArray(c?.instructions) ? c.instructions : [];
  const artifactRoot = String(payload.artifact_root || "");
  const artifactAbs = path.posix.join("/workspace", artifactRoot).replace(/\/+/g, "/");

  const outputReq = required.length > 0
    ? `Required artifacts (relative to ${artifactRoot}):\n- ${required.join("\n- ")}`
    : "Required artifacts: follow workflow step contract.";
  const handoffReq = payload?.handoff_contract_out
    ? [
        `Handoff artifacts for next stage:`,
        `- ${Array.isArray(payload.handoff_contract_out.required_artifacts) ? payload.handoff_contract_out.required_artifacts.join("\n- ") : ""}`,
        payload.handoff_contract_out.typed_handoff?.file
          ? `- typed handoff manifest: ${payload.handoff_contract_out.typed_handoff.file}`
          : "",
      ].filter(Boolean).join("\n")
    : "";
  // For pm_spec: inject goal into the first instruction so the model cannot ignore it.
  const projectType = String(run?.project_type || "").trim();
  let effectiveLines = lines;
  if (String(stepDef?.id || "") === "pm_spec" && lines.length > 0) {
    // Pass the FULL goal — truncating to 200 chars loses numbered requirements
    const goalForPm = goal.length > 2000 ? goal.slice(0, 2000) + "..." : goal;
    effectiveLines = projectType && projectType !== "webapp_crm"
      ? [`Write ALL scope, user stories, and acceptance criteria SPECIFICALLY for this system. Do NOT use a CRM template or CRM-specific assumptions unless the goal explicitly asks for CRM behavior.\n\nFull Goal:\n${goalForPm}`, ...lines]
      : [`Write ALL scope, user stories, and acceptance criteria SPECIFICALLY for this system.\n\nFull Goal:\n${goalForPm}`, ...lines];
  }
  if (String(stepDef?.id || "") === "arch_design" && projectType === "single_file_html") {
    effectiveLines = [
      `Design SPECIFICALLY for a static single-file HTML landing page. Do NOT introduce CRM, database, auth, or CRUD modules unless the goal explicitly asks for them.\n\nFull Goal:\n${goal.length > 2000 ? goal.slice(0, 2000) + "..." : goal}`,
      "In plan/interfaces.md, include concrete markdown headings for the browser-facing contract such as '## GET /' and at least one static asset or UI event heading such as '## GET /styles.css' or '## Event: faq.toggle'.",
      "Under each plan/interfaces.md heading, write request shape, response or payload shape, and auth requirement. For static assets, state 'request body: none' and describe the returned asset.",
      "In plan/workplan.md, keep BE Tasks concrete for static hosting/assembly work such as serving public assets, wiring the root route, and packaging preview startup. Do not write vague 'N/A' placeholders.",
      ...lines,
    ];
  }
  const guidance = effectiveLines.length > 0
    ? `Execution requirements:\n- ${effectiveLines.join("\n- ")}`
    : "Execution requirements: complete this step with verifiable outputs.";
  const promptScriptNote = promptScript
    ? [
        `Prompt Script ID: ${promptScript.script_id}`,
        `Prompt Script LLM Role: ${promptScript.llm_role || promptScript.role || ""}`,
        `Prompt Script Artifact Type: ${promptScript.artifact_type}`,
        `Prompt Script Validation: ${JSON.stringify(promptScript.validation || {})}`,
        promptScript.system_prompt ? `Prompt Script Goal: ${promptScript.system_prompt}` : "",
      ].filter(Boolean).join("\n")
    : "";
  const executionAdapterNote = payload?.execution_adapter_packet
    ? [
        `Execution Adapter: ${payload.execution_adapter_packet.adapter_id}`,
        `Execution Target Paths: ${Array.isArray(payload.execution_adapter_packet.target_paths) ? payload.execution_adapter_packet.target_paths.join(", ") : ""}`,
        `Execution Required Outputs: ${Array.isArray(payload.execution_adapter_packet.required_outputs) ? payload.execution_adapter_packet.required_outputs.join(", ") : ""}`,
        payload.tool_adapter_request?.adapter_type
          ? `Tool Adapter Request: ${payload.tool_adapter_request.adapter_type}/${payload.tool_adapter_request.provider}`
          : "",
      ].filter(Boolean).join("\n")
    : "";
  return [
    `[CodingTeam Step] ${title}`,
    `Workflow: ${run.workflow_id}`,
    `Project Type: ${run.project_type}`,
    `Step ID: ${stepDef.id}`,
    `Role: ${stepDef.role}`,
    `Goal: ${goal}`,
    promptScriptNote,
    executionAdapterNote,
    guidance,
    outputReq,
    handoffReq,
    `Absolute artifact output root: ${artifactAbs}`,
    "Write files under the artifact output root exactly with the required relative paths.",
    "Constraints:",
    "- Prefer small, reviewable changes.",
    "- Keep outputs deterministic and explicit.",
    "- Include concise validation evidence.",
  ].join("\n");
}

export function validatePromptScriptBinding({ stepDef, promptScriptRegistry, promptScript }) {
  const promptScriptId = String(stepDef?.prompt_script_id || "").trim();
  if (!promptScriptId) {
    return { ok: true, prompt_script_id: "" };
  }
  if (!promptScriptRegistry || !promptScriptRegistry.scripts) {
    return { ok: false, code: "PROMPT_SCRIPT_REGISTRY_MISSING", detail: "prompt script registry not loaded" };
  }
  if (!promptScript) {
    return { ok: false, code: "PROMPT_SCRIPT_NOT_FOUND", detail: `prompt script '${promptScriptId}' not found` };
  }
  if (String(promptScript.role || "") !== String(stepDef.role || "")) {
    return {
      ok: false,
      code: "PROMPT_SCRIPT_ROLE_MISMATCH",
      detail: `prompt script '${promptScriptId}' role '${promptScript.role}' does not match step role '${stepDef.role}'`,
    };
  }
  return { ok: true, prompt_script_id: promptScriptId };
}
