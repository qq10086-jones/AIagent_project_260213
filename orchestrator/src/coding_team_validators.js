import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "./schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "contracts");
const PM_ACCEPTANCE_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_pm_acceptance.schema.json"), "utf8")
);
const ARCH_RISK_REPORT_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_arch_risk_report.schema.json"), "utf8")
);
const CODING_TEAM_WORKPLAN_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_workplan.schema.json"), "utf8")
);
const RELEASE_ARTIFACT_MANIFEST_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "coding_team_release_artifact_manifest.schema.json"), "utf8")
);

export const PM_REQUIRED_FILES = ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"];
export const PM_REQUIRED_SECTION_MATCHERS = [
  { id: "scope", patterns: ["scope"] },
  { id: "user_stories", patterns: ["user stor", "user stories"] },
  { id: "acceptance_criteria", patterns: ["acceptance criteria"] },
  { id: "non_goals", patterns: ["non-goal", "non goal", "non_goals"] },
  { id: "artifact_list", patterns: ["artifact list", "artifacts"] },
];

export const ARCH_REQUIRED_FILES = ["plan/arch.md", "plan/interfaces.md", "risk/risk_report.json", "plan/workplan.md", "plan/workplan.json"];
export const ARCH_REQUIRED_SECTION_MATCHERS = [
  { id: "module_breakdown", patterns: ["module"] },
  { id: "interfaces", patterns: ["interface"] },
  { id: "dependency_choices", patterns: ["dependency"] },
  { id: "risk_notes", patterns: ["risk"] },
];

const PM_SPEC_REQUIRED_HEADINGS = [
  { id: "scope", patterns: ["scope"] },
  { id: "user_stories", patterns: ["user stories", "user story"] },
  { id: "acceptance_criteria", patterns: ["acceptance criteria"] },
  { id: "non_goals", patterns: ["non-goals", "non goals", "non-goals"] },
  { id: "artifact_list", patterns: ["artifact list", "artifacts"] },
];

const ARCH_REQUIRED_HEADINGS = [
  { id: "module_breakdown", patterns: ["module breakdown", "modules"] },
  { id: "interfaces", patterns: ["interfaces", "interface contracts"] },
  { id: "dependency_choices", patterns: ["dependency choices", "dependencies"] },
  { id: "risk_notes", patterns: ["risk notes", "risks"] },
];

const INTERFACE_HEADING_PATTERNS = [
  /^get\s+\S+/,
  /^post\s+\S+/,
  /^put\s+\S+/,
  /^patch\s+\S+/,
  /^delete\s+\S+/,
  /^interface[:\s]/,
  /^event[:\s]/,
  /^rpc[:\s]/,
];

function collectWorkplanTaskText(workplanJson) {
  const allTasks = [
    ...(Array.isArray(workplanJson?.be_tasks) ? workplanJson.be_tasks : []),
    ...(Array.isArray(workplanJson?.fe_tasks) ? workplanJson.fe_tasks : []),
  ];
  return allTasks
    .filter((item) => item && typeof item === "object")
    .map((item) => `${String(item.id || "").trim()} ${String(item.description || "").trim()} ${String(item.verify || "").trim()}`.trim())
    .join("\n")
    .toLowerCase();
}

function normalizeAcceptanceCriteria(acceptanceJson) {
  if (Array.isArray(acceptanceJson?.criteria)) {
    return acceptanceJson.criteria.map((item) => String(item || "").trim()).filter(Boolean);
  }
  return [];
}

function extractInterfaceContracts(interfacesText = "") {
  const headings = extractMarkdownHeadings(interfacesText);
  return headings
    .map((heading) => {
      const match = heading.match(/^(get|post|put|patch|delete)\s+(\S+)/i);
      if (!match) return null;
      return {
        method: String(match[1] || "").toUpperCase(),
        path: String(match[2] || "").trim(),
      };
    })
    .filter(Boolean);
}

function corpusIncludesAny(corpus, patterns = []) {
  return patterns.some((pattern) => pattern.test(corpus));
}

function validateWorkplanProjectFit({
  archText = "",
  interfacesText = "",
  specText = "",
  acceptanceJson = null,
  workplanJson = null,
}) {
  const corpus = [
    String(archText || ""),
    String(interfacesText || ""),
    String(specText || ""),
    collectWorkplanTaskText(workplanJson),
  ].join("\n").toLowerCase();
  const looksLikeCrmFlow = /\/api\/customers/.test(corpus);
  if (!looksLikeCrmFlow) return [];

  const errors = [];
  const acceptanceCriteria = normalizeAcceptanceCriteria(acceptanceJson).join("\n").toLowerCase();
  const interfaceContracts = extractInterfaceContracts(interfacesText);
  const allowedMethods = new Set(interfaceContracts.map((item) => item.method));
  const beTasks = Array.isArray(workplanJson?.be_tasks) ? workplanJson.be_tasks : [];
  const feTasks = Array.isArray(workplanJson?.fe_tasks) ? workplanJson.fe_tasks : [];
  const isMinimalScope = /\b(minimal|reviewable|lightweight|small)\b/.test(`${specText}\n${acceptanceCriteria}`.toLowerCase());

  if (/\b(flask|django|fastapi|python app|python backend)\b/.test(corpus)) {
    errors.push("workplan_project_fit:python_runtime_conflicts_with_node_backend_contract");
  }
  if (/\b(api[-\s]?key auth|x-api-key|authentication middleware|auth middleware|requests without .*401)\b/.test(corpus)) {
    errors.push("workplan_project_fit:unexpected_auth_scope_for_minimal_crm");
  }
  if (!allowedMethods.has("DELETE") && /\bdelete\b/.test(corpus)) {
    errors.push("workplan_project_fit:delete_scope_not_present_in_interfaces");
  }
  if (!corpusIncludesAny(acceptanceCriteria, [/\bpaginat/, /\bpage\b/]) && /\bpaginat/.test(corpus)) {
    errors.push("workplan_project_fit:pagination_not_requested_by_acceptance");
  }
  if (!corpusIncludesAny(`${specText}\n${acceptanceCriteria}`.toLowerCase(), [/\bmobile\b/, /\bresponsive\b/]) && /\b(mobile|responsive)\b/.test(corpus)) {
    errors.push("workplan_project_fit:responsive_scope_not_requested");
  }
  if (isMinimalScope) {
    if (beTasks.length > 5) errors.push("workplan_project_fit:be_task_count_exceeds_minimal_scope");
    if (feTasks.length > 5) errors.push("workplan_project_fit:fe_task_count_exceeds_minimal_scope");
  }
  return errors;
}

function readJsonFile(absPath) {
  try {
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

function readTextFile(absPath) {
  try {
    return fs.readFileSync(absPath, "utf8");
  } catch {
    return "";
  }
}

function extractMarkdownHeadings(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^#{1,6}\s+/.test(line))
    .map((line) => line.replace(/^#{1,6}\s+/, "").trim().toLowerCase());
}

function findMissingHeadings(text, headingMatchers = []) {
  const headings = extractMarkdownHeadings(text);
  return headingMatchers
    .filter((matcher) => !matcher.patterns.some((pattern) => headings.some((heading) => heading.includes(pattern))))
    .map((matcher) => matcher.id);
}

function validateInterfacesDocument(text) {
  const headings = extractMarkdownHeadings(text);
  if (headings.length === 0) {
    return {
      ok: false,
      code: "ARCH_INTERFACES_EMPTY",
      detail: "plan/interfaces.md does not contain any markdown headings",
      headings: [],
    };
  }
  const interfaceHeadings = headings.filter((heading) =>
    INTERFACE_HEADING_PATTERNS.some((pattern) => pattern.test(heading))
  );
  if (interfaceHeadings.length === 0) {
    return {
      ok: false,
      code: "ARCH_INTERFACES_UNSPECIFIED",
      detail: "plan/interfaces.md must contain at least one concrete interface heading",
      headings,
    };
  }
  return { ok: true, headings, interface_headings: interfaceHeadings };
}

function normalizeRoot(workspaceRoot, artifactRoot) {
  const relRoot = String(artifactRoot || "").trim().replace(/\\/g, "/");
  if (!relRoot) return null;
  return path.resolve(workspaceRoot, relRoot);
}

export function validatePmOutput({ workspaceRoot, artifactRoot }) {
  const rootAbs = normalizeRoot(workspaceRoot, artifactRoot);
  if (!rootAbs) {
    return { checked: true, ok: false, code: "PM_ARTIFACT_ROOT_MISSING", detail: "artifact_root missing" };
  }

  const specPath = path.resolve(rootAbs, PM_REQUIRED_FILES[0]);
  const acceptancePath = path.resolve(rootAbs, PM_REQUIRED_FILES[1]);
  const milestonesPath = path.resolve(rootAbs, PM_REQUIRED_FILES[2]);

  const missingFiles = [specPath, acceptancePath, milestonesPath].filter((item) => !fs.existsSync(item));
  if (missingFiles.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "PM_REQUIRED_FILES_MISSING",
      detail: `missing PM files: ${missingFiles.map((item) => path.basename(item)).join(", ")}`,
    };
  }

  const specText = readTextFile(specPath).toLowerCase();
  const rawSpecText = readTextFile(specPath);
  const rawMilestoneText = readTextFile(milestonesPath);
  const milestoneText = rawMilestoneText.toLowerCase();
  const acceptanceJson = readJsonFile(acceptancePath);
  const missingSections = [];
  const missingHeadings = findMissingHeadings(rawSpecText, PM_SPEC_REQUIRED_HEADINGS);
  missingSections.push(...missingHeadings);
  for (const matcher of PM_REQUIRED_SECTION_MATCHERS) {
    const found = matcher.patterns.some((pattern) => specText.includes(pattern) || milestoneText.includes(pattern));
    if (!found) {
      if (!missingSections.includes(matcher.id)) missingSections.push(matcher.id);
    }
  }
  const acceptanceErrors = validateJsonSchemaLite(PM_ACCEPTANCE_SCHEMA, acceptanceJson, "$");
  if (acceptanceErrors.length > 0) {
    missingSections.push(...acceptanceErrors.map((item) => `acceptance.json:${item}`));
  }

  if (missingSections.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "PM_REQUIRED_SECTIONS_MISSING",
      detail: `missing PM sections: ${missingSections.join(", ")}`,
    };
  }

  return {
    checked: true,
    ok: true,
    files_checked: PM_REQUIRED_FILES,
    headings_checked: PM_SPEC_REQUIRED_HEADINGS.map((item) => item.id),
    schema_checked: PM_ACCEPTANCE_SCHEMA.$id,
  };
}

export function validateArchitectOutput({ workspaceRoot, artifactRoot }) {
  const rootAbs = normalizeRoot(workspaceRoot, artifactRoot);
  if (!rootAbs) {
    return { checked: true, ok: false, code: "ARCH_ARTIFACT_ROOT_MISSING", detail: "artifact_root missing" };
  }

  const archPath = path.resolve(rootAbs, ARCH_REQUIRED_FILES[0]);
  const interfacesPath = path.resolve(rootAbs, ARCH_REQUIRED_FILES[1]);
  const riskPath = path.resolve(rootAbs, ARCH_REQUIRED_FILES[2]);
  const workplanPath = path.resolve(rootAbs, ARCH_REQUIRED_FILES[3]);
  const workplanJsonPath = path.resolve(rootAbs, ARCH_REQUIRED_FILES[4]);

  const missingFiles = [archPath, interfacesPath, riskPath, workplanPath, workplanJsonPath].filter((item) => !fs.existsSync(item));
  if (missingFiles.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "ARCH_REQUIRED_FILES_MISSING",
      detail: `missing Architect files: ${missingFiles.map((item) => path.basename(item)).join(", ")}`,
    };
  }

  const rawArchText = readTextFile(archPath);
  const archText = rawArchText.toLowerCase();
  const rawInterfacesText = readTextFile(interfacesPath);
  const rawSpecText = readTextFile(path.resolve(rootAbs, "plan/spec.md"));
  const acceptanceJson = readJsonFile(path.resolve(rootAbs, "plan/acceptance.json"));
  const workplanText = readTextFile(workplanPath).toLowerCase();
  const riskJson = readJsonFile(riskPath);
  const workplanJson = readJsonFile(workplanJsonPath);
  const missingSections = [];
  const missingHeadings = findMissingHeadings(rawArchText, ARCH_REQUIRED_HEADINGS);
  missingSections.push(...missingHeadings);
  for (const matcher of ARCH_REQUIRED_SECTION_MATCHERS) {
    const found = matcher.patterns.some((pattern) => archText.includes(pattern) || workplanText.includes(pattern));
    if (!found) {
      if (!missingSections.includes(matcher.id)) missingSections.push(matcher.id);
    }
  }
  const riskErrors = validateJsonSchemaLite(ARCH_RISK_REPORT_SCHEMA, riskJson, "$");
  if (riskErrors.length > 0) {
    missingSections.push(...riskErrors.map((item) => `risk_report.json:${item}`));
  }
  const workplanErrors = validateJsonSchemaLite(CODING_TEAM_WORKPLAN_SCHEMA, workplanJson, "$");
  if (workplanErrors.length > 0) {
    missingSections.push(...workplanErrors.map((item) => `workplan.json:${item}`));
  }
  if (workplanErrors.length === 0) {
    missingSections.push(...validateWorkplanProjectFit({
      archText: rawArchText,
      interfacesText: rawInterfacesText,
      specText: rawSpecText,
      acceptanceJson,
      workplanJson,
    }));
  }
  const interfacesValidation = validateInterfacesDocument(rawInterfacesText);
  if (!interfacesValidation.ok) {
    missingSections.push(interfacesValidation.code);
  }

  if (missingSections.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "ARCH_REQUIRED_SECTIONS_MISSING",
      detail: `missing Architect sections: ${missingSections.join(", ")}`,
    };
  }

  return {
    checked: true,
    ok: true,
    files_checked: ARCH_REQUIRED_FILES,
    headings_checked: ARCH_REQUIRED_HEADINGS.map((item) => item.id),
    interfaces_checked: interfacesValidation.interface_headings,
    schema_checked: ARCH_RISK_REPORT_SCHEMA.$id,
    workplan_schema_checked: CODING_TEAM_WORKPLAN_SCHEMA.$id,
  };
}

export const RELEASE_REQUIRED_FILES = ["release/release_notes.md", "release/artifact_manifest.json", "release/README.md", "release/start.sh"];

export function validateReleaseOutput({ workspaceRoot, artifactRoot }) {
  const rootAbs = normalizeRoot(workspaceRoot, artifactRoot);
  if (!rootAbs) {
    return { checked: true, ok: false, code: "RELEASE_ARTIFACT_ROOT_MISSING", detail: "artifact_root missing" };
  }

  const notesPath = path.resolve(rootAbs, "release/release_notes.md");
  const manifestPath = path.resolve(rootAbs, "release/artifact_manifest.json");
  const readmePath = path.resolve(rootAbs, "release/README.md");
  const startScriptPath = path.resolve(rootAbs, "release/start.sh");

  const missingFiles = [notesPath, manifestPath, readmePath, startScriptPath].filter((item) => !fs.existsSync(item));
  if (missingFiles.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "RELEASE_REQUIRED_FILES_MISSING",
      detail: `missing release files: ${missingFiles.map((item) => path.relative(rootAbs, item)).join(", ")}`,
    };
  }

  const notesText = readTextFile(notesPath);
  if (!notesText || notesText.trim().length < 10) {
    return {
      checked: true,
      ok: false,
      code: "RELEASE_NOTES_EMPTY",
      detail: "release/release_notes.md is empty or too short",
    };
  }

  const readmeText = readTextFile(readmePath);
  if (!readmeText || !/npm install/i.test(readmeText) || !/node server\.js/i.test(readmeText)) {
    return {
      checked: true,
      ok: false,
      code: "RELEASE_README_INVALID",
      detail: "release/README.md must include npm install and node server.js instructions",
    };
  }

  const startScriptText = readTextFile(startScriptPath);
  if (!startScriptText || !/node server\.js/i.test(startScriptText)) {
    return {
      checked: true,
      ok: false,
      code: "RELEASE_START_SCRIPT_INVALID",
      detail: "release/start.sh must include node server.js",
    };
  }

  const manifest = readJsonFile(manifestPath);
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    return {
      checked: true,
      ok: false,
      code: "RELEASE_MANIFEST_INVALID_JSON",
      detail: "release/artifact_manifest.json is not valid JSON",
    };
  }

  const schemaErrors = validateJsonSchemaLite(RELEASE_ARTIFACT_MANIFEST_SCHEMA, manifest, "$");
  if (schemaErrors.length > 0) {
    return {
      checked: true,
      ok: false,
      code: "RELEASE_MANIFEST_SCHEMA_INVALID",
      detail: `artifact_manifest.json schema errors: ${schemaErrors.join("; ")}`,
    };
  }

  return {
    checked: true,
    ok: true,
    files_checked: RELEASE_REQUIRED_FILES,
    schema_checked: RELEASE_ARTIFACT_MANIFEST_SCHEMA.$id,
    artifact_count: Array.isArray(manifest.artifacts) ? manifest.artifacts.length : 0,
  };
}
