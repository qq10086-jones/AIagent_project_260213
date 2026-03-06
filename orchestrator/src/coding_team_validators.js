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

export const PM_REQUIRED_FILES = ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"];
export const PM_REQUIRED_SECTION_MATCHERS = [
  { id: "scope", patterns: ["scope"] },
  { id: "user_stories", patterns: ["user stor", "user stories"] },
  { id: "acceptance_criteria", patterns: ["acceptance criteria"] },
  { id: "non_goals", patterns: ["non-goal", "non goal", "non_goals"] },
  { id: "artifact_list", patterns: ["artifact list", "artifacts"] },
];

export const ARCH_REQUIRED_FILES = ["plan/arch.md", "risk/risk_report.json", "plan/workplan.md"];
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
  const riskPath = path.resolve(rootAbs, ARCH_REQUIRED_FILES[1]);
  const workplanPath = path.resolve(rootAbs, ARCH_REQUIRED_FILES[2]);

  const missingFiles = [archPath, riskPath, workplanPath].filter((item) => !fs.existsSync(item));
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
  const workplanText = readTextFile(workplanPath).toLowerCase();
  const riskJson = readJsonFile(riskPath);
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
    schema_checked: ARCH_RISK_REPORT_SCHEMA.$id,
  };
}
