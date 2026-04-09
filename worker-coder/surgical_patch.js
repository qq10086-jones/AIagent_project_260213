/**
 * surgical_patch.js — Deterministic micro-fix engine for trivial syntax errors.
 * Phase 1 (P1-1): Fixes JSON/JS/Python syntax errors without LLM re-runs.
 *
 * Constraints:
 * - Max 1 fix attempt per delegateTask call (enforced by caller)
 * - All writes go through scope_guard.validateRequestedWrite
 * - Operates on executionWorkspaceRoot (isolated sandbox)
 * - Does NOT re-run static checks (caller handles recheck)
 */
import fs from "fs";
import path from "path";
import { normalizeRelPath, validateRequestedWrite } from "./scope_guard.js";

/**
 * Analyze static check records to determine if any are fixable by deterministic patch.
 * @param {{ records: Array<{file: string, kind: string, ok: boolean, exit_code: number, stderr: string}> }} opts
 * @returns {{ fixable: boolean, candidates: Array<{file: string, kind: string, fixType: string}> }}
 */
export function canFix({ records }) {
  if (!Array.isArray(records) || records.length === 0) return { fixable: false, candidates: [] };

  const candidates = [];
  for (const rec of records) {
    if (rec.ok) continue;
    const fixType = classifyFixType(rec);
    if (fixType) candidates.push({ file: rec.file, kind: rec.kind, fixType });
  }
  return { fixable: candidates.length > 0, candidates };
}

/**
 * Apply deterministic micro-fixes to files in the execution workspace.
 * @param {{ executionWorkspaceRoot: string, records: Array, allowedTargetPaths: string[], workspaceRoot: string }} opts
 * @returns {{ attempted: boolean, success: boolean, patches_applied: Array<{file: string, fixType: string, detail: string}>, failure_reason: string|null }}
 */
export function applySurgicalPatches({ executionWorkspaceRoot, records, allowedTargetPaths = [], workspaceRoot }) {
  if (!Array.isArray(records)) {
    return { attempted: false, success: false, patches_applied: [], failure_reason: "no records" };
  }

  // Check if any records actually need fixing
  const hasFailures = records.some((r) => !r.ok);
  if (!hasFailures) {
    return { attempted: false, success: false, patches_applied: [], failure_reason: "all records ok" };
  }

  const patches_applied = [];
  let anyFailed = false;

  for (const rec of records) {
    if (rec.ok) continue;
    const fixType = classifyFixType(rec);
    if (!fixType) continue;

    // Scope guard check
    const scopeCheck = validateRequestedWrite({
      workspaceRoot,
      targetPath: rec.file,
      allowedTargetPaths,
    });
    if (!scopeCheck.ok) {
      anyFailed = true;
      continue;
    }

    const absPath = path.resolve(executionWorkspaceRoot, normalizeRelPath(rec.file));
    if (!fs.existsSync(absPath)) {
      anyFailed = true;
      continue;
    }

    let content;
    try {
      content = fs.readFileSync(absPath, "utf8");
    } catch {
      anyFailed = true;
      continue;
    }

    const result = applyFix(fixType, content, rec.stderr);
    if (!result.ok) {
      anyFailed = true;
      continue;
    }

    try {
      fs.writeFileSync(absPath, result.content, "utf8");
      patches_applied.push({ file: rec.file, fixType, detail: result.detail });
    } catch {
      anyFailed = true;
    }
  }

  if (patches_applied.length === 0) {
    return { attempted: true, success: false, patches_applied: [], failure_reason: "no fixable errors or all fixes failed" };
  }

  return {
    attempted: true,
    success: !anyFailed && patches_applied.length > 0,
    patches_applied,
    failure_reason: anyFailed ? "some fixes failed" : null,
  };
}

// --- Internal helpers ---

function classifyFixType(rec) {
  if (!rec || rec.ok) return null;
  const stderr = String(rec.stderr || "").toLowerCase();

  if (rec.kind === "json_parse") {
    if (/trailing comma/i.test(stderr)) {
      return "json_trailing_comma";
    }
    // "Expected double-quoted property name" after a trailing comma before } or ]
    if (/expected.*property name/i.test(stderr)) {
      return "json_trailing_comma";
    }
    if (/expected.*','|unexpected (token|non-whitespace)/i.test(stderr) || /position\s+\d+/i.test(stderr)) {
      return "json_structure_fix";
    }
    return null;
  }

  if (rec.kind === "node_syntax") {
    if (/unexpected end of input/i.test(stderr)) {
      return "js_unclosed_bracket";
    }
    return null;
  }

  if (rec.kind === "py_compile") {
    if (/indentationerror/i.test(stderr)) {
      return "py_indentation";
    }
    return null;
  }

  return null;
}

function applyFix(fixType, content, stderr) {
  switch (fixType) {
    case "json_trailing_comma":
      return fixJsonTrailingComma(content);
    case "json_structure_fix":
      return fixJsonStructure(content, stderr);
    case "js_unclosed_bracket":
      return fixJsUnclosedBracket(content);
    case "py_indentation":
      return fixPyIndentation(content, stderr);
    default:
      return { ok: false, content, detail: "unknown fix type" };
  }
}

/**
 * Remove trailing commas before } or ] in JSON.
 */
function fixJsonTrailingComma(content) {
  // Match comma followed by optional whitespace/newlines then } or ]
  const fixed = content.replace(/,(\s*[}\]])/g, "$1");
  if (fixed === content) return { ok: false, content, detail: "no trailing comma found" };

  // Verify the fix actually produces valid JSON
  try {
    JSON.parse(fixed);
    return { ok: true, content: fixed, detail: "removed trailing comma(s)" };
  } catch {
    return { ok: false, content, detail: "trailing comma removal did not fix JSON" };
  }
}

/**
 * Attempt to fix JSON by trying common structural repairs:
 * - Insert missing comma between values
 * - Remove duplicate commas
 */
function fixJsonStructure(content, stderr) {
  // Strategy 1: Remove duplicate/extra commas
  let attempt = content.replace(/,(\s*,)+/g, ",");
  try {
    JSON.parse(attempt);
    return { ok: true, content: attempt, detail: "removed duplicate commas" };
  } catch { /* continue */ }

  // Strategy 2: Try inserting commas between adjacent string/number/bool/null values
  // Pattern: "value"\n"key" or "value"\nnumber
  attempt = content.replace(/(["}\]\d])\s*\n(\s*")/g, "$1,\n$2");
  try {
    JSON.parse(attempt);
    return { ok: true, content: attempt, detail: "inserted missing comma between values" };
  } catch { /* continue */ }

  // Strategy 3: Fix missing comma after } or ] followed by " or {
  attempt = content.replace(/([}\]])\s*\n(\s*["{[])/g, "$1,\n$2");
  try {
    JSON.parse(attempt);
    return { ok: true, content: attempt, detail: "inserted missing comma after block" };
  } catch { /* continue */ }

  return { ok: false, content, detail: "could not fix JSON structure" };
}

/**
 * Fix JS file with unclosed brackets at EOF by appending missing closers.
 */
function fixJsUnclosedBracket(content) {
  const counts = { "{": 0, "}": 0, "(": 0, ")": 0, "[": 0, "]": 0 };
  let inString = false;
  let stringChar = "";
  let escaped = false;

  for (const ch of content) {
    if (escaped) { escaped = false; continue; }
    if (ch === "\\") { escaped = true; continue; }
    if (inString) {
      if (ch === stringChar) inString = false;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      inString = true;
      stringChar = ch;
      continue;
    }
    if (ch in counts) counts[ch]++;
  }

  const missing = [];
  const diff_curly = counts["{"] - counts["}"];
  const diff_paren = counts["("] - counts[")"];
  const diff_square = counts["["] - counts["]"];

  // Only fix if exactly 1-2 brackets are missing (conservative)
  const totalMissing = Math.max(0, diff_curly) + Math.max(0, diff_paren) + Math.max(0, diff_square);
  if (totalMissing === 0 || totalMissing > 2) {
    return { ok: false, content, detail: `bracket imbalance too complex: {${diff_curly} (${diff_paren} [${diff_square}` };
  }

  for (let i = 0; i < diff_curly; i++) missing.push("}");
  for (let i = 0; i < diff_paren; i++) missing.push(")");
  for (let i = 0; i < diff_square; i++) missing.push("]");

  const fixed = content.trimEnd() + "\n" + missing.join("\n") + "\n";
  return { ok: true, content: fixed, detail: `appended ${missing.join("")} at EOF` };
}

/**
 * Fix Python indentation errors by normalizing mixed tabs/spaces to 4-space indent.
 */
function fixPyIndentation(content, stderr) {
  // Check if there are mixed tabs and spaces
  const lines = content.split("\n");
  let hasMixedIndent = false;
  for (const line of lines) {
    const leadingWhitespace = line.match(/^(\s*)/)[1];
    if (leadingWhitespace.includes("\t") && leadingWhitespace.includes(" ")) {
      hasMixedIndent = true;
      break;
    }
  }

  if (!hasMixedIndent && !content.includes("\t")) {
    return { ok: false, content, detail: "no mixed indentation detected" };
  }

  // Replace all leading tabs with 4 spaces
  const fixed = lines.map((line) => {
    const match = line.match(/^(\s*)(.*)/);
    if (!match) return line;
    const indent = match[1].replace(/\t/g, "    ");
    return indent + match[2];
  }).join("\n");

  if (fixed === content) return { ok: false, content, detail: "indentation unchanged" };
  return { ok: true, content: fixed, detail: "normalized tabs to 4-space indent" };
}
