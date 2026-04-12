/**
 * ast_parser.mjs — lightweight source scanner for static_audit
 *
 * Zero-dependency v1: line-based regex with template-literal awareness.
 * Not a real AST. Scope limited to common JS patterns used in generated
 * FE/BE code. Designed to be replaced with acorn later without changing
 * the scanner contracts.
 */

import fs from "fs";

/** Read source with line numbers. */
export function readSourceLines(absPath) {
  if (!fs.existsSync(absPath)) return null;
  const content = fs.readFileSync(absPath, "utf8");
  return { content, lines: content.split(/\r?\n/) };
}

/**
 * Find every assignment to innerHTML / outerHTML plus every insertAdjacentHTML() call.
 * Handles multi-line template literals by greedy-matching across lines.
 * Returns: [{kind, startLine, rawExpression, lineSnippet}]
 */
export function findHtmlSinkAssignments(source) {
  if (!source) return [];
  const findings = [];
  const content = source.content;

  // innerHTML / outerHTML assignments (may span multiple lines until ; or closing backtick).
  // We use a stateful scan rather than trying to craft one monster regex.
  const sinkRe = /\b(innerHTML|outerHTML)\s*=\s*/g;
  let m;
  while ((m = sinkRe.exec(content)) !== null) {
    const startIdx = m.index;
    const sinkKind = m[1];
    const endIdx = findStatementEnd(content, m.index + m[0].length);
    const expr = content.slice(m.index + m[0].length, endIdx).trim();
    const startLine = lineNumberAt(content, startIdx);
    findings.push({
      kind: sinkKind,
      startLine,
      rawExpression: expr,
      lineSnippet: truncateSnippet(content.slice(startIdx, endIdx)),
    });
  }

  // insertAdjacentHTML calls
  const insertRe = /\binsertAdjacentHTML\s*\(/g;
  while ((m = insertRe.exec(content)) !== null) {
    const startIdx = m.index;
    const endIdx = findBalancedClose(content, m.index + m[0].length - 1);
    const args = endIdx === -1 ? "" : content.slice(m.index + m[0].length, endIdx).trim();
    const startLine = lineNumberAt(content, startIdx);
    findings.push({
      kind: "insertAdjacentHTML",
      startLine,
      rawExpression: args,
      lineSnippet: truncateSnippet(content.slice(startIdx, endIdx === -1 ? startIdx + 100 : endIdx + 1)),
    });
  }

  return findings;
}

/**
 * Extract ${...} interpolation contents from a raw expression.
 * Returns array of strings (the code inside the braces).
 */
export function extractTemplateInterpolations(rawExpression) {
  if (!rawExpression || typeof rawExpression !== "string") return [];
  const results = [];
  let i = 0;
  while (i < rawExpression.length) {
    const dollar = rawExpression.indexOf("${", i);
    if (dollar === -1) break;
    const end = findBalancedClose(rawExpression, dollar + 1, "{", "}");
    if (end === -1) break;
    results.push(rawExpression.slice(dollar + 2, end).trim());
    i = end + 1;
  }
  return results;
}

/**
 * Check if an interpolation expression is "safe" (wrapped in escapeHtml, a literal,
 * a number, or a known-safe helper call).
 */
export function isSafeInterpolation(expr) {
  if (!expr) return true;
  const trimmed = expr.trim();
  // Literal string / number / boolean
  if (/^(['"`]).*\1$/.test(trimmed)) return true;
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return true;
  if (/^(true|false|null|undefined)$/.test(trimmed)) return true;
  // Wrapped in escapeHtml(...)
  if (/^escapeHtml\s*\(/.test(trimmed)) return true;
  // Known-safe conversions
  if (/^String\s*\(\s*escapeHtml\s*\(/.test(trimmed)) return true;
  // Ternary whose both branches are safe
  const tern = trimmed.match(/^(.+?)\s*\?\s*(.+?)\s*:\s*(.+)$/);
  if (tern) {
    return isSafeInterpolation(tern[2]) && isSafeInterpolation(tern[3]);
  }
  // Numeric arithmetic like items.length or foo + 1
  if (/\.length\b/.test(trimmed) && !/\[/.test(trimmed)) return true;
  return false;
}

// ── helpers ────────────────────────────────────────────────────────────────

function lineNumberAt(content, idx) {
  let line = 1;
  for (let i = 0; i < idx && i < content.length; i++) {
    if (content[i] === "\n") line++;
  }
  return line;
}

function truncateSnippet(s, max = 180) {
  const flat = s.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}

/**
 * Find the end of a statement starting at idx. Handles template literals and strings
 * so we don't treat backticks/quotes inside strings as terminators.
 * Returns index of the terminating ; or \n after balanced delimiters.
 */
function findStatementEnd(content, idx) {
  let i = idx;
  let depthParen = 0, depthBrace = 0, depthBracket = 0;
  while (i < content.length) {
    const ch = content[i];
    if (ch === "'" || ch === '"') {
      // skip string literal
      const quote = ch;
      i++;
      while (i < content.length && content[i] !== quote) {
        if (content[i] === "\\") i += 2; else i++;
      }
      i++;
      continue;
    }
    if (ch === "`") {
      i++;
      while (i < content.length && content[i] !== "`") {
        if (content[i] === "\\") { i += 2; continue; }
        if (content[i] === "$" && content[i + 1] === "{") {
          const closeIdx = findBalancedClose(content, i + 1, "{", "}");
          if (closeIdx === -1) return content.length;
          i = closeIdx + 1;
          continue;
        }
        i++;
      }
      i++;
      continue;
    }
    if (ch === "(") depthParen++;
    else if (ch === ")") depthParen--;
    else if (ch === "{") depthBrace++;
    else if (ch === "}") depthBrace--;
    else if (ch === "[") depthBracket++;
    else if (ch === "]") depthBracket--;
    else if ((ch === ";" || ch === "\n") && depthParen === 0 && depthBrace === 0 && depthBracket === 0) {
      return i;
    }
    i++;
  }
  return content.length;
}

/**
 * Find the matching close of an open bracket. The caller passes the index of the opening char.
 * Works with () or {}.
 */
function findBalancedClose(content, openIdx, openChar = "(", closeChar = ")") {
  // Auto-detect if caller passed a ( location
  if (content[openIdx] === "(") { openChar = "("; closeChar = ")"; }
  else if (content[openIdx] === "{") { openChar = "{"; closeChar = "}"; }
  let depth = 1;
  let i = openIdx + 1;
  while (i < content.length) {
    const ch = content[i];
    if (ch === "'" || ch === '"' || ch === "`") {
      const quote = ch;
      i++;
      while (i < content.length && content[i] !== quote) {
        if (content[i] === "\\") i += 2; else i++;
      }
      i++;
      continue;
    }
    if (ch === openChar) depth++;
    else if (ch === closeChar) {
      depth--;
      if (depth === 0) return i;
    }
    i++;
  }
  return -1;
}
