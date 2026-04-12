/**
 * class_injection.mjs — detect unsafe interpolation of user values into CSS class attributes.
 *
 * Patterns we flag:
 *   1. class="...${expr}..."  where expr involves a known user-value identifier
 *      (status, priority, type, kind, state, category, role, severity)
 *   2. className=`...${expr}...` with same danger identifiers
 *   3. classList.add(expr) / classList.toggle(expr) / setAttribute("class", expr)
 *      where expr is derived from user data
 *
 * Safe forms (not flagged):
 *   - Whitelist lookup: CLASSES[value] || "default"
 *   - String literals: class="badge-open"
 *   - Ternary/switch over literal values: status === "open" ? "badge-open" : "badge-closed"
 */

import fs from "fs";
import path from "path";

const DANGER_IDENTS = /\b(status|priority|type|kind|state|category|role|severity|variant)\b/;

export async function run({ workspaceRoot, artifactRoot }) {
  const started = Date.now();
  const findings = [];
  const scanned = [];

  const targets = [
    path.resolve(workspaceRoot, artifactRoot, "impl/fe_changes/public/app.js"),
    path.resolve(workspaceRoot, artifactRoot, "impl/fe_changes/public/index.html"),
  ];

  for (const abs of targets) {
    if (!fs.existsSync(abs)) continue;
    const rel = path.relative(path.resolve(workspaceRoot, artifactRoot), abs).replace(/\\/g, "/");
    scanned.push(rel);
    const content = fs.readFileSync(abs, "utf8");

    // Pattern 1+2: class="..." or className="..." attribute with template interpolation
    const classAttrRe = /\b(class|className)\s*=\s*(?:"|'|`)([^"'`]*\$\{[^}]*\}[^"'`]*)(?:"|'|`)/g;
    let m;
    while ((m = classAttrRe.exec(content)) !== null) {
      const attrValue = m[2];
      const interpRe = /\$\{([^}]*)\}/g;
      let im;
      while ((im = interpRe.exec(attrValue)) !== null) {
        const expr = im[1].trim();
        if (isSafeClassExpression(expr)) continue;
        if (!DANGER_IDENTS.test(expr) && !/\.(toLowerCase|toUpperCase)\(\)/.test(expr)) continue;
        const line = lineNumberAt(content, m.index);
        findings.push({
          severity: "medium",
          code: "CSS_CLASS_INJECTION",
          file: rel,
          line,
          expression: expr.length > 80 ? `${expr.slice(0, 80)}…` : expr,
          snippet: truncate(m[0]),
          detail: `CSS class attribute interpolates user-controlled value \`${expr}\` directly. An attacker-controlled status/priority string could inject unintended class names.`,
          fix_hint: "Use a whitelist map: const CLASS_MAP = {open:'badge-open', ...}; class=\"${CLASS_MAP[value] || 'badge-default'}\"",
        });
      }
    }

    // Pattern 3: classList.add(expr) / classList.toggle(expr)
    const classListRe = /\bclassList\.(add|toggle|remove)\s*\(([^)]+)\)/g;
    while ((m = classListRe.exec(content)) !== null) {
      const argExpr = m[2].trim();
      // skip string literal args
      if (/^["'`][^"'`]*["'`](,\s*.*)?$/.test(argExpr)) continue;
      if (!DANGER_IDENTS.test(argExpr)) continue;
      if (isSafeClassExpression(argExpr)) continue;
      const line = lineNumberAt(content, m.index);
      findings.push({
        severity: "medium",
        code: "CSS_CLASSLIST_UNSAFE_VALUE",
        file: rel,
        line,
        expression: argExpr.length > 80 ? `${argExpr.slice(0, 80)}…` : argExpr,
        snippet: truncate(m[0]),
        detail: `classList.${m[1]}() receives user-controlled value directly.`,
        fix_hint: `Pass the whitelisted class name only: classList.${m[1]}(CLASS_MAP[value] || 'badge-default')`,
      });
    }

    // Pattern 4: setAttribute("class", expr) or setAttribute('className', expr)
    const setAttrRe = /\bsetAttribute\s*\(\s*["'](class|className)["']\s*,\s*([^)]+)\)/g;
    while ((m = setAttrRe.exec(content)) !== null) {
      const argExpr = m[2].trim();
      if (/^["'`][^"'`]*["'`]$/.test(argExpr)) continue;
      if (!DANGER_IDENTS.test(argExpr)) continue;
      if (isSafeClassExpression(argExpr)) continue;
      const line = lineNumberAt(content, m.index);
      findings.push({
        severity: "medium",
        code: "CSS_SETATTRIBUTE_UNSAFE_VALUE",
        file: rel,
        line,
        expression: argExpr.length > 80 ? `${argExpr.slice(0, 80)}…` : argExpr,
        snippet: truncate(m[0]),
        detail: `setAttribute("class", ...) receives user-controlled value directly.`,
        fix_hint: `Use a whitelist lookup before setAttribute.`,
      });
    }
  }

  const critical = findings.filter((f) => f.severity === "critical").length;
  const high = findings.filter((f) => f.severity === "high").length;
  const medium = findings.filter((f) => f.severity === "medium").length;
  const low = findings.filter((f) => f.severity === "low").length;
  const status = critical > 0 || high > 0 ? "fail" : medium > 0 ? "pass_with_warnings" : "pass";

  return {
    scanner_id: "class_injection",
    status,
    findings,
    summary: { critical, high, medium, low, total: findings.length },
    scanned_files: scanned,
    duration_ms: Date.now() - started,
  };
}

function isSafeClassExpression(expr) {
  const trimmed = expr.trim();
  // Object lookup — `MAP[value]` or `MAP[value] || 'default'`
  if (/^[A-Z_][A-Z0-9_]*\s*\[.+\](\s*\|\|\s*["'`][^"'`]*["'`])?$/.test(trimmed)) return true;
  // Parens-wrapped lookup
  if (/^\([A-Z_][A-Z0-9_]*\s*\[.+\].*\)$/.test(trimmed)) return true;
  // Ternary whose both branches are string literals
  const tern = trimmed.match(/^(.+?)\s*\?\s*(["'`][^"'`]*["'`])\s*:\s*(["'`][^"'`]*["'`])$/);
  if (tern) return true;
  return false;
}

function lineNumberAt(content, idx) {
  let line = 1;
  for (let i = 0; i < idx && i < content.length; i++) {
    if (content[i] === "\n") line++;
  }
  return line;
}

function truncate(s, max = 180) {
  const flat = s.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}
