/**
 * xss_scanner.mjs — detect unescaped user data reaching HTML sinks.
 *
 * Sinks: innerHTML=, outerHTML=, insertAdjacentHTML(...)
 * A finding is raised when an interpolation inside the sink's RHS template
 * literal is not wrapped in escapeHtml() (or otherwise provably safe).
 */

import path from "path";
import { readSourceLines, findHtmlSinkAssignments, extractTemplateInterpolations, isSafeInterpolation } from "../lib/ast_parser.mjs";

export async function run({ workspaceRoot, artifactRoot }) {
  const started = Date.now();
  const findings = [];
  const scanned = [];

  const feAppJs = path.resolve(workspaceRoot, artifactRoot, "impl/fe_changes/public/app.js");
  const feIndex = path.resolve(workspaceRoot, artifactRoot, "impl/fe_changes/public/index.html");

  for (const absPath of [feAppJs]) {
    const source = readSourceLines(absPath);
    if (!source) continue;
    const relPath = path.relative(path.resolve(workspaceRoot, artifactRoot), absPath).replace(/\\/g, "/");
    scanned.push(relPath);

    const sinks = findHtmlSinkAssignments(source);
    for (const sink of sinks) {
      const interpolations = extractTemplateInterpolations(sink.rawExpression);
      for (const expr of interpolations) {
        if (isSafeInterpolation(expr)) continue;
        findings.push({
          severity: guessSeverity(expr, sink.kind),
          code: "XSS_UNESCAPED_INTERPOLATION",
          file: relPath,
          line: sink.startLine,
          sink: sink.kind,
          interpolation: expr.length > 100 ? `${expr.slice(0, 100)}…` : expr,
          snippet: sink.lineSnippet,
          detail: `${sink.kind} sink receives unescaped interpolation \`\${${expr}}\`; user-controlled data can inject HTML.`,
          fix_hint: `Wrap with escapeHtml(): \`\${escapeHtml(${expr})}\``,
        });
      }
    }
  }

  // Also light-check index.html for raw {{...}} patterns that look like unsanitized templating.
  // (skeleton step may emit template placeholders that should not reach runtime)
  const html = readSourceLines(feIndex);
  if (html) {
    const relHtml = path.relative(path.resolve(workspaceRoot, artifactRoot), feIndex).replace(/\\/g, "/");
    scanned.push(relHtml);
    const mustache = html.content.match(/\{\{\s*[a-zA-Z_$][\w.$]*\s*\}\}/g);
    if (mustache && mustache.length > 0) {
      findings.push({
        severity: "low",
        code: "HTML_UNRESOLVED_PLACEHOLDER",
        file: relHtml,
        line: null,
        detail: `Found ${mustache.length} mustache-style placeholder(s) in HTML — if these are rendered as-is, they may leak template variable names.`,
        fix_hint: "Ensure template placeholders are replaced at build time or removed.",
      });
    }
  }

  const critical = findings.filter((f) => f.severity === "critical").length;
  const high = findings.filter((f) => f.severity === "high").length;
  const medium = findings.filter((f) => f.severity === "medium").length;
  const low = findings.filter((f) => f.severity === "low").length;

  let status;
  if (critical > 0 || high > 0) status = "fail";
  else if (medium > 0) status = "pass_with_warnings";
  else status = "pass";

  return {
    scanner_id: "xss_scanner",
    status,
    findings,
    summary: { critical, high, medium, low, total: findings.length },
    scanned_files: scanned,
    duration_ms: Date.now() - started,
  };
}

function guessSeverity(expr, sink) {
  const lowered = expr.toLowerCase();
  // Likely numeric/count fields — low risk (type coercion to string is mostly safe)
  if (/\b(count|total|num|number|size|length|index|sum|avg|min|max|pages?|age)\b/.test(lowered)) {
    return "low";
  }
  // confirm dialog / modal message XSS is high-impact and easily triggered
  if (/message|confirm|dialog|name|title|description|comment|content|body|author|email|address|notes?|phone|url|link|subject|text/.test(lowered)) {
    return "critical";
  }
  if (sink === "insertAdjacentHTML") return "high";
  return "medium";
}
