/**
 * composite_planner.js
 *
 * Multi-clause composite workflow routing, discovery payload builders,
 * vNext dispatch input builder, and task output formatting helpers.
 * Extracted from index.js as part of WS-11-05 decomposition.
 */

import { parseIntent } from "../nlp/router.js";

export const RE_COMPOSITE_CUE = /(?:\u7136\u540e|\u4e26\u4e14|\u540c\u65f6|\u63a5\u7740|\u968f\u540e|\u53e6\u5916|\u4ee5\u53ca|;|\uff1b|\n)/i;
export const RE_PRECLOSE = /(?:\u76d8\u5c3e|\u76e4\u5c3e|\u6536\u76d8\u524d|\u6536\u76e4\u524d|preclose)/i;
export const RE_POSTCLOSE = /(?:\u76d8\u540e|\u76e4\u5f8c|\u95ea\u8baf|\u9583\u8a0a|tdnet|postclose|post-close)/i;
export const RE_NEWS_DAILY = /(?:\u65e5\u62a5|daily report|\u5e02\u573a\u65b0\u95fb|news report)/i;
export const RE_NEWS_HOT = /(?:\u70ed\u70b9\u65b0\u95fb|\u71b1\u9ede\u65b0\u805e|hot news|trending news|latest hot|24h.*news|\u4e3b\u52a8.*\u65b0\u95fb|\u4e3b\u52d5.*\u65b0\u805e)/i;
export const RE_GEO_MARKET_IMPACT = /(?:\u4e2d\u4e1c|\u4e2d\u6771|geopolitic|middle east|ukraine|\u4fc4\u4e4c|\u5c40\u52bf|\u51b2\u7a81).*(?:\u65e5\u80a1|\u65e5\u672c\u80a1\u5e02|\u65e5\u672c\u5e02\u573a|\u80a1\u5e02|\u4ea4\u6613\u65e5|\u5efa\u8bae|\u5f71\u54cd|impact|next trading day|japan stocks)/i;
export const RE_DISCOVERY_CUE = /(?:\u5efa\u4ed3|\u5efa\u5009|\u4ed3\u4f4d|\u5009\u4f4d|\u9009\u80a1|\u9078\u80a1|\u6a19\u7684|\u6807\u7684|\u627e.*\u6807\u7684|\u5206\u6279|position plan|portfolio plan|discovery|build[- ]?position|entry plan|staged entry|candidates?|stock picks?|find .*stocks?|find .*candidates?|allocation)/i;
export const RE_DISCOVERY_INDEX = /(?:\u5efa\u4ed3|\u5efa\u5009|\u9009\u80a1|\u9078\u80a1|\u6807\u7684|\u6a19\u7684|\u5206\u6279|discovery|position plan|portfolio plan|build[- ]?position|entry plan|staged entry|candidates?|stock picks?|allocation)/i;

export function hasCompositeCue(text) {
  return RE_COMPOSITE_CUE.test(String(text || ""));
}

export function splitCompositeClauses(text) {
  const s = String(text || "")
    .replace(/[\uff1b;]/g, "|")
    .replace(/\n+/g, "|")
    .replace(/(?:\u7136\u540e|\u4e26\u4e14|\u540c\u65f6|\u63a5\u7740|\u968f\u540e|\u53e6\u5916|\u4ee5\u53ca)/g, "|");
  return s.split("|").map(x => x.trim()).filter(x => x.length >= 2);
}

export function hasDiscoveryCue(text) {
  return RE_DISCOVERY_CUE.test(String(text || ""));
}

export function parseCapitalJpy(text) {
  const s = String(text || "");
  let m = s.match(/([0-9]+(?:\.[0-9]+)?)\s*(?:w|W|\u4e07)/);
  if (m) return Math.round(Number(m[1]) * 10000);
  m = s.match(/([0-9]{2,9}(?:\.[0-9]+)?)\s*(?:\u65e5\u5143|\u5186|JPY)/i);
  if (m) return Math.round(Number(m[1]));
  return null;
}

export function extractGoalText(text) {
  const s = String(text || "").trim();
  if (!s) return "";
  const m1 = s.match(/(?:\u76ee\u6807)[:\uff1a]?\s*([^\uff0c\u3002\uff1b;\n]+)/);
  if (m1 && m1[1]) return `\u76ee\u6807${m1[1].trim()}`;
  const m2 = s.match(/([0-9]{1,2}\s*(?:\u4e2a\u6708|\u500b\u6708|\u6708)[^\uff0c\u3002\uff1b;\n]{0,40}?[0-9]{1,2}(?:\.[0-9]+)?\s*%[^\uff0c\u3002\uff1b;\n]{0,20})/);
  if (m2 && m2[1]) return m2[1].trim();
  return "";
}

export function buildDiscoveryPayloadFromText(text) {
  const s = String(text || "").trim();
  const payload = {};
  const isImmediateOpsQuery = /(\u672c\u91d1|\u8d44\u91d1|\u7a7a\u4ed3|\u4ed3\u4f4d|\u600e(?:\u4e48|\u6a23)\u64cd\u4f5c|\u5982\u4f55\u64cd\u4f5c|\u660e\u5929\u600e\u4e48\u64cd\u4f5c|\u660e\u65e5\u3069\u3046\u3059\u308b)/i.test(s);
  const capital = parseCapitalJpy(s);
  if (capital && Number.isFinite(capital) && capital > 0) payload.capital_base_jpy = capital;
  if (/(?:\u4f4e\u98ce\u9669|\u4f4e\u98a8\u96aa|\u7a33\u5065|\u7a69\u5065|\u4fdd\u5b88|conservative|low risk)/i.test(s)) {
    payload.risk_profile = "low";
  } else if (/(?:\u9ad8\u98ce\u9669|\u9ad8\u98a8\u96aa|\u6fc0\u8fdb|\u6fc0\u9032|\u8fdb\u53d6|\u9032\u53d6|aggressive|high risk)/i.test(s)) {
    payload.risk_profile = "high";
  } else if (/(?:\u4e2d\u98ce\u9669|\u4e2d\u98a8\u96aa|\u5e73\u8861|balanced|medium risk)/i.test(s)) {
    payload.risk_profile = "medium";
  }
  const jp = /(?:\bJP\b|\u65e5\u80a1|\u65e5\u672c|\u4e1c\u4eac|\u6771\u4eac)/i.test(s);
  const us = /(?:\bUS\b|\u7f8e\u80a1|\u7f8e\u56fd|\u7f8e\u570b)/i.test(s);
  if (jp && us) payload.market = "ALL";
  else if (jp) payload.market = "JP";
  else if (us) payload.market = "US";
  const mMonth = s.match(/([0-9]{1,2})\s*(?:\u4e2a\u6708|\u500b\u6708|\u6708)/);
  if (mMonth) payload.horizon_days = Number(mMonth[1]) * 30;
  const mRet = s.match(/([0-9]{1,2}(?:\.[0-9]+)?)\s*%/);
  if (mRet) payload.target_return_pct = Number(mRet[1]);
  const goalText = extractGoalText(s);
  if (goalText) {
    payload.goal = goalText;
  } else if (/(?:\u76ee\u6807|\u589e\u503c|\u56de\u62a5|\u6536\u76ca|\u5efa\u4ed3\u8ba1\u5212|\u5efa\u5009\u8a08\u5283|\u5206\u6279)/.test(s)) {
    payload.goal = s.slice(0, 120);
  }
  if (isImmediateOpsQuery) {
    payload.quick_mode = true;
    payload.time_budget_s = 75;
    payload.max_attempts = Math.min(Number(payload.max_attempts || 2), 2);
    payload.min_candidates = Math.min(Number(payload.min_candidates || 2), 2);
  } else if (!Number.isFinite(Number(payload.time_budget_s))) {
    payload.time_budget_s = 150;
  }
  return payload;
}

export function fallbackRouteClause(clause) {
  const raw = String(clause || "").trim();
  const lower = raw.toLowerCase();
  if (!raw) return null;
  if (RE_PRECLOSE.test(raw) || lower.includes("pre-close")) {
    return { tool_name: "news.preclose_brief_jp", payload: {} };
  }
  if (RE_POSTCLOSE.test(raw)) {
    return { tool_name: "news.tdnet_close_flash", payload: {} };
  }
  if (RE_NEWS_DAILY.test(raw) || (lower.includes("news") && lower.includes("report"))) {
    return { tool_name: "news.daily_report", payload: {} };
  }
  if (RE_NEWS_HOT.test(raw) || (lower.includes("hot") && lower.includes("news")) || (lower.includes("trending") && lower.includes("news"))) {
    return { tool_name: "news.active_hot_search", payload: { lookback_hours: 24, top_n: 8, include_positions: true } };
  }
  if (RE_GEO_MARKET_IMPACT.test(raw)) {
    return {
      tool_name: "quant.discovery_workflow",
      payload: { market: "JP", auto_expand_market: false, goal: raw.slice(0, 160), risk_profile: "medium" },
    };
  }
  if (hasDiscoveryCue(raw)) {
    return { tool_name: "quant.discovery_workflow", payload: buildDiscoveryPayloadFromText(raw) };
  }
  if (/\u8bbe\u7f6e.*\u8d44\u91d1|\u8bbe\u7f6e.*\u672c\u91d1|set.*capital|set account/i.test(raw)) {
    const m = raw.match(/([0-9]+(?:\.[0-9]+)?)/);
    const capital = m ? Number(m[1]) : null;
    if (capital) return { tool_name: "portfolio.set_account", payload: { starting_capital: capital, ccy: /usd/i.test(raw) ? "USD" : "JPY" } };
  }
  if (/(\u672c\u91d1|\u8d44\u91d1|\u7a7a\u4ed3|\u4ed3\u4f4d|\u600e(?:\u4e48|\u6a23)\u64cd\u4f5c|\u5982\u4f55\u64cd\u4f5c|\u660e\u5929\u600e\u4e48\u64cd\u4f5c|\u660e\u65e5\u3069\u3046\u3059\u308b)/i.test(raw)) {
    const payload = buildDiscoveryPayloadFromText(raw);
    if (/(\u65e5\u5143|\u5186|JPY)/i.test(raw)) payload.market = payload.market || "JP";
    if (!payload.goal) payload.goal = "\u7a7a\u4ed3\u72b6\u6001\u4e0b\u7684\u6b21\u65e5\u64cd\u4f5c\u5efa\u8bae";
    payload.quick_mode = true;
    payload.time_budget_s = 75;
    payload.max_attempts = Math.min(Number(payload.max_attempts || 2), 2);
    payload.min_candidates = Math.min(Number(payload.min_candidates || 2), 2);
    return { tool_name: "quant.discovery_workflow", payload };
  }
  return null;
}

export function extractRuleBasedStepsFromText(text) {
  const s = String(text || "");
  const out = [];
  const add = (idx, tool_name, payload = {}) => { if (idx < 0) return; out.push({ idx, tool_name, payload }); };

  add(s.search(RE_PRECLOSE), "news.preclose_brief_jp", {});
  add(s.search(RE_POSTCLOSE), "news.tdnet_close_flash", {});
  add(s.search(RE_NEWS_DAILY), "news.daily_report", {});
  add(s.search(RE_NEWS_HOT), "news.active_hot_search", { lookback_hours: 24, top_n: 8, include_positions: true });

  const idxGeo = s.search(RE_GEO_MARKET_IMPACT);
  if (idxGeo >= 0) {
    add(idxGeo, "news.active_hot_search", { lookback_hours: 24, top_n: 8, include_positions: true });
    add(idxGeo + 1, "quant.discovery_workflow", { market: "JP", auto_expand_market: false, goal: s.slice(0, 160), risk_profile: "medium" });
  }

  const idxDisc = hasDiscoveryCue(s) ? s.search(RE_DISCOVERY_INDEX) : -1;
  if (idxDisc >= 0) add(idxDisc, "quant.discovery_workflow", buildDiscoveryPayloadFromText(s));

  out.sort((a, b) => a.idx - b.idx);
  return out.map(({ tool_name, payload }) => ({ tool_name, payload }));
}

export async function planCompositeWorkflowFromText(userInput, memory = {}) {
  const ruleSteps = extractRuleBasedStepsFromText(userInput);
  if (ruleSteps.length >= 2) {
    return { name: `chat-composite-${Date.now()}`, steps: ruleSteps };
  }
  if (!hasCompositeCue(userInput)) return null;
  const clauses = splitCompositeClauses(userInput);
  if (clauses.length < 2) return null;

  const steps = [];
  const seen = new Set();
  const pushStep = (step) => {
    if (!step?.tool_name) return;
    const key = `${step.tool_name}|${JSON.stringify(step.payload || {})}`;
    if (seen.has(key)) return;
    seen.add(key);
    steps.push({ tool_name: step.tool_name, payload: step.payload || {} });
  };
  const localMemory = { ...(memory || {}) };
  for (const clause of clauses) {
    let intent;
    try { intent = await parseIntent(clause, localMemory); } catch { intent = null; }
    if (intent?.payload?.symbol) localMemory.last_symbol = intent.payload.symbol;
    if (intent?.requires_tools && intent?.tool_name && intent?.confidence >= 0.55) {
      pushStep({ tool_name: intent.tool_name, payload: intent.payload || {} });
      continue;
    }
    const fallback = fallbackRouteClause(clause);
    if (fallback?.tool_name) pushStep({ tool_name: fallback.tool_name, payload: fallback.payload || {} });
  }
  for (const step of ruleSteps) pushStep(step);
  if (steps.length < 2) return null;
  return { name: `chat-composite-${Date.now()}`, steps };
}

export function detectLanguageQuick(text) {
  const s = String(text || "");
  if (/[\u4e00-\u9fff]/.test(s)) return "zh";
  if (/[\u3040-\u30ff]/.test(s)) return "ja";
  return "en";
}

export function buildForcedIntentFromRule(text) {
  const step = fallbackRouteClause(text);
  if (!step?.tool_name) return null;
  return {
    intent: "ops",
    mode_suggested: "run",
    requires_tools: true,
    tool_name: step.tool_name,
    payload: step.payload || {},
    confidence: 0.99,
    language: detectLanguageQuick(text),
  };
}

export function buildVNextDispatchInput({ source = "api", rawInput = "", msg = null, payload = {} }) {
  if (msg) {
    return {
      source: "discord",
      raw_input: rawInput,
      channel_id: msg.channel?.id || "",
      thread_id: msg.id || "",
      user_id: msg.author?.id || "",
      attachments: Array.isArray(msg.attachments)
        ? msg.attachments.map(item => ({ filename: item.name || "", url: item.url || "", content_type: item.contentType || "" }))
        : [],
      ...payload,
    };
  }
  return { source, raw_input: rawInput, ...payload };
}

export function summarizeOutputBrief(output) {
  if (!output || typeof output !== "object") return "Done";
  if (String(output.provider_used || "").toLowerCase() === "codex" || output.command_used || output.files_changed || output.diff_stats) {
    const ok = output.ok === true;
    const files = Array.isArray(output.files_changed) ? output.files_changed.length : 0;
    return `${output.provider_used || "coding"} ${ok ? "ok" : "failed"} | files:${files}`;
  }
  const raw = output.analysis || output.summary || output.message || output.stdout || output.raw || "Done";
  return String(raw).replace(/\s+/g, " ").trim().slice(0, 120);
}

export function formatCodingDelegateResult(output, status, streamError = "", runId = "", taskId = "") {
  const out = (output && typeof output === "object") ? output : {};
  const isOk = status === "succeeded" && out.ok !== false;
  const provider = out.provider_used || "codex";
  const model = out.model_used || "default";
  const files = Array.isArray(out.files_changed) ? out.files_changed : [];
  const diff = out.diff_stats && typeof out.diff_stats === "object" ? out.diff_stats : {};
  const artifacts = out.artifacts && typeof out.artifacts === "object" ? out.artifacts : {};
  const diag = out.diagnostics && typeof out.diagnostics === "object" ? out.diagnostics : {};
  const fallbackError = out.error || streamError || "";

  const lines = [
    `[Coder] ${isOk ? "Delegation succeeded" : "Delegation failed"}`,
    `provider=${provider} | model=${model}`,
  ];
  if (runId) lines.push(`run_id=${runId}`);
  if (taskId) lines.push(`task_id=${taskId}`);
  lines.push(`files_changed=${files.length} | diff(+${Number(diff.added || 0)} / -${Number(diff.deleted || 0)})`);
  if (files.length > 0) {
    const preview = files.slice(0, 5).join(", ");
    lines.push(`changed: ${preview}${files.length > 5 ? ", ..." : ""}`);
  }
  const artifactPaths = [artifacts.diff_bundle, artifacts.raw_stdout, artifacts.raw_stderr, artifacts.test_log, artifacts.patch_file].filter(Boolean);
  if (artifactPaths.length > 0) {
    lines.push(`artifacts: ${artifactPaths.slice(0, 3).join(" | ")}${artifactPaths.length > 3 ? " | ..." : ""}`);
  }
  if (!isOk) {
    if (diag.error_code) lines.push(`error_code=${diag.error_code}`);
    if (fallbackError) lines.push(`error=${String(fallbackError)}`);
  }
  return lines.join("\n").slice(0, 1024);
}
