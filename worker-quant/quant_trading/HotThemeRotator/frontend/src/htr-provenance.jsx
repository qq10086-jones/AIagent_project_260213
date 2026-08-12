// htr-provenance.jsx — the one shared data-provenance strip (P37-02, Rule 15.10.7
// + Rule 11.7 four-variant parity + Rule 11.9.4 honest degradation).
//
// Three questions that are really one question — "is what I am looking at real
// and current?" — were previously answered in three different places, or not at
// all:
//
//   * `meta.candidatesSource` ("screener_v2" | "sample") was emitted by the API
//     and consumed by NOTHING. The backend labelled its own fallback honestly
//     and the UI dropped the label on the floor.
//   * the section-level mock-fallback banner existed ONLY inside V3, so V1, V2
//     and V4 rendered sample markets/themes/news with no warning whatsoever.
//   * `meta.pipelineHealth` (Rule 15.10) had no surface at all, so a TDnet poll
//     failure or a partial event-universe refresh could not appear in the UI.
//
// This strip answers all three, once, above <main>, so every variant gets it by
// construction rather than by four call sites that can drift apart. Rule 15.10.7
// forbids showing the health aggregate without its components, so the component
// list is not collapsible, not truncated, and has no "+N more".
//
// Fail-closed in both directions. An ABSENT source is treated as sample, and an
// ABSENT health block is treated as unknown — never as real and never as green.
// The offline mock has neither field, and that is exactly the case where the
// screen must not look authoritative.

// Rule 11.9.4 — "real" is a positive assertion the payload has to make. Anything
// else (missing meta, missing field, an unrecognised producer) is sample.
// Deliberately NOT derived from candidates.length: a populated array proves the
// mock loaded, not that the screener ran.
function resolveCandidateSource(meta) {
  const source = (meta && meta.candidatesSource) || null;
  const real = source === "screener_v2";
  return {
    real,
    source: source || "absent",
    tradeDate: (meta && meta.tradeDate) || null,
    label: real ? "真实筛选" : "示例候选 SAMPLE",
  };
}

// Rule 15.10 — healthy | degraded | failed, plus `unknown` for "the backend did
// not tell us". Unknown is NOT healthy: a missing answer and a good answer must
// never render the same.
function resolvePipelineHealth(meta) {
  const raw = (meta && meta.pipelineHealth) || null;
  const status = (raw && raw.status) || "unknown";
  const known = status === "healthy" || status === "degraded" || status === "failed";
  return {
    status: known ? status : "unknown",
    asof: (raw && raw.asof) || null,
    summary: (raw && raw.summary) || null,
    components: (raw && Array.isArray(raw.degradedComponents)) ? raw.degradedComponents : [],
  };
}

const HTR_HEALTH_TONE = {
  healthy: { tone: "ok", label: "健康" },
  degraded: { tone: "warn", label: "降级" },
  failed: { tone: "bad", label: "失败" },
  // `unknown` is a WARNING tone on purpose. A missing answer and a good answer
  // must never render the same, and green is the one thing it must not be.
  unknown: { tone: "warn", label: "健康态未知" },
};

const HTR_SECTION_LABELS = {
  offline: "全部区块（后端不可达）",
  markets: "市场温度",
  themes: "主题热力",
  newsTimeline: "新闻",
  decisionLog: "决策日志",
  candidates: "候选清单",
  gates: "门槛",
  positions: "持仓",
  dailyCockpit: "Daily Cockpit",
};

function fallbackSections(degraded) {
  const deg = degraded || {};
  if (deg.offline) return ["offline"];
  return Object.keys(HTR_SECTION_LABELS).filter((k) => k !== "offline" && deg[k]);
}

function HtrProvenancePill({ tone, label, value, title }) {
  const palette = {
    ok: { fg: "var(--htr-bull)", bg: "var(--htr-bull-bg)", bd: "var(--htr-bull)" },
    warn: { fg: "var(--htr-warn)", bg: "var(--htr-warn-bg)", bd: "var(--htr-warn)" },
    bad: { fg: "var(--htr-bear)", bg: "var(--htr-bear-bg)", bd: "var(--htr-bear)" },
    mute: { fg: "var(--htr-ink-3)", bg: "var(--htr-surface-3)", bd: "var(--htr-line)" },
  }[tone] || { fg: "var(--htr-ink-3)", bg: "var(--htr-surface-3)", bd: "var(--htr-line)" };
  return (
    <span
      title={title || undefined}
      style={{
        display: "inline-flex", alignItems: "baseline", gap: 6,
        padding: "3px 9px", borderRadius: 999, fontSize: 11.5, lineHeight: 1.45,
        color: palette.fg, background: palette.bg, border: `1px solid ${palette.bd}`,
        fontWeight: tone === "bad" || tone === "warn" ? 600 : 500,
      }}
    >
      <span style={{ opacity: 0.85, fontWeight: 500 }}>{label}</span>
      <span style={{ fontFamily: "var(--htr-font-num)" }}>{value}</span>
    </span>
  );
}

function DataProvenanceStrip() {
  const data = (typeof window !== "undefined" && window.HTR_DATA) || {};
  const meta = data.meta || {};
  const src = resolveCandidateSource(meta);
  const health = resolvePipelineHealth(meta);
  const tone = HTR_HEALTH_TONE[health.status];
  const sections = fallbackSections(data.__degraded);

  return (
    <div
      data-htr-provenance-strip="1"
      role="status"
      style={{
        display: "flex", flexDirection: "column", gap: 6,
        padding: "8px 16px", borderBottom: "1px solid var(--htr-line)",
        background: "var(--htr-surface-2)",
      }}
    >
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
        {/* Source. The `sample` state is loud and unconditional — it does not
            soften because /api/dashboard returned 200. A successful HTTP call
            that served the sample fixture is exactly the case being warned
            about. */}
        <HtrProvenancePill
          tone={src.real ? "ok" : "bad"}
          label={src.real ? "数据来源" : "⚠ 数据来源"}
          value={src.real
            ? `${src.label} · ${src.source}`
            : `${src.label} — 非真实行情，不可据以下单`}
          title={`meta.candidatesSource = ${src.source}`}
        />
        <HtrProvenancePill tone="mute" label="交易日" value={src.tradeDate || "未知"} />
        {/* Health. `unknown` renders as a warning, never as silence and never
            as green: no answer and a good answer must not look the same. */}
        <HtrProvenancePill
          tone={tone.tone}
          label={health.status === "healthy" ? "管线" : "⚠ 管线"}
          value={health.status === "unknown"
            ? `${tone.label} — 后端未报告 Rule 15.10 健康态，不可视作正常`
            : `${tone.label} · ${health.status}`}
          title={health.summary || undefined}
        />
        <HtrProvenancePill tone="mute" label="健康 as-of" value={health.asof || "未知"} />
        <HtrProvenancePill tone="mute" label="面板刷新" value={meta.asof || "未知"} />
      </div>

      {/* Rule 15.10.7 — the aggregate is never shown without its components.
          Every entry, always expanded, never truncated. */}
      {health.components.length > 0 && (
        <ul data-htr-health-components="1" style={{ margin: 0, padding: "2px 0 0 2px", listStyle: "none", display: "flex", flexDirection: "column", gap: 3 }}>
          {health.components.map((c) => (
            <li key={c.code || c.component} style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: 6, fontSize: 11, lineHeight: 1.5, color: "var(--htr-ink-2)" }}>
              <span style={{ color: "var(--htr-warn)" }}>└</span>
              <b style={{ color: "var(--htr-ink)" }}>{c.label || c.component}</b>
              <code style={{ fontFamily: "var(--htr-font-num)", fontSize: 10.5, color: "var(--htr-ink-3)" }}>{c.code}</code>
              {c.detail && <span style={{ color: "var(--htr-ink-3)" }}>{c.detail}</span>}
              {/* Rule 15.10.5 — perishable degradations get their own marker,
                  because "we will catch it on the next run" is FALSE for them
                  and true for every other row in this list. */}
              {c.perishable && (
                <span
                  data-htr-perishable="1"
                  style={{
                    padding: "1px 7px", borderRadius: 4, fontSize: 10, fontWeight: 700,
                    color: "var(--htr-bear)", background: "var(--htr-bear-bg)",
                    border: "1px solid var(--htr-bear)",
                  }}
                >易逝 · 数据永久丢失，非延后重试</span>
              )}
            </li>
          ))}
        </ul>
      )}

      {/* Rule 11.9.4 — section-level mock fallback. Lived only in V3 before this;
          V1/V2/V4 rendered sample markets and news with no warning at all. */}
      {sections.length > 0 && (
        <div data-htr-mock-sections="1" style={{ fontSize: 11, lineHeight: 1.5, color: "var(--htr-warn)" }}>
          ⚠ 示例数据：<b>{sections.map((k) => HTR_SECTION_LABELS[k]).join("、")}</b> 暂未就绪，
          下方对应区块为占位/示例数据，非真实行情 (Rule 11.9.4)。
        </div>
      )}
    </div>
  );
}
