// htr-v3.jsx — redesigned V3 "市场温度仪表盘".
//
// IA fix for the overlap/squash problem (Rule 11.7):
//   • Document-flow scroll, no viewport lock — the whole dashboard scrolls.
//   • 3 rails with alignItems:start — each column takes its own natural height;
//     a tall center no longer stretches/crushes the short rails.
//   • CENTER stacks only Leader → K-line → Strategy (always-visible essentials).
//   • The 4 former accordion cards (factors / history / AI brief / debate) are
//     consolidated into ONE tabbed detail panel: a single body at natural height,
//     so expansion can never overlap a neighbor (the old failure mode).
//   • `layout` tweak ("tabs" | "stack") lets you compare the new consolidated
//     panel against the old stacked-accordion approach side by side.

const { useState: v3useState, useEffect: v3useEffect } = React;

function V3MarketDashboard({ layout = "tabs" }) {
  const data = window.HTR_DATA;
  const [selected, setSelected] = v3useState(() => {
    try { return localStorage.getItem("htr_symbol") || data.candidates[0].symbol; } catch (e) { return data.candidates[0].symbol; }
  });
  v3useEffect(() => { try { localStorage.setItem("htr_symbol", selected); } catch (e) {} }, [selected]);
  // selected may be an out-of-pool ticker the user searched (stashed in HTR_OOP).
  const resolved = data.candidates.find((c) => c.symbol === selected)
    || (window.HTR_OOP && window.HTR_OOP[selected]);
  const baseTop = resolved || data.candidates[0];
  // Code-review fix — HTR_OOP is memory-only; after a reload, a persisted
  // out-of-pool `htr_symbol` resolves to nothing and silently falls back to
  // candidates[0] while the picker highlights nothing. Re-sync `selected` so the
  // shown symbol and localStorage match (no sticky wrong-symbol on refresh).
  v3useEffect(() => {
    if (!resolved && selected !== data.candidates[0].symbol) setSelected(data.candidates[0].symbol);
  }, [selected, resolved]);
  // Overlay the authoritative per-symbol detail (/strategy /profile /outcomes
  // /kline /llm_brief /debate_brief) on top of the mock defaults. One wiring
  // point — every downstream detail card receives the enriched `top`.
  const top = useEnrichedCandidate(baseTop);
  const livePrice = useTickingPrice(top.price);

  return (
    <div className="htr v3-root" style={{ width: "100%", minHeight: "100%", background: "var(--htr-bg)", padding: "14px 18px 24px", display: "flex", flexDirection: "column", gap: 14 }}>
      <V3TopBar onSelect={setSelected} gates={data.gates} meta={data.meta} />
      {/* Rule 11.9.4 — flag any section that fell back to mock / is unavailable. */}
      {(() => {
        const deg = data.__degraded || {};
        const SEC = { offline: "全部区块（后端不可达）", markets: "市场温度", themes: "主题热力", newsTimeline: "新闻", decisionLog: "决策日志", candidates: "候选清单", gates: "门槛", positions: "持仓", dailyCockpit: "Daily Cockpit" };
        const bad = deg.offline ? ["offline"] : Object.keys(SEC).filter((k) => k !== "offline" && deg[k]);
        if (!bad.length) return null;
        return (
          <div style={{ padding: "7px 14px", background: "var(--htr-warn-bg)", border: "1px solid var(--htr-warn)", borderRadius: 6, fontSize: 11.5, color: "var(--htr-warn)", lineHeight: 1.5 }}>
            ⚠ 示例数据：<b>{bad.map((k) => SEC[k]).join("、")}</b> 暂未就绪，下方对应区块为占位/示例数据，非真实行情 (Rule 11.9.4)。
          </div>
        );
      })()}
      <V3TemperatureHero markets={data.markets} />

      {/* HERO BENTO (P24-07) — the command area. The deep-dive column (leader +
          chart + strategy) interlocks with the positions stack (portfolio + exit)
          at matched height, so the top band fills with no void (Rule 11.7.8).
          On mobile the positions stack is ordered before the chart (Rule 11.7.7). */}
      <div className="v3-hero-bento">
        <div className="v3-hero-main">
          <V3LeaderCard candidate={top} livePrice={livePrice} />
          <V3KLinePanel candidate={top} />
          <V3StrategyCard candidate={top} />
        </div>
        <div className="v3-hero-side">
          <V3PortfolioCard positions={data.positions} defaultSymbol={top.symbol} />
          {/* Rule 11.17 — Position Exit Discipline Board (shared governed card, four-variant parity). */}
          <ExitBoardCard />
        </div>
      </div>

      {/* PLAN — the Daily Action Board (Rule 11.16) full-width; mobile-ordered up
          near the top with the other decision surfaces (Rule 11.7.7). */}
      <div className="v3-plan-row">
        {/* Rule 11.16 — Daily Action Board (shared governed card, four-variant parity). */}
        <ActionBoardCard />
      </div>

      {/* WORKBENCH BENTO — navigation + research tiles below the command area:
          the candidate picker, the per-symbol detail tabs, and the theme/macro
          context, as three balanced tiles that fill the width. */}
      <div className="v3-work-bento">
        <div className="v3-work-col">
          <V3CandidatePicker candidates={data.candidates} selected={top.symbol} onSelect={setSelected} />
        </div>
        <div className="v3-work-col">
          {layout === "tabs"
            ? <V3DetailTabs candidate={top} />
            : <V3DetailStack candidate={top} />}
        </div>
        <div className="v3-work-col">
          <V3ThemeCard themes={data.themes} />
          <V3MacroNewsCard macro={data.macroNews} />
        </div>
      </div>

      {/* FEEDS ROW — review / context, full-width below so no rail towers over its
          siblings and leaves a dead void (Rule 11.7.8). On mobile these land last,
          after the decision surfaces + picker + deep-dive. */}
      <div className="v3-feeds-row">
        <V3CandidateHistoryCard />
        <EventDeskCard />
        <V3NewsCard items={data.newsTimeline} />
        <V3FeedsTabs cockpit={data.dailyCockpit} entries={data.decisionLog} tradeDate={data.meta.tradeDate} />
      </div>
    </div>
  );
}

// ───────────────────────── Card chrome ─────────────────────────
function CardHead({ title, sub, right, accent }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: "1px solid var(--htr-line)", background: accent ? "var(--htr-accent-soft)" : "var(--htr-surface-2)", borderRadius: "8px 8px 0 0", gap: 8, rowGap: 6, flexWrap: "wrap" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 700, whiteSpace: "nowrap" }}>{title}</div>
        {sub && <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", letterSpacing: "0.04em", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{sub}</div>}
      </div>
      {right && <div style={{ flexShrink: 0 }}>{right}</div>}
    </div>
  );
}

// ───────────────────────── Top bar ─────────────────────────
function V3TopBar({ onSelect, gates, meta }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,auto) minmax(0,1fr) minmax(0,auto)", alignItems: "center", gap: 16, minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 30, height: 30, borderRadius: 7, background: "var(--htr-accent)", color: "var(--htr-accent-ink)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 800, letterSpacing: "0.02em" }}>HTR</div>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.01em" }}>市场温度仪表盘</div>
          <div style={{ fontSize: 10, color: "var(--htr-ink-3)", letterSpacing: "0.08em" }}>HOTTHEMEROTATOR · MARKET TEMPERATURE · JST</div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, minWidth: 0, flexWrap: "wrap" }}>
        <V3TickerSearchBox onSelect={onSelect} />
        <WatchlistChip onSelectSymbol={onSelect} />
        {/* ProposalInboxChip / CalibrationChip / NotifierChip hoisted to the global
            app-nav (index.html) so they are reachable from every variant (P13-02 P2-B). */}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, justifyContent: "flex-end" }}>
        <span className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-3)", whiteSpace: "nowrap" }}>{meta.asof}</span>
        {/* Rule 11.9 — honest freshness/session label derived from asof vs tradeDate.
            On a non-trading day the panel says "休市 · 数据为 {tradeDate} 收盘" instead
            of implying a live quote (prices are static, no simulated ticking). */}
        {(() => {
          const fr = marketFreshness(meta);
          if (!fr) return null;
          return (
            <span className={"htr-chip" + (fr.closed ? " warn" : "")} title="价格为该交易日收盘价，非实时；本面板不模拟价格跳动 (Rule 11.9)">
              <span style={{ width: 6, height: 6, background: fr.closed ? "var(--htr-warn)" : "var(--htr-bull)", borderRadius: "50%" }} /> {fr.label}
            </span>
          );
        })()}
        {/* P3-C — shared CalibPill (was an inline duplicate of the htr-v1 component) */}
        <CalibPill sample={meta.calibration.sample} />
      </div>
    </div>
  );
}

function V3TickerSearchBox({ onSelect }) {
  const [input, setInput] = React.useState("");
  const [error, setError] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  // Analyze ANY ticker with market data, not only today's candidate pool. In-pool
  // tickers select directly; out-of-pool ones are validated via the real
  // /api/symbol/{T}/profile, then synthesized into a candidate (mock defaults via
  // HTR_enrich) and stashed in window.HTR_OOP so the dashboard can render them.
  // Per the candidate flow, an out-of-pool ticker is honestly labeled as such
  // (in_screener=False, "非今日热点候选"); the score stays uncalibrated.
  const submit = async (e) => {
    e.preventDefault();
    const raw = input.trim().toUpperCase();
    if (!raw) return;
    let t = /^\d{4}$/.test(raw) ? `${raw}.T` : raw;
    if (!/^\d{4}\.T$/.test(t)) { setError("格式无效 (4 位数字 + .T)"); return; }
    setError(null);
    if (window.HTR_DATA.candidates.some((c) => c.symbol === t)) { onSelect(t); setInput(""); return; }
    setLoading(true);
    const prof = await fetch(`/api/symbol/${encodeURIComponent(t)}/profile`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null)).catch(() => null);
    setLoading(false);
    if (!prof || !prof.latest_close) { setError(`${t} 无行情数据`); return; }  // 0/null/NaN all rejected
    const stub = window.HTR_enrich({
      symbol: t, price: prof.latest_close, chg: 0,
      nameJa: "", nameCn: "", theme: "", themeId: "", priority: "搜索", rank: 0,
      score: Math.round((prof.screener_score || 0) * 100), scoreStatus: prof.score_status,
      factors: prof,
      kfold: { status: prof.kfold_status, reason: prof.kfold_reason, oos: prof.kfold_oos_brier, inSample: prof.kfold_in_sample_brier },
      one_liner: prof.in_screener ? "用户搜索 · 今日在筛选池内" : "用户搜索 · 池外标的，非今日热点候选",
      reason: "", risk: "", dataQuality: "", out_of_pool: true,
    });
    window.HTR_OOP = window.HTR_OOP || {};
    window.HTR_OOP[t] = stub;
    onSelect(t); setInput("");
  };
  return (
    <form onSubmit={submit} style={{ position: "relative", display: "flex", alignItems: "center", gap: 5 }}>
      <input value={input} onChange={(e) => { setInput(e.target.value); if (error) setError(null); }} placeholder="搜索 ticker (例: 8035 或 8035.T)" spellCheck={false}
             style={{ fontSize: 12, padding: "5px 10px", width: 210, border: "1px solid var(--htr-line)", borderRadius: 6, background: "var(--htr-surface)", color: "var(--htr-ink)", fontFamily: "var(--htr-font-mono)" }} />
      <button type="submit" disabled={!input.trim() || loading} style={{ fontSize: 12, padding: "5px 12px", border: "1px solid var(--htr-accent)", background: "var(--htr-accent)", color: "var(--htr-accent-ink)", borderRadius: 6, cursor: "pointer", opacity: (input.trim() && !loading) ? 1 : 0.5 }}>{loading ? "分析中…" : "分析"}</button>
      {error && <div style={{ position: "absolute", top: "calc(100% + 5px)", left: 0, fontSize: 10.5, padding: "4px 9px", background: "var(--htr-bear-bg)", color: "var(--htr-bear)", border: "1px solid var(--htr-bear)", borderRadius: 5, whiteSpace: "nowrap", zIndex: 50 }}>{error}</div>}
    </form>
  );
}

// ───────────────────────── Temperature hero ─────────────────────────
function V3TemperatureHero({ markets }) {
  const legend = [["≥ 70 过热", "var(--htr-bear)"], ["50–69 升温", "var(--htr-warn)"], ["30–49 平温", "var(--htr-info)"], ["< 30 冷", "var(--htr-ink-3)"]];
  return (
    <div className="htr-card" style={{ padding: "14px 18px", display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <span style={{ fontSize: 15, fontWeight: 700 }}><Term k="市场温度">市场温度计</Term> · 今天市场什么感觉</span>
          <span style={{ fontSize: 11, color: "var(--htr-ink-3)" }}>温度 = 多因子合成（价格动量 · 量能 · 板块共振 · 隐含波动率）</span>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 10.5, color: "var(--htr-ink-3)", flexWrap: "wrap" }}>
          {legend.map(([t, c]) => <span key={t} style={{ display: "flex", alignItems: "center", gap: 5 }}><i style={{ width: 9, height: 9, background: c, borderRadius: 3 }} /> {t}</span>)}
        </div>
      </div>
      <div className="v3-temp-grid">
        {markets.map((m) => <V3MarketTile key={m.id} m={m} />)}
      </div>
    </div>
  );
}

const TEMP_GAUGE_GRAD = "linear-gradient(90deg,var(--htr-info) 0%,var(--htr-warn) 55%,var(--htr-heat-hot) 100%)";
function V3MarketTile({ m }) {
  const accent = HTR.heatColor(m.temp), bg = HTR.heatBg(m.temp);
  const hasSpark = Array.isArray(m.spark) && m.spark.length >= 2;
  const tempPct = Math.max(3, Math.min(100, Number(m.temp) || 0));
  return (
    <div style={{ background: bg, border: `1px solid ${accent}33`, borderRadius: 10, padding: "12px 14px", display: "flex", flexDirection: "column", gap: 9, position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 0, right: 0, padding: "3px 8px", background: accent, color: "var(--htr-accent-ink)", fontSize: 9, fontWeight: 700, borderBottomLeftRadius: 7, letterSpacing: "0.04em" }}>{m.state}</div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 700 }}>{m.label}</div>
        <div style={{ fontSize: 10, color: "var(--htr-ink-3)", marginTop: 1 }}>{m.sub}</div>
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between" }}>
        <div>
          <div className="htr-num" style={{ fontSize: 23, fontWeight: 800, color: accent, lineHeight: 1 }}>{m.temp}<span style={{ fontSize: 12, marginLeft: 1 }}>°</span></div>
          <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.08em", marginTop: 3 }}>温度</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="htr-num" style={{ fontSize: 14, fontWeight: 700 }}>{HTR.fmtPrice(m.price, m.id === "USDJPY" ? 2 : 0)}</div>
          <div className={"htr-num " + HTR.changeClass(m.chg)} style={{ fontSize: 11.5, fontWeight: 600 }}>{HTR.arrow(m.chg)} {HTR.fmtPct(m.chg)}</div>
        </div>
      </div>
      {/* Temperature gauge — visualizes the real 0-100 reading; fills the tile
          meaningfully whether or not an intraday trend exists. */}
      <div style={{ height: 6, borderRadius: 3, background: "var(--htr-surface-3)", overflow: "hidden", position: "relative" }}>
        <div style={{ position: "absolute", inset: 0, background: TEMP_GAUGE_GRAD, opacity: 0.22 }} />
        <div style={{ width: tempPct + "%", height: "100%", background: TEMP_GAUGE_GRAD, borderRadius: 3 }} />
      </div>
      {/* Intraday trend if we have it; otherwise an honest note, not an empty chart (Rule 11.9.4). */}
      {hasSpark
        ? <Sparkline data={m.spark} width={undefined} height={26} stroke={accent} fill={`color-mix(in oklab, ${accent} 16%, transparent)`} />
        : <div style={{ fontSize: 9.5, color: "var(--htr-ink-4)", fontFamily: "var(--htr-font-mono)", letterSpacing: "0.03em" }}>无盘中走势数据</div>}
    </div>
  );
}

// ───────────────────────── Leader card ─────────────────────────
function V3LeaderCard({ candidate, livePrice }) {
  const wl = useWatchlist();
  const watching = wl.has(candidate.symbol);
  const kf = candidate.kfold;
  const findTier = (k) => (candidate.ladder || []).find((r) => r.kind === k);
  return (
    <div className="htr-card" style={{ padding: "16px 20px", background: "linear-gradient(135deg, var(--htr-accent-soft) 0%, var(--htr-surface) 46%)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 14, flexWrap: "wrap" }}>
        <span className="htr-eyebrow" style={{ color: "var(--htr-accent)" }}>主题领涨 → 候选 LEADER</span>
        {candidate.theme && candidate.theme !== "screener_v2" && <span className="htr-chip accent">{candidate.theme}</span>}
        <CatalystBadges c={candidate} />
        <span className="htr-chip warn">{candidate.priority}</span>
        {kf && kf.status === "downgraded" && (
          <span className="htr-chip" title={kf.reason} style={{ borderColor: "var(--htr-bear)", color: "var(--htr-bear)", cursor: "help" }}>ⓘ K-fold 降级</span>
        )}
        <span style={{ flex: 1 }} />
        <button onClick={() => watching ? wl.remove(candidate.symbol) : wl.add(candidate.symbol)} className="htr-chip" style={{ cursor: "pointer", background: watching ? "var(--htr-warn-bg)" : "transparent", borderColor: watching ? "var(--htr-warn)" : "var(--htr-line-2)", color: watching ? "var(--htr-warn)" : "var(--htr-ink-2)" }}>{watching ? "★ 已关注" : "☆ 加关注"}</button>
      </div>
      <div className="v3-leader-grid">
        <div style={{ minWidth: 0 }}>
          <div className="htr-mono" style={{ fontSize: 30, fontWeight: 800, letterSpacing: "-0.02em" }}>{candidate.symbol}</div>
          <div style={{ fontSize: 12.5, color: "var(--htr-ink-3)", marginTop: 3 }}>{[candidate.nameJa, candidate.nameCn].filter(Boolean).join(" · ") || candidate.symbol}</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 12 }}>
            <span className="htr-num" style={{ fontSize: 40, fontWeight: 700, letterSpacing: "-0.01em" }}><AnimatedPrice value={livePrice} decimals={0} /></span>
            <span className={"htr-num " + HTR.changeClass(candidate.chg)} style={{ fontSize: 14, fontWeight: 700 }}>{candidate.chg == null ? "—" : <>{HTR.arrow(candidate.chg)} {HTR.fmtPct(candidate.chg)}</>}</span>
          </div>
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 14, lineHeight: 1.55, color: "var(--htr-ink)", marginBottom: 12 }}>{candidate.one_liner}</div>
          <div className="v3-ministat-grid">
            <MiniStat label={HTR.LABELS.scoreUncalibrated} value={candidate.score} accent />
            <MiniStat label="买入·均衡" value={HTR.fmtPrice(findTier("entry_balanced")?.price, 0)} />
            <MiniStat label="首止盈" value={HTR.fmtPrice(findTier("exit_1")?.price, 0)} up />
            <MiniStat label="止损参考" value={HTR.fmtPrice(findTier("stop")?.price, 0)} danger />
          </div>
        </div>
        {/* P4-E — V3LadderMini dropped (the 3rd ladder restatement); the 7 tiers
            live in the K-line overlay + the governed StrategyCard table, and the
            key entry/exit/stop tiers are surfaced in the ministats above. */}
      </div>
    </div>
  );
}

function MiniStat({ label, value, accent, danger, up }) {
  const c = danger ? "var(--htr-bear)" : up ? "var(--htr-bull)" : accent ? "var(--htr-accent)" : "var(--htr-ink)";
  return (
    <div style={{ padding: "7px 9px", background: "var(--htr-surface)", border: "1px solid var(--htr-line)", borderRadius: 6, borderLeft: `3px solid ${danger ? "var(--htr-bear)" : up ? "var(--htr-bull)" : accent ? "var(--htr-accent)" : "var(--htr-line-2)"}`, minWidth: 0 }}>
      <div className="htr-num" style={{ fontSize: 13.5, fontWeight: 700, whiteSpace: "nowrap", color: c }}>{value}</div>
      <div style={{ fontSize: 11, color: "var(--htr-ink-3)", letterSpacing: "0.02em", whiteSpace: "nowrap", marginTop: 2 }}>{label}</div>
    </div>
  );
}

// (V3LadderMini removed P13-03b — dead code since P4-E dropped it from LeaderCard;
//  the 7 tiers live in the K-line overlay + the governed StrategyCard table.)

// ───────────────────────── K-line panel ─────────────────────────
function V3KLinePanel({ candidate }) {
  const [expanded, setExpanded] = React.useState(true);
  const [boxRef, { width: boxW }] = useElementSize();
  const ladderForChart = (candidate.ladder || []).map((r) => ({ kind: r.kind, label: r.label, price: r.price }));
  const H = 300;
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <div onClick={() => setExpanded((e) => !e)} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: expanded ? "1px solid var(--htr-line)" : "none", background: "var(--htr-surface-2)", cursor: "pointer", userSelect: "none" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 700 }}>价格走势 · 七档阶梯 · {candidate.symbol}</span>
          <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>悬停看 OHLCV · MA20/60 · 量能 · 滚轮缩放</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {candidate._status && candidate._status.kline === "failed" && <span className="htr-chip warn" style={{ fontSize: 10 }} title="真实 K 线未就绪，下方为示例 (Rule 11.9.4)">示例K线</span>}
          <span className="htr-chip">{candidate.kline.length}D</span>
          <span style={{ fontSize: 13, color: "var(--htr-ink-3)" }}>{expanded ? "▾" : "▸"}</span>
        </div>
      </div>
      {expanded && (
        <div ref={boxRef} style={{ height: H, padding: 8 }}>
          {boxW > 0 && <KLineChart key={candidate.symbol} data={candidate.kline} ladder={ladderForChart} width={boxW - 16} height={H - 16} withVolume withMA with52wLines />}
        </div>
      )}
    </div>
  );
}

// ───────────────────────── Strategy card (Rule 11.6 governed) ─────────────────────────
function V3StrategyCard({ candidate }) {
  // Rule 11.6.9 — default-collapsed body (density); the safety triad
  // (banner + Rule 3 disclaimer + risk_warnings) stays visible outside the guard.
  const [expanded, setExpanded] = React.useState(false);
  const s = candidate.strategy;
  const sev = (v) => ({ block: "var(--htr-bear)", warn: "var(--htr-warn)", info: "var(--htr-ink-3)" }[v] || "var(--htr-ink-3)");
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <div onClick={() => setExpanded((e) => !e)} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: "1px solid var(--htr-line)", background: "var(--htr-surface-2)", cursor: "pointer", userSelect: "none" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 700 }}><Term>策略卡片</Term> · {candidate.symbol}</span>
          <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>ladder + Rule 12 风险标记 · advice-only</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="htr-chip">{s.risk_warnings.length} 风险</span>
          <span style={{ fontSize: 13, color: "var(--htr-ink-3)" }}>{expanded ? "▾" : "▸"}</span>
        </div>
      </div>
      {/* Rule 11.6.1 + 11.6.3 + 11.6.4 + 11.6.9 — NON-COLLAPSIBLE safety triad
          (advice-only banner + Rule 3 disclaimer + risk_warnings), always in the
          DOM even when the detail body is collapsed. */}
      <div style={{ padding: "12px 14px 14px" }}>
        <div style={{ padding: "8px 12px", marginBottom: 10, background: "var(--htr-warn-bg)", border: "1px solid var(--htr-warn)", borderRadius: 6, fontSize: 12, lineHeight: 1.5, color: "var(--htr-warn)", fontWeight: 600 }}>⚠ {s.banner}</div>
        <div style={{ marginBottom: 12, padding: "6px 10px", background: "var(--htr-surface)", border: "1px dashed var(--htr-line-2)", borderRadius: 6, fontSize: 10.5, color: "var(--htr-ink-3)", fontFamily: "var(--htr-font-mono)" }}>{s.rule_3_disclaimer}</div>
        {/* risk warnings (Rule 11.6.4 — always shown, even when empty, even collapsed) */}
        <div>
          <div className="htr-eyebrow" style={{ marginBottom: 6 }}>风险标记 (Rule 12)</div>
          {s.risk_warnings.length === 0 && <div style={{ fontSize: 12, color: "var(--htr-ink-3)" }}>当前无活跃风险标记。</div>}
          {s.risk_warnings.map((w, i) => (
            <div key={i} style={{ display: "flex", alignItems: "baseline", gap: 10, padding: "4px 0 4px 9px", fontSize: 12, lineHeight: 1.45, borderLeft: `2.5px solid ${sev(w.severity)}`, marginBottom: 3 }}>
              <span className="htr-mono" style={{ fontSize: 10.5, color: sev(w.severity), fontWeight: 700, minWidth: 46 }}>{w.severity.toUpperCase()}</span>
              <span style={{ color: "var(--htr-ink)", flex: 1 }}>{w.message}</span>
              <span className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>{w.rule_ref}</span>
            </div>
          ))}
        </div>
        {/* Compact key levels — the three decisive price references, visible even
            when collapsed, so the card carries real numbers instead of slack.
            Rule 11.6.6 noun labels only; the full 7-tier ladder is the expand. */}
        {(() => {
          const find = (k) => (s.ladder_tiers || []).find((t) => t.kind === k);
          const entry = find("entry_balanced"), stop = find("stop"), tp1 = find("exit_1");
          if (!entry && !stop && !tp1) return null;
          const cell = (lbl, t, color) => (
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="htr-eyebrow" style={{ fontSize: 9, marginBottom: 2 }}>{lbl}</div>
              <div className="htr-num" style={{ fontSize: 15, fontWeight: 700, color }}>{t ? HTR.fmtPrice(t.price, 0) : "—"}</div>
              {t && <div className="htr-num" style={{ fontSize: 9.5, color: "var(--htr-ink-3)" }}>{t.pct_vs_ref > 0 ? "+" : ""}{t.pct_vs_ref.toFixed(1)}%</div>}
            </div>
          );
          return (
            <div style={{ display: "flex", gap: 10, marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--htr-line)" }}>
              {cell("建议价位·均衡", entry, "var(--htr-info)")}
              {cell("止损参考", stop, "var(--htr-bear)")}
              {cell("目标参考·首", tp1, "var(--htr-bull)")}
            </div>
          );
        })()}
      </div>
      {expanded && (
        <div style={{ padding: "0 14px 14px" }}>
          {/* 7-tier ladder (Rule 9.6 — all 7 visible; Rule 11.6.6 — noun labels) */}
          <div style={{ marginBottom: 12 }}>
            <div className="htr-eyebrow" style={{ marginBottom: 6 }}>价格阶梯 · 当前 <span className="htr-mono" style={{ textTransform: "none" }}>{HTR.fmtPrice(s.current_price, 0)}</span> · 标签为建议价位/目标参考</div>
            <div style={{ borderTop: "1px solid var(--htr-line)" }}>
              {s.ladder_tiers.map((t) => {
                const c = ladderColor(t.kind);
                return (
                  <div key={t.kind} style={{ display: "grid", gridTemplateColumns: "1fr 92px 72px", fontSize: 12.5, lineHeight: 1.6, borderBottom: "1px solid var(--htr-line-soft)", alignItems: "center" }}>
                    <div style={{ padding: "5px 0", color: "var(--htr-ink-2)", display: "flex", alignItems: "center", gap: 7 }}><span style={{ width: 7, height: 7, borderRadius: 2, background: c }} />{t.label}</div>
                    <div className="htr-num" style={{ padding: "5px 0", textAlign: "right", fontWeight: 600 }}>{HTR.fmtPrice(t.price, 0)}</div>
                    <div className="htr-num" style={{ padding: "5px 0", textAlign: "right", color: t.pct_vs_ref >= 0 ? "var(--htr-bull)" : "var(--htr-bear)" }}>{t.pct_vs_ref > 0 ? "+" : ""}{t.pct_vs_ref.toFixed(2)}%</div>
                  </div>
                );
              })}
            </div>
          </div>
          {/* catalyst calendar + Rule 11.6.7 disclaimer */}
          {s.catalyst_calendar.length > 0 && (
            <div>
              <div className="htr-eyebrow" style={{ marginBottom: 6 }}>催化日历</div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, lineHeight: 1.6, color: "var(--htr-ink-2)" }}>{s.catalyst_calendar.map((c, i) => <li key={i}>{c.label}</li>)}</ul>
              <div style={{ fontSize: 10, color: "var(--htr-ink-3)", fontStyle: "italic", marginTop: 4 }}>{s.catalyst_disclaimer}</div>
            </div>
          )}
          <div style={{ marginTop: 12, fontSize: 10, color: "var(--htr-ink-3)", fontFamily: "var(--htr-font-mono)" }}>score_status=<span style={{ color: "var(--htr-ink-2)" }}>{s.score_status}</span></div>
        </div>
      )}
    </div>
  );
}

// ───────────────────────── Detail tabs (the IA consolidation) ─────────────────────────
function V3DetailTabs({ candidate }) {
  const [tab, setTab] = React.useState("factor");
  const tabs = [
    { id: "factor", label: "因子构成" },
    { id: "outcomes", label: "历史命中" },
    { id: "ai", label: "AI 综合" },
    { id: "debate", label: "Bull / Bear / Judge" },
  ];
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 2, padding: "0 6px", borderBottom: "1px solid var(--htr-line)", background: "var(--htr-surface-2)", overflowX: "auto" }}>
        {tabs.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{ padding: "11px 14px", fontSize: 12.5, background: "transparent", border: "none", borderBottom: tab === t.id ? "2px solid var(--htr-accent)" : "2px solid transparent", color: tab === t.id ? "var(--htr-accent)" : "var(--htr-ink-2)", cursor: "pointer", fontWeight: tab === t.id ? 700 : 500, fontFamily: "inherit", whiteSpace: "nowrap" }}>{t.label}</button>
        ))}
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 10, color: "var(--htr-ink-3)", padding: "0 10px", whiteSpace: "nowrap" }}>展开向下推 · 不遮挡</span>
      </div>
      <div style={{ padding: "14px 16px" }}>
        {tab === "factor" && <FactorBody candidate={candidate} />}
        {tab === "outcomes" && <OutcomesBody candidate={candidate} />}
        {tab === "ai" && <AiBody candidate={candidate} />}
        {tab === "debate" && <DebateBody candidate={candidate} />}
      </div>
    </div>
  );
}

// Alternate layout: the OLD stacked-accordion approach (for comparison via tweak)
function V3DetailStack({ candidate }) {
  const Section = ({ title, sub, children, def = false }) => {
    const [open, setOpen] = React.useState(def);
    return (
      <div className="htr-card" style={{ overflow: "hidden" }}>
        <div onClick={() => setOpen((o) => !o)} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: open ? "1px solid var(--htr-line)" : "none", background: "var(--htr-surface-2)", cursor: "pointer", userSelect: "none" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}><span style={{ fontSize: 13, fontWeight: 700 }}>{title}</span><span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>{sub}</span></div>
          <span style={{ fontSize: 13, color: "var(--htr-ink-3)" }}>{open ? "▾" : "▸"}</span>
        </div>
        {open && <div style={{ padding: "14px 16px" }}>{children}</div>}
      </div>
    );
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Section title={<><Term>因子构成</Term> + 历史命中</>} sub="Rule 8.6.1 sample frequency · NOT win rate" def><FactorBody candidate={candidate} /><div style={{ height: 14 }} /><OutcomesBody candidate={candidate} /></Section>
      <Section title={<><Term>AI 综合</Term> · {candidate.symbol}</>} sub="LLM Per-Ticker Brief"><AiBody candidate={candidate} /></Section>
      <Section title={`Bull / Bear / Judge · ${candidate.symbol}`} sub="multi-leg debate · Rule 8.3.1 per-leg gate"><DebateBody candidate={candidate} /></Section>
    </div>
  );
}

function FactorBody({ candidate }) {
  const st = candidate._status && candidate._status.profile;
  const f = candidate.factors;
  // A signed magnitude bar (centre = 0) that visualises the real factor value —
  // NOT a probability (this tab is "排序信号 · 非概率"). bar in [-1, 1].
  const Row = ({ label, value, bar }) => (
    <div style={{ padding: "4px 0", borderBottom: "1px solid var(--htr-line-soft)", fontSize: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span style={{ color: "var(--htr-ink-3)" }}>{label}</span><span className="htr-mono" style={{ color: "var(--htr-ink)" }}>{value}</span>
      </div>
      {bar != null && (
        <div style={{ height: 3, marginTop: 4, background: "var(--htr-surface-3)", borderRadius: 2, position: "relative" }}>
          <div style={{ position: "absolute", left: "50%", top: -1, width: 1, height: 5, background: "var(--htr-line-2)" }} />
          <div style={{ position: "absolute", left: bar >= 0 ? "50%" : `${50 + bar * 50}%`, width: `${Math.abs(bar) * 50}%`, height: "100%", background: bar >= 0 ? "var(--htr-bull)" : "var(--htr-bear)", borderRadius: 2 }} />
        </div>
      )}
    </div>
  );
  if (st === "pending") return <AsyncBodyState status="pending" />;
  if (!f) return <AsyncBodyState status="failed" />;
  // Null/gap-safe formatters — the real /api/symbol/{T}/profile does NOT carry
  // every field the mock did (e.g. sortino_60 / vol_z); show "—" honestly
  // rather than crash or fabricate. (Backend gap tracked in PROJECT_STATUS.)
  const num = (x, d) => (x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d));
  const pct = (x, d) => (x == null || Number.isNaN(Number(x)) ? "—" : `${(Number(x) * 100).toFixed(d)}%`);
  const bar = (x, scale) => (x == null || Number.isNaN(Number(x)) ? null : Math.max(-1, Math.min(1, Number(x) / scale)));
  return (
    <div>
      <AsyncBodyState status={st} />
      <div className="htr-eyebrow" style={{ marginBottom: 8 }}>分数构成 · 排序信号（非概率）</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 24px" }}>
        <Row label="screener_score" value={num(f.screener_score, 3)} bar={bar(f.screener_score, 1)} />
        <Row label="mom_20" value={pct(f.mom_20, 1)} bar={bar(f.mom_20, 0.5)} />
        <Row label="mom_60" value={pct(f.mom_60, 1)} bar={bar(f.mom_60, 0.5)} />
        <Row label="sharpe_20" value={num(f.sharpe_20, 2)} bar={bar(f.sharpe_20, 3)} />
        <Row label="sortino_60" value={num(f.sortino_60, 2)} bar={bar(f.sortino_60, 3)} />
        <Row label="vol_z" value={num(f.vol_z, 2)} bar={bar(f.vol_z, 3)} />
        <Row label="adv (10⁶ ¥)" value={f.adv == null ? "—" : (f.adv / 1e6).toFixed(0)} />
        <Row label="in_screener" value={f.in_screener ? "✓" : "—"} />
      </div>
      <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginTop: 8, fontStyle: "italic" }}>{HTR.LABELS.rule94Note}</div>
    </div>
  );
}

function OutcomesBody({ candidate }) {
  const st = candidate._status && candidate._status.outcomes;
  if (st === "pending") return <AsyncBodyState status="pending" />;
  const o = candidate.outcomes;
  if (!o) return <AsyncBodyState status="failed" />;
  return (
    <div>
      <AsyncBodyState status={st} />
      <div className="htr-eyebrow" style={{ marginBottom: 8 }}>历史预测命中（{o.horizon}）· 原始频率，非胜率</div>
      <div style={{ display: "flex", gap: 18, fontSize: 12.5, fontFamily: "var(--htr-font-mono)", marginBottom: 10, flexWrap: "wrap" }}>
        <span>总数 <b>{o.count}</b></span><span>已结算 <b>{o.n_complete}</b></span><span>命中 <b>{o.n_hit}</b></span>
        <span>原始频率 <b>{o.hit_rate != null ? `${(o.hit_rate * 100).toFixed(1)}%` : "—"}</b></span>
        <span className="htr-chip warn" style={{ fontFamily: "var(--htr-font-body)" }}>uncalibrated_evidence</span>
      </div>
      <div style={{ maxHeight: 600, overflow: "auto", fontSize: 11.5, fontFamily: "var(--htr-font-mono)" }}>
        <div style={{ display: "grid", gridTemplateColumns: "110px 70px 80px 80px", gap: 8, padding: "3px 0", color: "var(--htr-ink-3)", borderBottom: "1px solid var(--htr-line)", position: "sticky", top: 0, background: "var(--htr-surface)" }}>
          <span>trade_date</span><span>raw_score</span><span>结果</span><span>来源</span>
        </div>
        {o.items.map((r, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "110px 70px 80px 80px", gap: 8, padding: "3px 0", borderBottom: "1px solid var(--htr-line-soft)" }}>
            <span>{r.trade_date}</span><span>{r.raw_score.toFixed(3)}</span>
            <span style={{ color: r.outcome === 1 ? "var(--htr-bull)" : r.outcome === 0 ? "var(--htr-bear)" : "var(--htr-ink-3)", fontWeight: 700 }}>{r.outcome === 1 ? "命中" : r.outcome === 0 ? "未命中" : "待结算"}</span>
            <span style={{ color: "var(--htr-ink-3)", fontSize: 10.5 }}>{r.backdated ? "bootstrap" : "live"}</span>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginTop: 8, fontStyle: "italic" }}>{o.rule_9_4_note}</div>
    </div>
  );
}

function AiBody({ candidate }) {
  const st = candidate._status && candidate._status.aiBrief;
  if (st === "pending") return <AsyncBodyState status="pending" />;
  const b = candidate.aiBrief;
  if (!b) return <AsyncBodyState status="failed" />;
  return (
    <div style={{ fontSize: 13, lineHeight: 1.65 }}>
      <AsyncBodyState status={st} />
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span className="htr-chip accent">{b.model_version}</span>
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>LLM 叙事综合 · Rule 8.3.1 守门通过（无胜率/概率/%）</span>
      </div>
      <div style={{ color: "var(--htr-ink)", whiteSpace: "pre-wrap" }}>{b.narrative}</div>
      <details style={{ marginTop: 12 }}>
        <summary style={{ fontSize: 11, color: "var(--htr-ink-3)", cursor: "pointer" }}>事实证据（{b.factual_grounding.length} 条）· 生成 {b.generation_ts}</summary>
        <div style={{ marginTop: 8, padding: "8px 12px", background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", borderRadius: 6, fontFamily: "var(--htr-font-mono)", fontSize: 11, color: "var(--htr-ink-2)" }}>
          {b.factual_grounding.map((l, i) => <div key={i} style={{ padding: "1px 0" }}>{l}</div>)}
        </div>
      </details>
    </div>
  );
}

function DebateBody({ candidate }) {
  const st = candidate._status && candidate._status.debate;
  if (st === "pending") return <AsyncBodyState status="pending" />;
  const d = candidate.debate;
  if (!d) return <AsyncBodyState status="failed" />;
  const Leg = ({ leg, color, label }) => (
    <div style={{ padding: "10px 12px", background: "var(--htr-surface)", border: `1px solid ${color}`, borderRadius: 8, minWidth: 0 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 12, lineHeight: 1.55, color: "var(--htr-ink)" }}>{leg.narrative}</div>
    </div>
  );
  return (
    <div>
      <AsyncBodyState status={st} />
      <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginBottom: 10 }}>每条 leg 独立通过 Rule 8.3.1 regex 检查 · {d.model_version}</div>
      <div className="v3-debate-grid">
        <Leg leg={d.bull} color="var(--htr-bull)" label="🐂 Bull" />
        <Leg leg={d.bear} color="var(--htr-bear)" label="🐻 Bear" />
        <Leg leg={d.judge} color="var(--htr-accent)" label="⚖ Judge" />
      </div>
    </div>
  );
}

// ───────────────────────── Left rail ─────────────────────────
function V3CandidatePicker({ candidates, selected, onSelect }) {
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title="候选清单" sub={`screener_v2 · 点击切换 · 当前 ${selected}`} right={<span className="htr-chip">{candidates.length}</span>} />
      <div style={{ maxHeight: 360, overflow: "auto" }}>
        {candidates.map((c) => {
          const active = c.symbol === selected;
          return (
            <div key={c.symbol} onClick={() => onSelect(c.symbol)} role="button" tabIndex={0}
                 onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(c.symbol); } }}
                 style={{ display: "grid", gridTemplateColumns: "26px 1fr auto", gap: 8, alignItems: "center", padding: "9px 12px", borderBottom: "1px solid var(--htr-line-soft)", borderLeft: active ? "3px solid var(--htr-accent)" : "3px solid transparent", background: active ? "var(--htr-accent-soft)" : "transparent", cursor: "pointer", userSelect: "none" }}>
              <span className="htr-mono" style={{ fontSize: 11, color: "var(--htr-ink-3)" }}>#{c.rank}</span>
              <div style={{ minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                  <span className="htr-mono" style={{ fontSize: 13, fontWeight: 700 }}>{c.symbol}</span>
                  <span className={"htr-num " + HTR.changeClass(c.chg)} style={{ fontSize: 11 }}>{HTR.fmtPct(c.chg)}</span>
                </div>
                <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{[c.nameCn, c.theme === "screener_v2" ? null : c.theme].filter(Boolean).join(" · ") || c.symbol}</div>
                {c.newsCatalyzed && <div style={{ display: "flex", gap: 4, marginTop: 3, flexWrap: "wrap" }}><CatalystBadges c={c} size="sm" /></div>}
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="htr-num" style={{ fontSize: 14, fontWeight: 700 }}>{c.score}</div>
                <div style={{ width: 44, height: 4, background: "var(--htr-line-soft)", borderRadius: 2, overflow: "hidden", marginTop: 3 }}><div style={{ width: `${c.score}%`, height: "100%", background: HTR.heatColor(c.score) }} /></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// P10-26 slice 3 — surface the HTR-native macro/policy/cross-market news heat that
// the per-ticker feed never captured. Renders nothing until the overlay is built.
function V3MacroNewsCard({ macro }) {
  const [exp, setExp] = React.useState({});   // per-category expand (Rule 11.1)
  if (!macro || !macro.macro) return null;
  const CAT = { monetary: "货币政策", fiscal: "财政政策", fx: "汇率", trade: "贸易", overseas: "海外/跨市场" };
  const cats = Object.keys(CAT)
    .map((k) => ({ k, label: CAT[k], n: ((macro.macro[k] || {}).news_count) || 0, recent: (macro.macro[k] || {}).recent || [] }))
    .filter((c) => c.n > 0);
  if (!cats.length) return null;
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title="宏观新闻热度" sub={`${macro.total_fetched || 0} 条 · 政策/汇率/跨市场`} right={<span className="htr-chip">P10-26</span>} />
      <div style={{ padding: "10px 14px" }}>
        {cats.map((c) => {
          const isExp = !!exp[c.k];
          const shown = isExp ? c.recent.slice(0, 10) : c.recent.slice(0, 3);  // 单击最多展开 10 条
          const more = c.recent.length - 3;
          const link = (it, i) => (
            <a key={i} href={it.url || "#"} target={it.url ? "_blank" : undefined} rel="noopener noreferrer" title={it.title}
               style={{ display: "block", fontSize: 10.5, color: "var(--htr-ink-2)", textDecoration: "none", lineHeight: 1.45, marginTop: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>· {it.title}</a>
          );
          return (
            <div key={c.k} style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5 }}>
                <span style={{ fontWeight: 600 }}>{c.label}</span>
                <span className="htr-mono" style={{ color: "var(--htr-accent)" }}>{c.n} 条</span>
              </div>
              {/* Rule 11.7.3 — news is an unbounded feed: when expanded, scroll inside the card. */}
              <div style={isExp ? { maxHeight: 132, overflowY: "auto" } : undefined}>{shown.map(link)}</div>
              {!isExp && more > 0 && (
                <div onClick={() => setExp((e) => ({ ...e, [c.k]: true }))} title={`展开剩余 ${more} 条`}
                     style={{ cursor: "pointer", userSelect: "none", fontSize: 11, color: "var(--htr-accent)", marginTop: 2 }}>⋯ 展开（还有 {more} 条）</div>
              )}
              {isExp && (
                <div onClick={() => setExp((e) => ({ ...e, [c.k]: false }))}
                     style={{ cursor: "pointer", userSelect: "none", fontSize: 11, color: "var(--htr-ink-3)", marginTop: 2 }}>▴ 收起</div>
              )}
            </div>
          );
        })}
        <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", fontStyle: "italic", marginTop: 4 }}>HTR-native 宏观采集 · 透明关键词分类 (Rule 8.3)</div>
      </div>
    </div>
  );
}

function V3ThemeCard({ themes }) {
  const total = themes.reduce((a, b) => a + b.heat, 0);
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title={<Term>主题热力</Term>} sub="Theme Heat · 六板块" />
      <div style={{ padding: "12px 14px 6px", display: "flex", flexWrap: "wrap", gap: 6 }}>
        {themes.map((t) => {
          const w = Math.max(28, (t.heat / total) * 100), c = HTR.heatColor(t.heat), bg = HTR.heatBg(t.heat);
          return (
            <div key={t.id} style={{ flexBasis: `calc(${w}% - 6px)`, flexGrow: 1, background: bg, border: `1px solid ${c}33`, borderRadius: 7, padding: "10px 11px 9px", display: "flex", flexDirection: "column", gap: 4, minHeight: t.heat >= 70 ? 84 : 64 }}>
              <div style={{ fontSize: 12, fontWeight: 700 }}>{t.label}</div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                <span className="htr-num" style={{ fontSize: 19, fontWeight: 800, color: c, lineHeight: 1 }}>{t.heat}°</span>
                <span className={"htr-num " + HTR.changeClass(t.mom)} style={{ fontSize: 10.5 }}>{HTR.arrow(t.mom)} {Math.abs(t.mom).toFixed(1)}</span>
              </div>
              <div style={{ fontSize: 11, color: "var(--htr-ink-3)" }}>{t.leaders.slice(0, 2).join(" · ")}</div>
            </div>
          );
        })}
      </div>
      <hr className="htr-divider" />
      <div style={{ padding: "12px 14px" }}>
        <div className="htr-eyebrow" style={{ marginBottom: 8 }}>主题排序</div>
        <ThemeHeatBars themes={themes} />
      </div>
    </div>
  );
}

// ───────────────────────── Right rail ─────────────────────────
function V3PortfolioCard({ positions, defaultSymbol }) {
  const [modal, setModal] = React.useState(null);
  if (!positions || !positions.available) return <div className="htr-card" style={{ padding: "14px 16px" }}><div className="htr-eyebrow">持仓 · Portfolio</div><div style={{ fontSize: 12, color: "var(--htr-ink-3)", marginTop: 4 }}>持仓数据未就绪</div></div>;
  const fmtY = (v) => "¥" + Math.round(Number(v)).toLocaleString("en-US");
  const holdings = positions.holdings || [];
  const posValue = holdings.reduce((a, h) => a + (h.market_value || h.qty * h.market_price), 0);
  const totalPnl = holdings.reduce((a, h) => a + (h.unrealized_pnl || 0), 0);
  const totalCost = posValue - totalPnl;
  const totalPnlPct = totalCost > 0 ? (totalPnl / totalCost) * 100 : 0;
  const maxAbsPct = Math.max(1, ...holdings.map((h) => Math.abs(h.unrealized_return_pct || 0)));
  const posPct = positions.nav > 0 ? (posValue / positions.nav) * 100 : 0;
  const up = totalPnl >= 0, pc = up ? "var(--htr-bull)" : "var(--htr-bear)";
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title="持仓 · Portfolio" sub={`${positions.strategy_id} · ${holdings.length} 标的`} right={
        <span style={{ display: "flex", gap: 5 }}>
          <button className="htr-chip accent" style={{ cursor: "pointer" }} onClick={() => setModal("fill")} title="录入已在外部券商完成的交易 (Rule 3 carve-out)">+ 成交</button>
          <button className="htr-chip" style={{ cursor: "pointer" }} onClick={() => setModal("cash")} title="录入现金事件">现金</button>
        </span>
      } />
      {modal && <ManualEntryModal kind={modal} defaultSymbol={modal === "fill" ? defaultSymbol : ""} onClose={() => setModal(null)} />}

      {/* summary: NAV + total unrealized PnL + NAV trend sparkline */}
      <div style={{ padding: "12px 14px 6px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, alignItems: "end" }}>
          <div>
            <div className="htr-eyebrow">NAV · 总资产</div>
            <div className="htr-num" style={{ fontSize: 19, fontWeight: 700 }}>{fmtY(positions.nav)}</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="htr-eyebrow">浮动盈亏</div>
            <div className="htr-num" style={{ fontSize: 16, fontWeight: 700, color: pc }}>{up ? "+" : ""}{fmtY(totalPnl)}</div>
            <div className="htr-num" style={{ fontSize: 11, fontWeight: 700, color: pc }}>{up ? "+" : ""}{totalPnlPct.toFixed(2)}%</div>
          </div>
        </div>
        {positions.navHistory && positions.navHistory.length > 1 && (
          <div style={{ marginTop: 8 }}>
            <Sparkline data={positions.navHistory} width={undefined} height={34} area />
          </div>
        )}
        {/* allocation: positions vs cash */}
        <div style={{ marginTop: 8 }}>
          <div style={{ display: "flex", height: 8, borderRadius: 5, overflow: "hidden", background: "var(--htr-line-soft)" }}>
            <div style={{ width: `${posPct}%`, background: "var(--htr-accent)" }} title={`持仓 ${posPct.toFixed(0)}%`} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--htr-ink-3)", marginTop: 4 }}>
            <span><i style={{ display: "inline-block", width: 7, height: 7, borderRadius: 2, background: "var(--htr-accent)", marginRight: 4 }} />持仓 {fmtY(posValue)} · {posPct.toFixed(0)}%</span>
            <span><i style={{ display: "inline-block", width: 7, height: 7, borderRadius: 2, background: "var(--htr-line-2)", marginRight: 4 }} />现金 {fmtY(positions.cash)} · {(100 - posPct).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* holdings — bounded scroll so 1 or 20 positions both fit */}
      <div style={{ display: "flex", justifyContent: "space-between", padding: "7px 14px", borderTop: "1px solid var(--htr-line)", borderBottom: "1px solid var(--htr-line)", background: "var(--htr-surface-2)", fontSize: 11, color: "var(--htr-ink-3)", letterSpacing: "0.04em" }}>
        <span>标的</span><span>市值 · 浮盈</span>
      </div>
      <div style={{ maxHeight: 224, overflow: "auto" }}>
        {holdings.length === 0 && <div style={{ padding: "12px 14px", fontSize: 12, color: "var(--htr-ink-3)" }}>当前无持仓（仅现金）</div>}
        {holdings.map((h) => {
          const hu = h.unrealized_pnl >= 0, hc = hu ? "var(--htr-bull)" : "var(--htr-bear)";
          const mv = h.market_value || h.qty * h.market_price;
          const barW = Math.min(50, (Math.abs(h.unrealized_return_pct || 0) / maxAbsPct) * 50);
          return (
            <div key={h.symbol} style={{ padding: "8px 14px", borderBottom: "1px solid var(--htr-line-soft)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="htr-mono" style={{ fontSize: 12.5, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{h.symbol}</div>
                  <div style={{ fontSize: 11, color: "var(--htr-ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{h.qty} 股 @ ¥{h.avg_cost.toLocaleString("en-US")}</div>
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div className="htr-num" style={{ fontSize: 12.5, fontWeight: 600 }}>{fmtY(mv)}</div>
                  <div className="htr-num" style={{ fontSize: 10.5, fontWeight: 700, color: hc }}>{hu ? "+" : ""}{Math.round(h.unrealized_pnl).toLocaleString("en-US")} ({hu ? "+" : ""}{h.unrealized_return_pct.toFixed(1)}%)</div>
                </div>
              </div>
              {/* diverging PnL bar — green right / red left from a center baseline */}
              <div style={{ position: "relative", height: 5, marginTop: 6, background: "var(--htr-line-soft)", borderRadius: 3 }}>
                <div style={{ position: "absolute", left: "50%", top: -1, bottom: -1, width: 1, background: "var(--htr-line-2)" }} />
                <div style={{ position: "absolute", top: 0, bottom: 0, borderRadius: 3, background: hc, left: hu ? "50%" : `${50 - barW}%`, width: `${barW}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function V3NewsCard({ items }) {
  // Rule 11.9.4 — derive the high-severity count from real items.weight, not "2 high".
  const hi = (items || []).filter((n) => n.weight === "high").length;
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title={<Term>新闻催化</Term>} sub="近 6h" right={hi > 0 ? <span className="htr-chip warn">{hi} high</span> : null} />
      <div style={{ padding: "4px 14px", maxHeight: 320, overflow: "auto" }}><NewsTimeline items={items} max={7} compact /></div>
    </div>
  );
}

// P4-F — tab the two governance feeds (Cockpit | 决策日志) into one card so the
// right rail isn't 4 always-open cards. Reuses the existing card components.
function V3FeedsTabs({ cockpit, entries, tradeDate }) {
  const [tab, setTab] = React.useState("cockpit");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", gap: 6 }}>
        {[["cockpit", "Daily Cockpit"], ["log", "决策日志"]].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={{ cursor: "pointer", fontFamily: "inherit", fontSize: 11.5, fontWeight: tab === id ? 700 : 500, padding: "4px 11px", borderRadius: 999, border: "1px solid " + (tab === id ? "var(--htr-accent)" : "var(--htr-line-2)"), background: tab === id ? "var(--htr-accent)" : "transparent", color: tab === id ? "var(--htr-accent-ink)" : "var(--htr-ink-2)" }}>{label}</button>
        ))}
      </div>
      {tab === "cockpit" ? <V3CockpitCard cockpit={cockpit} /> : <V3DecisionLogCard entries={entries} tradeDate={tradeDate} />}
    </div>
  );
}

function V3CockpitCard({ cockpit }) {
  if (!cockpit) return null;
  const s = cockpit.summary, blocked = cockpit.notificationsInvoked;
  const Metric = ({ label, value }) => <div style={{ background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", borderRadius: 6, padding: "7px 9px" }}><div className="htr-num" style={{ fontSize: 17, fontWeight: 800 }}>{value}</div><div style={{ fontSize: 11, color: "var(--htr-ink-3)", letterSpacing: "0.04em" }}>{label}</div></div>;
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title={<Term>Daily Cockpit</Term>} sub={`${cockpit.activationStage} · Rule 12.0`} right={<span className={blocked ? "htr-chip bear" : "htr-chip bull"}>{blocked ? "blocked" : "pull-only"}</span>} />
      <div style={{ padding: "10px 14px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 7, marginBottom: 8 }}>
          <Metric label="watch" value={s.watchlistCount} /><Metric label="quotes" value={s.symbolsWithQuotes} /><Metric label="silent" value={s.silentQueueCount} />
        </div>
        {cockpit.watchlist.slice(0, 4).map((r) => (
          <div key={r.symbol} style={{ display: "grid", gridTemplateColumns: "62px 1fr auto", gap: 6, alignItems: "center", fontSize: 11.5, padding: "3px 0" }}>
            <span className="htr-mono" style={{ fontWeight: 700 }}>{r.symbol}</span>
            <span style={{ color: "var(--htr-ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.quoteStatus}</span>
            <span className="htr-chip warn" style={{ fontSize: 10.5 }}>TDnet {r.tdnetCount}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function V3DecisionLogCard({ entries, tradeDate }) {
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title={<span><Term>决策日志</Term> · <Term>§8.6</Term></span>} sub={tradeDate} />
      <div style={{ maxHeight: 300, overflow: "auto" }}><DecisionLog entries={entries} max={8} /></div>
    </div>
  );
}

// ───────────────────────── Candidate cohort review (Rule 11.11) ─────────────────────────
// Read-only replay of the append-only decision log: a past day's whole candidate
// roster (PIT, as flagged at cutoff) + its equal-weight realized outcome vs the
// same-window TOPIX benchmark. Cohort-first (11.11.2), backdated walled off by
// default (11.11.5), no win-rate labeling (11.11.6). Self-fetches /api/candidates/history.
// Event Desk (P16-E4 / Rule 11.13) — pick an event/theme → see WHO is exposed and
// how far each name has ALREADY moved (priced-in read). Read-only, descriptive: no
// event-outcome prediction, no probability/win-rate. Shared → bound into all variants.
function EventDeskCard() {
  const [themes, setThemes] = v3useState([]);
  const [events, setEvents] = v3useState([]);
  const [sel, setSel] = v3useState("semi");
  const [data, setData] = v3useState(null);
  const [err, setErr] = v3useState(false);
  const [loading, setLoading] = v3useState(false);

  v3useEffect(() => {
    fetch("/api/event-desk/themes", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d) { setThemes(d.themes || []); setEvents(d.events || []); } })
      .catch(() => {});
  }, []);

  v3useEffect(() => {
    setLoading(true);
    fetch("/api/event-desk?event=" + encodeURIComponent(sel), { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d) { setData(d); setErr(false); } else setErr(true); })
      .catch(() => setErr(true))
      .finally(() => setLoading(false));
  }, [sel]);

  const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
  const tone = (x) => (x == null ? "var(--htr-ink-3)" : x > 0 ? "var(--htr-bull)" : x < 0 ? "var(--htr-bear)" : "var(--htr-ink-2)");
  // Descriptive "how far already moved" labels — NOT buy/sell signals (Rule 11.13.2).
  const FRESH = {
    extended:     { label: "已大涨", color: "var(--htr-warn)", bg: "var(--htr-warn-bg)" },
    rolling_over: { label: "回吐中", color: "var(--htr-bear)", bg: "var(--htr-bear-bg)" },
    falling:      { label: "下行",   color: "var(--htr-ink-2)", bg: "var(--htr-surface-2)" },
    fresh:        { label: "未过热", color: "var(--htr-info)", bg: "var(--htr-info-bg)" },
    unknown:      { label: "—",      color: "var(--htr-ink-3)", bg: "var(--htr-surface-2)" },
  };

  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title={<span>事件作战台 · <Term>Rule 11.13</Term></span>} sub="谁受影响 · 涨完没有 · 只读·非预测" />
      <div style={{ padding: "11px 14px", display: "flex", flexDirection: "column", gap: 9 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>事件/主题</span>
          <select value={sel} onChange={(e) => setSel(e.target.value)}
            style={{ fontSize: 11, padding: "3px 6px", background: "var(--htr-surface-2)", color: "var(--htr-ink-1)", border: "1px solid var(--htr-line)", borderRadius: 5 }}>
            <optgroup label="主题">{themes.map((t) => <option key={t.id} value={t.id}>{t.id}</option>)}</optgroup>
            <optgroup label="事件">{events.map((e) => <option key={e} value={e}>{e}</option>)}</optgroup>
          </select>
          {loading && <span style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>载入中…</span>}
        </div>

        {err && !data && <div style={{ fontSize: 11, color: "var(--htr-warn)" }}>数据未就绪（行情源离线时降级，不伪造）</div>}

        {data && (
          <>
            {data.themes && data.themes.length > 0 && (
              <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>受影响主题：{data.themes.join(" · ")}（对照 {data.benchmark}）</div>
            )}
            <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) auto auto auto auto", gap: "3px 8px", fontSize: 11, alignItems: "center" }}>
              <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)" }}>标的</span>
              <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)", textAlign: "right" }}>5日</span>
              <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)", textAlign: "right" }}>20日</span>
              <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)", textAlign: "right" }}>距20日高</span>
              <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)", textAlign: "right" }}>状态</span>
              {(data.exposure || []).map((r) => {
                const f = FRESH[r.freshness] || FRESH.unknown;
                return (
                  <React.Fragment key={r.symbol}>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}><span className="htr-mono" style={{ fontWeight: 700 }}>{r.symbol}</span> <span style={{ color: "var(--htr-ink-3)" }}>{r.name}</span></span>
                    <span className="htr-num" style={{ textAlign: "right", color: tone(r.ret5d) }}>{pct(r.ret5d)}</span>
                    <span className="htr-num" style={{ textAlign: "right", color: tone(r.ret20d) }}>{pct(r.ret20d)}</span>
                    <span className="htr-num" style={{ textAlign: "right", color: "var(--htr-ink-3)" }}>{pct(r.distFromHigh20d)}</span>
                    <span style={{ textAlign: "right" }}><span className="htr-chip" style={{ color: f.color, background: f.bg, borderColor: f.color, fontSize: 9.5 }}>{f.label}</span></span>
                  </React.Fragment>
                );
              })}
            </div>
            <div style={{ fontSize: 10, color: "var(--htr-ink-3)", lineHeight: 1.5, borderTop: "1px solid var(--htr-line-soft)", paddingTop: 7 }}>
              {data.disclosure}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function V3CandidateHistoryCard() {
  const [dates, setDates] = v3useState({ live: [], backdated: [] });
  const [date, setDate] = v3useState("");
  const [includeBackdated, setIncludeBackdated] = v3useState(false);
  const [data, setData] = v3useState(null);
  const [trend, setTrend] = v3useState(null);
  const [expanded, setExpanded] = v3useState(false);
  const [err, setErr] = v3useState(false);

  v3useEffect(() => {
    // Cross-day pooled trend — the "全实盘累计" bottom line (Rule 11.11.2 full series).
    fetch("/api/candidates/history/trend", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((t) => { if (t) setTrend(t); })
      .catch(() => {});
  }, []);

  v3useEffect(() => {
    fetch("/api/candidates/history/dates", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!d) return;
        setDates(d);
        // Sync the selector to the latest live date so the dropdown value matches
        // the data shown (avoids the select showing option[0] while the default
        // fetch returns the latest date).
        if (!date && d.live && d.live.length) setDate(d.live[d.live.length - 1]);
      })
      .catch(() => {});
  }, []);

  v3useEffect(() => {
    const q = new URLSearchParams();
    if (date) q.set("date", date);
    if (includeBackdated) q.set("include_backdated", "true");
    fetch("/api/candidates/history?" + q.toString(), { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d) { setData(d); setErr(false); } else setErr(true); })
      .catch(() => setErr(true));
  }, [date, includeBackdated]);

  const pct = (x) => (x == null ? "—" : (x * 100).toFixed(2) + "%");
  const tone = (x) => (x == null ? "var(--htr-ink-3)" : x > 0 ? "var(--htr-bull)" : x < 0 ? "var(--htr-bear)" : "var(--htr-ink-2)");
  const STATUS_LABEL = { symbol_not_found: "数据缺失", insufficient_data: "未成熟", future_cutoff: "未到窗口", malformed_data: "数据异常", no_outcome: "待评估" };
  const pool = dates.live || [];

  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <CardHead title={<span>历史候选复盘 · <Term>Rule 11.11</Term></span>} sub="整批 vs 大盘 · 只读复盘" />
      <div style={{ padding: "11px 14px", display: "flex", flexDirection: "column", gap: 9 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <select value={date} onChange={(e) => setDate(e.target.value)}
            style={{ fontSize: 11, padding: "3px 6px", background: "var(--htr-surface-2)", color: "var(--htr-ink-1)", border: "1px solid var(--htr-line)", borderRadius: 5 }}>
            {pool.length === 0 && <option value="">（无实盘日期）</option>}
            {pool.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <label style={{ fontSize: 10.5, color: "var(--htr-ink-3)", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
            <input type="checkbox" checked={includeBackdated} onChange={(e) => setIncludeBackdated(e.target.checked)} />
            含回测补样
          </label>
        </div>

        {includeBackdated && (
          <div style={{ fontSize: 10.5, color: "var(--htr-warn)", background: "var(--htr-warn-bg)", border: "1px solid var(--htr-warn)", borderRadius: 5, padding: "5px 8px", lineHeight: 1.5 }}>
            ⚠ 合成样本 · 非真实战绩，不计入实盘聚合 (Rule 11.11.5)
          </div>
        )}

        {err && !data && <div style={{ fontSize: 11, color: "var(--htr-warn)" }}>数据未就绪</div>}

        {data && (
          <React.Fragment>
            <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", lineHeight: 1.5, padding: "6px 8px", background: "var(--htr-surface-2)", borderRadius: 6 }}>
              {data.honestyNote}
            </div>
            <div style={{ fontSize: 11, color: "var(--htr-ink-2)" }}>
              {data.tradeDate || "—"} · 候选 <b>{data.candidateCount}</b> 只 · 已成熟 <b>{data.completeCount}</b> · 回测补样隔离 {data.excludedBackdated}
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ color: "var(--htr-ink-3)", fontSize: 10 }}>
                  <th style={{ textAlign: "left", fontWeight: 600 }}>窗口</th>
                  <th style={{ textAlign: "right", fontWeight: 600 }}>整批均涨</th>
                  <th style={{ textAlign: "right", fontWeight: 600 }}>大盘</th>
                  <th style={{ textAlign: "right", fontWeight: 600 }}>超额</th>
                  <th style={{ textAlign: "right", fontWeight: 600 }}>上涨占比</th>
                  <th style={{ textAlign: "right", fontWeight: 600 }}>熟/未熟</th>
                </tr>
              </thead>
              <tbody>
                {data.horizons.map((h) => (
                  <tr key={h.horizon} style={{ borderTop: "1px solid var(--htr-line)" }}>
                    <td style={{ textAlign: "left", fontWeight: 600 }}>{h.horizon}</td>
                    <td style={{ textAlign: "right", color: tone(h.meanReturn) }}>{pct(h.meanReturn)}</td>
                    <td style={{ textAlign: "right", color: "var(--htr-ink-2)" }}>{pct(h.benchmarkReturn)}</td>
                    <td style={{ textAlign: "right", color: tone(h.excessReturn), fontWeight: 700 }}>{pct(h.excessReturn)}</td>
                    <td style={{ textAlign: "right", color: "var(--htr-ink-2)" }}>{h.positiveShare == null ? "—" : (h.positiveShare * 100).toFixed(0) + "%"}</td>
                    <td style={{ textAlign: "right", color: "var(--htr-ink-3)" }}>{h.maturedCount}/{h.immatureCount}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <button onClick={() => setExpanded(!expanded)}
              style={{ alignSelf: "flex-start", fontSize: 10.5, color: "var(--htr-ink-2)", background: "none", border: "none", cursor: "pointer", padding: "2px 0" }}>
              {expanded ? "▾ 收起个股明细" : "▸ 展开个股明细 (" + data.candidates.length + ")"}
            </button>
            {expanded && (
              <div style={{ display: "flex", flexDirection: "column", maxHeight: 320, overflow: "auto" }}>
                {data.candidates.map((c) => {
                  const r5 = c.realizedReturns ? c.realizedReturns["5D"] : null;
                  const matured = c.outcomeStatus === "complete";
                  return (
                    <div key={c.symbol} style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", fontSize: 10.5, padding: "3px 4px", borderBottom: "1px solid var(--htr-line)", gap: 8 }}>
                      <span style={{ fontWeight: 600, minWidth: 56 }}>{c.symbol}</span>
                      <span style={{ color: "var(--htr-ink-3)" }}>研究分 {(c.buyScore * 100).toFixed(0)}</span>
                      <span style={{ color: matured ? tone(r5) : "var(--htr-ink-3)", textAlign: "right", minWidth: 64 }}>
                        {matured ? "5D " + pct(r5) : (STATUS_LABEL[c.outcomeStatus] || c.outcomeStatus)}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </React.Fragment>
        )}

        {/* 全实盘累计 — cross-day pooled trend (Rule 11.11.2 full-series view).
            The single honest bottom line for "is the cohort beating the index
            across every forward sample so far?" */}
        {trend && trend.points && trend.points.length > 0 && (() => {
          const ex3 = trend.points.map((pt) => {
            const h = (pt.horizons || []).find((x) => x.horizon === "3D");
            return { date: pt.tradeDate, ex: h ? h.excessReturn : null };
          });
          const maxAbs = Math.max(0.005, ...ex3.map((d) => (d.ex == null ? 0 : Math.abs(d.ex))));
          return (
            <div style={{ borderTop: "1px solid var(--htr-line)", paddingTop: 8, marginTop: 2, display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ fontSize: 10.5, fontWeight: 700, color: "var(--htr-ink-2)" }}>
                全实盘累计 · {trend.points.length} 日 · 已成熟 {trend.totalComplete} 样本
              </div>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {trend.pooled.map((p) => (
                  <div key={p.horizon} style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>
                    {p.horizon} 超额 <b style={{ color: tone(p.excessReturn) }}>{pct(p.excessReturn)}</b>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 30 }} title="每日 3D 超额（高=幅度，绿/红=跑赢/跑输大盘）">
                {ex3.map((d) => (
                  <div key={d.date}
                    title={d.date + " · 3D 超额 " + (d.ex == null ? "样本未成熟" : (d.ex * 100).toFixed(2) + "%")}
                    style={{ flex: 1, minWidth: 4, height: (d.ex == null ? 2 : Math.max(2, Math.abs(d.ex) / maxAbs * 28)), background: d.ex == null ? "var(--htr-ink-3)" : (d.ex >= 0 ? "var(--htr-bull)" : "var(--htr-bear)"), borderRadius: 2, opacity: d.ex == null ? 0.3 : 1 }} />
                ))}
              </div>
              <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", lineHeight: 1.4 }}>
                每日 3D 整批 vs 同期 1306.T · 描述性观察，非胜率（{trend.honestyNote ? "Rule 11.11" : ""}）
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}

window.V3MarketDashboard = V3MarketDashboard;
