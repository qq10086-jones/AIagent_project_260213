// htr-v4.jsx — refined "决策日志为脊" (Workflow / Timeline Spine).
// Center is one chronological spine weaving news → candidate emergence →
// decision actions → macro. Left rail = macro context, right rail = current
// target + vertical ladder + research-mode actions (write to decision log only).
// Refinement: document-flow layout (no viewport crush), refined type, advice-only.

function V4WorkflowSpine() {
  const data = window.HTR_DATA;
  // P6-B — selectable tracked symbol (shared htr_symbol key); no longer locked to [0].
  const [sel, setSel] = React.useState(() => { try { return localStorage.getItem("htr_symbol") || data.candidates[0].symbol; } catch (e) { return data.candidates[0].symbol; } });
  React.useEffect(() => { try { localStorage.setItem("htr_symbol", sel); } catch (e) {} }, [sel]);
  // Rule 11.9.5 — overlay real per-symbol detail from /api/symbol/{T}/*.
  const top = useEnrichedCandidate(data.candidates.find((c) => c.symbol === sel) || data.candidates[0]);
  const livePrice = useTickingPrice(top.price);
  const events = React.useMemo(() => v4BuildEvents(data), [data]);
  return (
    <div className="htr v4-root" style={{ width: "100%", minHeight: "100%", background: "var(--htr-bg)", padding: "12px 18px 22px", display: "flex", flexDirection: "column", gap: 12 }}>
      <V4Header meta={data.meta} events={events} />
      <div className="v4-grid">
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="htr-card" style={{ overflow: "hidden" }}>
            <CardHead title={<Term k="市场温度">多市场温度</Term>} sub="External" />
            <div>{data.markets.map((m) => <MarketTempCell key={m.id} m={m} variant="row" />)}</div>
          </div>
          <div className="htr-card" style={{ overflow: "hidden" }}>
            <CardHead title={<Term>主题热力</Term>} sub="Theme" />
            <div style={{ padding: "10px 12px" }}><ThemeHeatBars themes={data.themes} /></div>
          </div>
          <V3PortfolioCard positions={data.positions} />
          {/* Rule 11.17 — Position Exit Discipline Board (shared governed card, four-variant parity). */}
          <ExitBoardCard />
        </div>

        <div className="htr-card" style={{ display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
          <V4SpineHeader trackedSymbol={top.symbol} score={top.score} />
          <V4CandidateStrip candidates={data.candidates} selected={top.symbol} onSelect={setSel} />
          <div style={{ overflow: "auto", padding: "10px 22px 18px", maxHeight: "calc(100vh - 220px)" }}>
            <V4Spine events={events} hero={top} livePrice={livePrice} />
          </div>
        </div>

        <V4RightRail candidate={top} livePrice={livePrice} />
      </div>
      {/* P24-10 — Action Board (wide table) + review feeds move full-width below the
          grid so the left column no longer towers over the spine while the spine
          cuts off (the void the operator flagged 2026-07-08). */}
      {/* Rule 11.16 — Daily Action Board (shared governed card, four-variant parity). */}
      <ActionBoardCard />
      <div className="v1-feeds-row">
        {/* Rule 11.11 / 11.13 — cohort review + event desk (shared governed cards). */}
        <V3CandidateHistoryCard />
        <EventDeskCard />
      </div>
      <GateStripCard gates={data.gates} />
    </div>
  );
}

function V4Header({ meta, events }) {
  // Rule 11.9.1/11.9.4 — counts DERIVED from the real event stream, never the old
  // hardcoded [7,6,8,3,0]. Only categories that actually occurred are shown; a
  // category with no real event is dropped rather than printed as a literal 0.
  const counts = (events || []).reduce((m, e) => {
    const g = e.kind && e.kind.indexOf("news_") === 0 ? "新闻"
      : (e.kind === "candidate" || e.kind === "leader") ? "候选浮现"
      : e.kind === "decision" ? "决策动作"
      : e.kind === "macro" ? "宏观事件"
      : e.kind === "session" ? "开盘" : null;
    if (g) m[g] = (m[g] || 0) + 1;
    return m;
  }, {});
  const chips = Object.entries(counts);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,auto) minmax(0,1fr) minmax(0,auto)", alignItems: "center", gap: 16, minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 28, height: 28, borderRadius: 6, background: "var(--htr-accent)", color: "var(--htr-accent-ink)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800 }}>HTR</div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700 }}>今日叙事 · Story of {meta.tradeDate}</div>
          <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.08em" }}>UNIFIED EVENT STREAM · §8.6 decision log · §10 governance</div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 7, flexWrap: "wrap" }}>
        {chips.map(([l, c]) => (
          <span key={l} style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "4px 10px", borderRadius: 999, border: "1px solid var(--htr-accent)", background: "var(--htr-accent-soft)", fontSize: 11.5, color: "var(--htr-accent)" }}>
            <span style={{ fontWeight: 600 }}>{l}</span><span className="htr-mono" style={{ fontWeight: 700, fontSize: 10.5 }}>{c}</span>
          </span>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, justifyContent: "flex-end" }}>
        <span className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-2)", whiteSpace: "nowrap" }}>{meta.asof}</span>
        <CalibPill sample={meta.calibration.sample} />
      </div>
    </div>
  );
}

function V4SpineHeader({ trackedSymbol, score }) {
  return (
    <div style={{ padding: "12px 22px", background: "var(--htr-surface-2)", borderBottom: "1px solid var(--htr-line)", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
      <div>
        <div style={{ fontSize: 14, fontWeight: 700 }}>今天发生了什么 · 为什么 {trackedSymbol} 跑出来</div>
        <div style={{ fontSize: 11.5, color: "var(--htr-ink-3)", marginTop: 2 }}>按时间倒序 · 新闻 + 决策日志 + 宏观 + 候选浮现 全部穿成一线</div>
      </div>
      <div style={{ display: "flex", gap: 9, alignItems: "center" }}>
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>{trackedSymbol} 研究分</span>
        <span className="htr-num" style={{ fontSize: 19, fontWeight: 700, color: "var(--htr-accent)" }}>{score}</span>
        <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)" }}>{HTR.LABELS.uncalibrated}</span>
      </div>
    </div>
  );
}

// P6-B — compact candidate picker so V4 isn't locked to candidates[0]; sets the
// shared selected symbol (htr_symbol). Horizontal-scroll chip row.
function V4CandidateStrip({ candidates, selected, onSelect }) {
  return (
    <div style={{ display: "flex", gap: 6, padding: "8px 22px", borderBottom: "1px solid var(--htr-line)", overflowX: "auto", whiteSpace: "nowrap", alignItems: "center" }}>
      <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)", flexShrink: 0, marginRight: 2 }}>切换候选</span>
      {candidates.slice(0, 12).map((c) => {
        const active = c.symbol === selected;
        return (
          <button key={c.symbol} onClick={() => onSelect(c.symbol)} title={`切换到 ${c.symbol}`}
            style={{ flexShrink: 0, cursor: "pointer", fontFamily: "inherit", fontSize: 11, fontWeight: active ? 700 : 500, padding: "3px 9px", borderRadius: 999, border: "1px solid " + (active ? "var(--htr-accent)" : "var(--htr-line-2)"), background: active ? "var(--htr-accent)" : "transparent", color: active ? "var(--htr-accent-ink)" : "var(--htr-ink-2)" }}>
            <span className="htr-mono">#{c.rank} {c.symbol}</span>
          </button>
        );
      })}
    </div>
  );
}

function V4Spine({ events, hero, livePrice }) {
  return (
    <div>
      {/* P6-A — leader pinned at the top, outside the time-sorted stream */}
      <div style={{ marginBottom: 16 }}><V4LeaderCard hero={hero} livePrice={livePrice} /></div>
      <div style={{ position: "relative", paddingLeft: 102 }}>
        <div style={{ position: "absolute", left: 80, top: 8, bottom: 8, width: 2, background: "var(--htr-line)" }} />
        {events.map((e, i) => <V4SpineRow key={i} ev={e} />)}
      </div>
    </div>
  );
}

function V4SpineRow({ ev }) {
  const dotColor = ({ news_high: "var(--htr-bear)", news_med: "var(--htr-warn)", news_low: "var(--htr-ink-3)", candidate: "var(--htr-accent)", decision: "var(--htr-info)", macro: "var(--htr-warn)", session: "var(--htr-ink-2)" })[ev.kind] || "var(--htr-ink-3)";
  return (
    <div style={{ position: "relative", paddingBottom: 12 }}>
      <div style={{ position: "absolute", left: -102, top: 2, width: 76, textAlign: "right" }}>
        <div className="htr-mono" style={{ fontSize: 11, fontWeight: 600, color: "var(--htr-ink-2)" }}>{ev.ts}</div>
        <div style={{ fontSize: 9, color: "var(--htr-ink-4)", letterSpacing: "0.06em", marginTop: 1 }}>{ev.kindLabel}</div>
      </div>
      <div style={{ position: "absolute", left: -24, top: 6, width: 16, height: 16, borderRadius: "50%", background: dotColor, border: "3px solid var(--htr-surface)", boxShadow: `0 0 0 1px ${dotColor}` }} />
      <V4SpineCard ev={ev} dotColor={dotColor} />
    </div>
  );
}

function V4SpineCard({ ev, dotColor }) {
  return (
    <div style={{ padding: "9px 13px", background: "var(--htr-surface)", border: "1px solid var(--htr-line)", borderLeft: `3px solid ${dotColor}`, borderRadius: 6 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 3, flexWrap: "wrap" }}>
        {ev.src && <span className="htr-chip" style={{ fontSize: 9.5 }}>{ev.src}</span>}
        {ev.action && <code style={{ fontFamily: "var(--htr-font-mono)", fontSize: 10, background: "var(--htr-surface-3)", padding: "1px 6px", borderRadius: 3, color: "var(--htr-ink-2)" }}>{ev.action}</code>}
        {ev.weight && <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)" }}>weight {ev.weight}</span>}
        <span style={{ flex: 1 }} />
        {ev.symbols && ev.symbols.map((s) => <span key={s} className="htr-chip" style={{ fontSize: 9.5 }}>{s}</span>)}
      </div>
      <div style={{ fontSize: 13, color: "var(--htr-ink)", lineHeight: 1.45 }}>{ev.text}</div>
      {ev.note && <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginTop: 4 }}>{ev.note}</div>}
    </div>
  );
}

function V4LeaderCard({ hero, livePrice }) {
  return (
    <div style={{ padding: "13px 15px", background: "linear-gradient(180deg, var(--htr-accent-soft) 0%, var(--htr-surface) 100%)", border: "1px solid var(--htr-accent)", borderRadius: 8, display: "flex", flexDirection: "column", gap: 11 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span style={{ padding: "3px 10px", background: "var(--htr-accent)", color: "var(--htr-accent-ink)", fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", borderRadius: 4 }}>今日候选首位</span>
        <span style={{ fontSize: 13, color: "var(--htr-ink)" }}>未校准排序信号 (Rule 9.4) · 非胜率</span>
        <CatalystBadges c={hero} size="sm" />
      </div>
      <div className="v4-leader-grid">
        <div><div className="htr-mono" style={{ fontSize: 19, fontWeight: 800 }}>{hero.symbol}</div><div style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{hero.nameCn}</div></div>
        <div><div className="htr-num" style={{ fontSize: 18, fontWeight: 700 }}><AnimatedPrice value={livePrice} decimals={0} /></div><div className={"htr-num " + HTR.changeClass(hero.chg)} style={{ fontSize: 10.5 }}>{HTR.arrow(hero.chg)} {HTR.fmtPct(hero.chg)}</div></div>
        <div><div className="htr-num" style={{ fontSize: 18, fontWeight: 800, color: "var(--htr-accent)" }}>{hero.score}</div><div style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{HTR.LABELS.scoreUncalibrated}</div></div>
        {hero._status && hero._status.kline === "ok" && Array.isArray(hero.kline) && hero.kline.length
          ? <div><Sparkline data={hero.kline.map((b) => b.close)} width={130} height={32} stroke="var(--htr-accent)" fill="var(--htr-accent-soft)" /><div style={{ fontSize: 9, color: "var(--htr-ink-3)", textAlign: "center", marginTop: 2 }}>价格 · {hero.kline.length}d (真实)</div></div>
          : <div />}
      </div>
      <div style={{ fontSize: 12, color: "var(--htr-ink-2)", lineHeight: 1.55, padding: "9px 11px", background: "var(--htr-surface)", borderRadius: 6, border: "1px solid var(--htr-line-soft)" }}>
        <strong style={{ color: "var(--htr-ink)" }}>判断: </strong>{hero.one_liner}
      </div>
    </div>
  );
}

function V4RightRail({ candidate, livePrice }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="htr-card" style={{ padding: "13px 15px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 9 }}>
          <span className="htr-eyebrow" style={{ color: "var(--htr-accent)" }}>当前主标的</span>
          <span style={{ flex: 1 }} />
          <span className="htr-chip warn">{candidate.priority}</span>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <div className="htr-mono" style={{ fontSize: 21, fontWeight: 800 }}>{candidate.symbol}</div>
          <div style={{ fontSize: 11.5, color: "var(--htr-ink-3)" }}>{candidate.nameCn}</div>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 7 }}>
          <span className="htr-num" style={{ fontSize: 25, fontWeight: 700 }}><AnimatedPrice value={livePrice} decimals={0} /></span>
          <span className={"htr-num " + HTR.changeClass(candidate.chg)} style={{ fontSize: 12.5, fontWeight: 700 }}>{HTR.arrow(candidate.chg)} {HTR.fmtPct(candidate.chg)}</span>
          <span style={{ flex: 1 }} />
          <div style={{ textAlign: "right" }}><div className="htr-num" style={{ fontSize: 19, fontWeight: 800, color: "var(--htr-accent)" }}>{candidate.score}</div><div style={{ fontSize: 9, color: "var(--htr-ink-3)" }}>研究分</div></div>
        </div>
      </div>
      <div className="htr-card" style={{ overflow: "hidden" }}>
        <CardHead title={<Term>七档价格阶梯</Term>} sub="Price Ladder · 建议价位" />
        <div style={{ padding: "12px 14px" }}><VerticalLadder ladder={candidate.ladder} currentPrice={livePrice} height={340} /></div>
        <div style={{ padding: "12px 14px", borderTop: "1px solid var(--htr-line)", background: "var(--htr-surface-2)" }}>
          <div className="htr-eyebrow" style={{ marginBottom: 7 }}>研究态度 · 仅供参考</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 7 }}>
            {[["分批介入", "bull"], ["设条件单", "info"], ["仅观察", "warn"], ["放弃", "bear"]].map(([t, tone]) => {
              const map = { bull: ["var(--htr-bull-bg)", "var(--htr-bull)"], bear: ["var(--htr-bear-bg)", "var(--htr-bear)"], info: ["var(--htr-info-bg)", "var(--htr-info)"], warn: ["var(--htr-warn-bg)", "var(--htr-warn)"] };
              const [bg, fg] = map[tone];
              return <span key={t} style={{ padding: "8px 10px", border: `1px solid ${fg}44`, background: bg, color: fg, borderRadius: 6, fontSize: 12, fontWeight: 700, textAlign: "center" }}>{t}</span>;
            })}
          </div>
          <div style={{ fontSize: 10, color: "var(--htr-ink-3)", marginTop: 9, lineHeight: 1.45 }}>研究态度参考，<strong>非可点按钮</strong>：HTR 不下单、不写决策日志、不连券商。执行由用户在外部券商手动完成 (Rule 3 advice-only)。</div>
        </div>
      </div>
      <FactorCard candidate={candidate} />
      <OutcomesCard candidate={candidate} />
    </div>
  );
}

function v4BuildEvents(data) {
  const evs = [];
  data.newsTimeline.forEach((n) => evs.push({ ts: n.ts, tsKey: v4ParseTs(n.ts), kind: n.weight === "high" ? "news_high" : n.weight === "medium" ? "news_med" : "news_low", kindLabel: "NEWS · " + n.weight.toUpperCase(), src: n.src, weight: n.weight, text: n.text, symbols: n.linkedSymbols }));
  const hhmm = (ts) => (String(ts).slice(11, 16) || "") + " JST";   // "…T15:30:…" → "15:30 JST"
  data.decisionLog.forEach((d) => {
    if (d.action === "session_open") evs.push({ ts: hhmm(d.ts), tsKey: v4ParseTs(hhmm(d.ts)), kind: "session", kindLabel: "SESSION", action: d.action, text: "JST 09:00 开盘" });
    else if (d.action === "macro_change") evs.push({ ts: hhmm(d.ts), tsKey: v4ParseTs(hhmm(d.ts)), kind: "macro", kindLabel: "MACRO", action: d.action, text: d.note, symbols: [d.symbol] });
    else if (d.action === "scan_completed") evs.push({ ts: hhmm(d.ts), tsKey: v4ParseTs(hhmm(d.ts)), kind: "decision", kindLabel: "DECISION LOG", action: d.action, text: d.note });
    else if (d.action === "news_overlay_hit") evs.push({ ts: hhmm(d.ts), tsKey: v4ParseTs(hhmm(d.ts)), kind: "decision", kindLabel: "新闻命中", action: d.action, text: d.note, symbols: d.symbol ? [d.symbol] : undefined });
  });
  // Codex-2026-06-06 #4 — surface the real candidate_persisted scan (§8.6) as ONE
  // aggregate event instead of N identical rows (the real decisionLog is N=候选数
  // persists at the same cutoff with the same model note).
  const persisted = data.decisionLog.filter((d) => d.action === "candidate_persisted");
  if (persisted.length) {
    const ts = hhmm(persisted[0].ts);
    const model = String(persisted[0].note || "").replace(/^模型\s*/, "") || "screener";
    evs.push({ ts, tsKey: v4ParseTs(ts), kind: "candidate", kindLabel: "候选入选 · §8.6", text: `${persisted.length} 支候选写入决策日志 · 模型 ${model}`, symbols: persisted.slice(0, 8).map((d) => d.symbol) });
  }
  // P6-A — the LEADER is no longer interleaved as a 15:30 event (it would be
  // buried under later-clock news once time-sorted). It is pinned to the TOP of
  // the spine as a non-event hero card (V4Spine), driven by the selected candidate.
  evs.sort((a, b) => b.tsKey - a.tsKey);
  return evs;
}
function v4ParseTs(ts) { const m = ts.match(/(\d{1,2}):(\d{2})/); return m ? parseInt(m[1], 10) * 60 + parseInt(m[2], 10) : 0; }

window.V4WorkflowSpine = V4WorkflowSpine;
