// htr-v1.jsx — refined "三栏专业终端" (Pro Terminal).
// Keeps the dense Bloomberg-style 3-column identity, but:
//   • document-flow height (minHeight) instead of a hard viewport lock, so the
//     chart + action zone never get crushed on a short screen — they keep their
//     natural size and the page scrolls.
//   • refined type scale (≥12px), calmer chrome, advice-only label on the action zone.

function V1ProTerminal() {
  const data = window.HTR_DATA;
  const [sel, setSel] = React.useState(() => { try { return localStorage.getItem("htr_symbol") || data.candidates[0].symbol; } catch (e) { return data.candidates[0].symbol; } });
  React.useEffect(() => { try { localStorage.setItem("htr_symbol", sel); } catch (e) {} }, [sel]);
  // Rule 11.9.5 — overlay real per-symbol detail from /api/symbol/{T}/* instead
  // of the offline mock synth (kline / strategy / factors / outcomes / AI / debate).
  const top = useEnrichedCandidate(data.candidates.find((c) => c.symbol === sel) || data.candidates[0]);
  const livePrice = useTickingPrice(top.price);

  return (
    <div className="htr v1-root" style={{ width: "100%", minHeight: "100%", background: "var(--htr-bg)", padding: "12px 16px 22px", display: "flex", flexDirection: "column", gap: 12 }}>
      <V1TopBar asof={data.meta.asof} refresh={data.meta.refreshLabel} calib={data.meta.calibration} markets={data.markets} />
      <div className="v1-grid">
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <V1Panel title={<Term k="市场温度">多市场温度</Term>} sub="External Temperature">
            {data.markets.map((m) => <MarketTempCell key={m.id} m={m} variant="row" />)}
          </V1Panel>
          <V1Panel title={<Term>主题热力</Term>} sub="Theme Heat · 今日">
            <div style={{ padding: "8px 12px 10px" }}><ThemeHeatBars themes={data.themes} /></div>
          </V1Panel>
          <V3PortfolioCard positions={data.positions} />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
          <V1HeroHeader candidate={top} livePrice={livePrice} />
          <V1KLineLadderPanel top={top} />
          <V1ActionZone candidate={top} livePrice={livePrice} />
          <V1RiskSizing candidate={top} positions={data.positions} />
          {/* P6-C — governed disclosure parity (Rule 9.6 / 11.6): risk_warnings +
              catalyst (collapsed StrategyCard, risk always visible) + factor +
              outcomes, reusing the shared governed cards. */}
          <V3StrategyCard candidate={top} />
          <FactorCard candidate={top} />
          <OutcomesCard candidate={top} />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <V1Panel title={<Term>新闻催化</Term>} sub="News Catalyst · 6h">
            <div style={{ padding: "2px 12px 8px", maxHeight: 260, overflow: "auto" }}><NewsTimeline items={data.newsTimeline} max={6} compact /></div>
          </V1Panel>
          <V1CockpitPanel cockpit={data.dailyCockpit} />
          <V1Panel title="候选清单" sub={`点击切换 · 当前 ${top.symbol}`}>
            <div style={{ maxHeight: 240, overflow: "auto" }}>{data.candidates.map((c) => <V1CandidateRow key={c.symbol} c={c} active={c.symbol === top.symbol} onClick={() => setSel(c.symbol)} />)}</div>
          </V1Panel>
          {/* Rule 11.17 — Position Exit Discipline Board (shared governed card, four-variant parity). */}
          <ExitBoardCard />
          {/* Rule 11.16 — Daily Action Board (shared governed card, four-variant parity). */}
          <ActionBoardCard />
          {/* Rule 11.11 — historical-candidate cohort review (shared governed card, four-variant parity). */}
          <V3CandidateHistoryCard />
          <EventDeskCard />
          <V1Panel title={<span>决策日志 · <Term>§8.6</Term></span>} sub={data.meta.tradeDate}>
            <div style={{ maxHeight: 220, overflow: "auto" }}><DecisionLog entries={data.decisionLog} max={6} /></div>
          </V1Panel>
        </div>
      </div>
      <GateStripCard gates={data.gates} />
    </div>
  );
}

function V1TopBar({ asof, refresh, calib, markets }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,auto) minmax(0,1fr) minmax(0,auto)", alignItems: "center", gap: 16, minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <div style={{ width: 28, height: 28, borderRadius: 6, background: "var(--htr-accent)", color: "var(--htr-accent-ink)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800 }}>HTR</div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700 }}>HotThemeRotator</div>
            <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.08em" }}>RESEARCH · v0.7.4 · JST</div>
          </div>
        </div>
        <span className="htr-chip">日股主</span>
        <span className="htr-chip info">A股/美股 温度因子</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}><V1SessionStrip markets={markets} /></div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, justifyContent: "flex-end" }}>
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)", whiteSpace: "nowrap" }}>刷新 <strong className="htr-num" style={{ color: "var(--htr-ink)" }}>{refresh}</strong></span>
        <span className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-2)", whiteSpace: "nowrap" }}>{asof}</span>
        <CalibPill sample={calib.sample} />
      </div>
    </div>
  );
}

function CalibPill({ sample }) {
  return (
    <span className="htr-chip warn" title="未校准研究分 — 还没经过足够的预测↔结果配对校准，不能当胜率读" style={{ cursor: "help" }}>
      <span style={{ width: 6, height: 6, background: "var(--htr-warn)", borderRadius: "50%" }} /> 未校准 · 研究分 <span style={{ color: "var(--htr-ink-3)" }}>n={sample}</span>
    </span>
  );
}

function V1SessionStrip({ markets }) {
  // Rule 11.9.3 — session state MUST be DERIVED from real backend market state
  // (the calendar-correct `markets[].state` from _market_session_state), never
  // hardcoded. Map the markets we actually have signals for to their sessions;
  // do NOT invent HK / London sessions we carry no real state for.
  const ms = Array.isArray(markets) ? markets : [];
  const stateOf = (...ids) => { for (const id of ids) { const m = ms.find((x) => x.id === id); if (m && m.state) return m.state; } return null; };
  const sessions = [
    { label: "东京", state: stateOf("N225", "TOPIX") },
    { label: "纽约", state: stateOf("SPX", "SOX") },
    { label: "上海", state: stateOf("SSE") },
    { label: "外汇", state: stateOf("USDJPY") },
  ].filter((s) => s.state);
  if (!sessions.length) return null;
  const dot = (st) => (st === "OPEN" || st === "LIVE") ? "var(--htr-bull)" : st === "PRE" ? "var(--htr-warn)" : st === "UNKNOWN" ? "var(--htr-ink-4)" : "var(--htr-ink-4)";
  return (
    <div style={{ display: "flex", gap: 16 }}>
      {sessions.map((s) => (
        <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: dot(s.state) }} />
          <span style={{ fontSize: 11.5, fontWeight: 600 }}>{s.label}</span>
          <span style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{s.state}</span>
        </div>
      ))}
    </div>
  );
}

function V1Panel({ title, sub, right, children }) {
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 13px", borderBottom: "1px solid var(--htr-line)", background: "var(--htr-surface-2)" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <div style={{ fontSize: 12.5, fontWeight: 700 }}>{title}</div>
          {sub && <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{sub}</div>}
        </div>
        {right}
      </div>
      <div>{children}</div>
    </div>
  );
}

function V1HeroHeader({ candidate, livePrice }) {
  return (
    <div className="htr-card" style={{ padding: "14px 18px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span className="htr-eyebrow">今日最值得盯 · #01</span>
        <span className="htr-chip warn">{candidate.priority}</span>
        {candidate.theme && candidate.theme !== "screener_v2" && <span className="htr-chip accent">{candidate.theme}</span>}
        <CatalystBadges c={candidate} />
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>决策时刻 <strong className="htr-num" style={{ color: "var(--htr-ink)" }}>{candidate.decisionCutoff}</strong></span>
      </div>
      <div className="v1-hero-grid">
        <div>
          <div className="htr-mono" style={{ fontSize: 23, fontWeight: 800, letterSpacing: "-0.01em" }}>{candidate.symbol}</div>
          <div style={{ fontSize: 11.5, color: "var(--htr-ink-3)" }}>{[candidate.nameJa, candidate.nameCn].filter(Boolean).join(" · ") || candidate.symbol}</div>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{ fontSize: 32, fontWeight: 700 }}><AnimatedPrice value={livePrice} decimals={0} /></span>
          <span className={"htr-num " + HTR.changeClass(candidate.chg)} style={{ fontSize: 14, fontWeight: 700 }}>{HTR.arrow(candidate.chg)} {HTR.fmtPct(candidate.chg)}</span>
        </div>
        <V1MetricInline label="研究分" value={candidate.score} />
        {/* derive R:R from the real ladder; theme heat from real themes (— if none) */}
        <V1MetricInline label="风险/收益" value={(() => { const L = candidate.ladder || []; const ft = (k) => (L.find((r) => r.kind === k) || {}).price; const eb = ft("entry_balanced"), e1 = ft("exit_1"), sp = ft("stop"); return (eb && e1 && sp && eb > sp) ? `1 : ${((e1 - eb) / (eb - sp)).toFixed(1)}` : "—"; })()} />
        <V1MetricInline label="主题热力" value={(() => { const t = ((window.HTR_DATA && window.HTR_DATA.themes) || []).find((x) => x.id === candidate.themeId); return t ? `${t.heat}°` : "—"; })()} accent />
      </div>
      <div style={{ fontSize: 13.5, color: "var(--htr-ink)", lineHeight: 1.55, padding: "8px 0 0", borderTop: "1px solid var(--htr-line-soft)" }}>
        <span style={{ color: "var(--htr-ink-3)", marginRight: 8 }}>判断</span>{candidate.one_liner}
      </div>
    </div>
  );
}

function V1MetricInline({ label, value, accent }) {
  return (
    <div style={{ textAlign: "right" }}>
      <div className="htr-num" style={{ fontSize: 18, fontWeight: 700, color: accent ? "var(--htr-accent)" : "var(--htr-ink)" }}>{value}</div>
      <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{label}</div>
    </div>
  );
}

function V1KLineLadderPanel({ top }) {
  const [boxRef, { width }] = useElementSize();
  const H = 360;
  return (
    <V1Panel title={<span>价格走势 · <Term>七档阶梯</Term></span>} sub={`${top.kline.length} sessions · ${top.symbol}`} right={<span style={{ display: "flex", gap: 6, alignItems: "center" }}>{top._status && top._status.kline === "failed" && <span className="htr-chip warn" style={{ fontSize: 10 }} title="真实 K 线未就绪，下方为示例 (Rule 11.9.4)">示例K线</span>}<span className="htr-chip"><Term>MA20</Term>/<Term>MA60</Term> · <Term k="52w 高">52w 高低</Term></span></span>}>
      <div className="v1-chart-row" style={{ padding: 8 }}>
        <div ref={boxRef} style={{ minWidth: 0, height: H }}>
          {width > 0 && <KLineChart key={top.symbol} data={top.kline} ladder={null} width={width - 12} height={H - 12} padding={{ top: 18, right: 30, bottom: 22, left: 12 }} withVolume withMA with52wLines />}
        </div>
        <V1LadderRail ladder={top.ladder} currentPrice={top.price} />
      </div>
    </V1Panel>
  );
}

function V1LadderRail({ ladder, currentPrice }) {
  const exits = ladder.filter((r) => r.kind.startsWith("exit"));
  const entries = ladder.filter((r) => r.kind.startsWith("entry"));
  const stop = ladder.find((r) => r.kind === "stop");
  const Lvl = ({ level, tone }) => {
    const color = tone === "bull" ? "var(--htr-bull)" : tone === "bear" ? "var(--htr-bear)" : "var(--htr-info)";
    const bg = tone === "bull" ? "var(--htr-bull-bg)" : tone === "bear" ? "var(--htr-bear-bg)" : "var(--htr-info-bg)";
    return (
      <div style={{ border: `1px solid ${color}55`, background: bg, borderRadius: 6, padding: "6px 8px", display: "grid", gridTemplateColumns: "1fr auto", gap: 4, alignItems: "baseline" }}>
        <span style={{ fontSize: 11, color, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{level.label.replace("买入 · ", "")}</span>
        <span className="htr-num" style={{ fontSize: 12.5, color, fontWeight: 800 }}>{HTR.fmtPrice(level.price, 0)}</span>
        <span style={{ fontSize: 9, color: "var(--htr-ink-3)" }}>{level.kind}</span>
        <span className="htr-num" style={{ fontSize: 10, color: "var(--htr-ink-3)", textAlign: "right" }}>{HTR.fmtPct(level.pct, 1)}</span>
      </div>
    );
  };
  return (
    <div style={{ borderLeft: "1px solid var(--htr-line)", padding: "2px 0 2px 10px", display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "var(--htr-ink-2)" }}>七档阶梯 · 建议价位</span>
        <span className="htr-num" style={{ fontSize: 11, color: "var(--htr-accent)", fontWeight: 700 }}>{HTR.fmtPrice(currentPrice, 0)}</span>
      </div>
      <div style={{ fontSize: 9.5, fontWeight: 700, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>卖出区</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>{exits.map((l) => <Lvl key={l.kind} level={l} tone="info" />)}</div>
      <div style={{ fontSize: 9.5, fontWeight: 700, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>买入区</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>{entries.map((l) => <Lvl key={l.kind} level={l} tone="bull" />)}</div>
      {stop && <Lvl level={stop} tone="bear" />}
    </div>
  );
}

function V1ActionZone({ candidate }) {
  const buyR = candidate.ladder.filter((r) => r.kind.startsWith("entry"));
  const sellR = candidate.ladder.filter((r) => r.kind.startsWith("exit"));
  const stop = candidate.ladder.find((r) => r.kind === "stop");
  const Block = ({ title, tone, children, note }) => {
    const color = tone === "bull" ? "var(--htr-bull)" : tone === "bear" ? "var(--htr-bear)" : "var(--htr-info)";
    return (
      <div style={{ padding: "12px 16px", borderRight: tone !== "bear" ? "1px solid var(--htr-line)" : "none", display: "flex", flexDirection: "column", gap: 5 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 3 }}><span style={{ width: 4, height: 13, background: color, borderRadius: 1 }} /><span style={{ fontSize: 12.5, fontWeight: 700 }}>{title}</span></div>
        {children}{note}
      </div>
    );
  };
  const Row = ({ label, price, pct, tone }) => {
    const color = tone === "bull" ? "var(--htr-bull)" : tone === "bear" ? "var(--htr-bear)" : "var(--htr-info)";
    return (
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", alignItems: "baseline", gap: 10, padding: "3px 0" }}>
        <span style={{ fontSize: 12, color: "var(--htr-ink-2)" }}>{label}</span>
        <span className="htr-num" style={{ fontSize: 13.5, fontWeight: 700, color }}>{HTR.fmtPrice(price, 0)}</span>
        <span className="htr-num" style={{ fontSize: 10.5, color: "var(--htr-ink-3)", minWidth: 40, textAlign: "right" }}>{HTR.fmtPct(pct, 1)}</span>
      </div>
    );
  };
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr" }}>
        <Block title="买入区 · 建议价位" tone="bull">{buyR.map((r) => <Row key={r.kind} label={r.label.replace("买入 · ", "")} price={r.price} pct={r.pct} tone="bull" />)}</Block>
        <Block title="卖出区 · 目标参考" tone="info">{sellR.map((r) => <Row key={r.kind} label={r.label} price={r.price} pct={r.pct} tone="info" />)}</Block>
        <Block title="风险线 · 止损参考" tone="bear" note={(() => {
          // Rule 11.9.1 — show a REAL per-symbol risk warning from the strategy
          // overlay, never a hardcoded macro caveat (the old USD/JPY literal that
          // displayed for every candidate). risk_warnings is [{message,...}].
          // Render the first warning's message; nothing when none exists.
          const rw = (candidate.strategy && candidate.strategy.risk_warnings) || [];
          const first = Array.isArray(rw) ? rw[0] : rw;
          const w = first && typeof first === "object" ? first.message : (typeof first === "string" ? first : null);
          return w ? <div style={{ marginTop: 6, padding: "7px 9px", background: "var(--htr-bear-bg)", borderRadius: 5, fontSize: 10.5, color: "var(--htr-bear)", lineHeight: 1.4 }}>{w}</div> : null;
        })()}><Row label={stop.label} price={stop.price} pct={stop.pct} tone="bear" /></Block>
      </div>
      <div style={{ padding: "8px 16px", borderTop: "1px solid var(--htr-line)", background: "var(--htr-surface-2)", fontSize: 10.5, color: "var(--htr-ink-3)" }}>
        ⚠ advice-only · 价位为建议/参考，非订单 · 由用户在外部券商手动执行 (Rule 3)
      </div>
    </div>
  );
}

function V1CockpitPanel({ cockpit }) {
  const rows = cockpit?.watchlist || [], s = cockpit?.summary || {};
  const blocked = cockpit && cockpit.notificationsInvoked;
  const Metric = ({ label, value }) => <div style={{ background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", borderRadius: 6, padding: "6px 8px" }}><div className="htr-num" style={{ fontSize: 15, fontWeight: 800 }}>{value}</div><div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{label}</div></div>;
  return (
    <V1Panel title={<Term>Daily Cockpit</Term>} sub={cockpit?.activationStage} right={<span className={blocked ? "htr-chip bear" : "htr-chip bull"}>{blocked ? "blocked" : "pull only"}</span>}>
      <div style={{ padding: "10px 12px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6, marginBottom: 7 }}><Metric label="watch" value={s.watchlistCount ?? rows.length} /><Metric label="quote" value={s.symbolsWithQuotes ?? 0} /><Metric label="silent" value={s.silentQueueCount ?? 0} /></div>
        {rows.slice(0, 4).map((r) => (
          <div key={r.symbol} style={{ display: "grid", gridTemplateColumns: "58px 1fr auto", gap: 6, alignItems: "center", fontSize: 11, padding: "2px 0" }}>
            <span className="htr-mono" style={{ fontWeight: 700 }}>{r.symbol}</span>
            <span style={{ color: "var(--htr-ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.quoteStatus}</span>
            <span className="htr-chip warn" style={{ fontSize: 9 }}>TDnet {r.tdnetCount}</span>
          </div>
        ))}
      </div>
    </V1Panel>
  );
}

function V1CandidateRow({ c, active, onClick }) {
  return (
    <div role="button" tabIndex={0} onClick={onClick} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } }}
         style={{ display: "grid", gridTemplateColumns: "22px 64px 1fr 50px 32px", gap: 8, alignItems: "center", padding: "8px 11px", borderBottom: "1px solid var(--htr-line-soft)", borderLeft: active ? "3px solid var(--htr-accent)" : "3px solid transparent", background: active ? "var(--htr-accent-soft)" : "transparent", fontSize: 11.5, cursor: "pointer", userSelect: "none" }}>
      <span className="htr-mono" style={{ color: "var(--htr-ink-3)" }}>#{c.rank}</span>
      <span className="htr-mono" style={{ fontWeight: 700 }}>{c.symbol}</span>
      <span style={{ color: "var(--htr-ink-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.nameCn || c.nameJa || (c.theme && c.theme !== "screener_v2" ? c.theme : "")}</span>
      <span className={"htr-num " + HTR.changeClass(c.chg)} style={{ fontSize: 10.5, textAlign: "right" }}>{HTR.fmtPct(c.chg, 2)}</span>
      <span className="htr-num" style={{ fontSize: 12, fontWeight: 700, textAlign: "right" }}>{c.score}</span>
    </div>
  );
}

// V1RiskSizing — genuine synthesis of portfolio (NAV/cash) + ladder into
// position-sizing & risk math. Exists nowhere else in the app. Read-only,
// research-mode, advice-only (Rule 3); position size is a deterministic
// function of a stated risk budget (Rule 11.1 exploratory calc), never an order.
function V1RiskSizing({ candidate, positions }) {
  const nav = positions && positions.available ? positions.nav : 0;
  const cash = positions && positions.available ? positions.cash : 0;
  const [riskPct, setRiskPct] = React.useState(1);     // % of NAV risked to stop
  const concCapPct = 20;                                // Rule 12.5 default
  const fmtY = (v) => "¥" + Math.round(v).toLocaleString("en-US");
  const L = candidate.ladder || [];
  const stop = L.find((r) => r.kind === "stop");
  const exit1 = L.find((r) => r.kind === "exit_1");
  const exitS = L.find((r) => r.kind === "exit_stretch");
  const entries = L.filter((r) => r.kind.startsWith("entry"));
  if (!stop || !exit1 || !entries.length || !nav) {
    return <V1Panel title="仓位与风险测算" sub="research mode"><div style={{ padding: 14, fontSize: 12, color: "var(--htr-ink-3)" }}>缺少持仓或阶梯数据，无法测算。</div></V1Panel>;
  }
  const riskBudget = nav * riskPct / 100;
  const concCapY = nav * concCapPct / 100;

  const rows = entries.map((e) => {
    const riskPS = e.price - stop.price;
    const qtyRisk = riskPS > 0 ? Math.floor(riskBudget / riskPS) : 0;
    const investRisk = qtyRisk * e.price;
    const pctNav = investRisk / nav * 100;
    const capped = investRisk > concCapY;
    const qtyEff = capped ? Math.floor(concCapY / e.price) : qtyRisk;
    const investEff = qtyEff * e.price;
    const rr = (exit1.price - e.price) / (e.price - stop.price);
    const rrS = (exitS.price - e.price) / (e.price - stop.price);
    return { e, riskPS, qtyEff, investEff, pctNav: investEff / nav * 100, capped, rr, rrS };
  });
  const bal = rows[1] || rows[0];
  // R:R visual (balanced tier): entry→stop = risk, entry→exit1 = reward
  const risk = bal.e.price - stop.price, reward = exit1.price - bal.e.price;
  const tot = risk + reward, riskFrac = risk / tot * 100;

  return (
    <V1Panel
      title={<span><Term>仓位与风险测算</Term> · {candidate.symbol}</span>}
      sub="持仓 × 七档阶梯合成 · 研究参考 · advice-only"
      right={<span className="htr-chip">R = {riskPct}% NAV</span>}
    >
      <div style={{ padding: "12px 16px 0", display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", fontSize: 11.5, color: "var(--htr-ink-2)" }}>
        <span>NAV <b className="htr-num">{fmtY(nav)}</b></span>
        <span>现金 <b className="htr-num">{fmtY(cash)}</b></span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>风险预算
          <span style={{ display: "inline-flex", border: "1px solid var(--htr-line-2)", borderRadius: 5, overflow: "hidden" }}>
            {[0.5, 1, 2].map((p) => (
              <button key={p} onClick={() => setRiskPct(p)} style={{ padding: "2px 8px", fontSize: 11, border: "none", cursor: "pointer", background: riskPct === p ? "var(--htr-accent)" : "transparent", color: riskPct === p ? "#fff" : "var(--htr-ink-2)", fontFamily: "inherit" }}>{p}%</button>
            ))}
          </span>
          <b className="htr-num">{fmtY(riskBudget)}</b>
        </span>
        <span style={{ color: "var(--htr-ink-3)" }}>集中度上限 {concCapPct}% (Rule 12.5)</span>
      </div>

      <div className="v1-intel-grid" style={{ padding: "12px 16px 14px" }}>
        {/* left: per-tier sizing table */}
        <div style={{ minWidth: 0 }}>
          <div className="htr-eyebrow" style={{ marginBottom: 8 }}>买入档 · 测算仓位（每档独立，按风险预算反推股数）</div>
          <div style={{ display: "grid", gridTemplateColumns: "84px 70px 60px 1fr 56px", gap: 8, padding: "0 0 6px", fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.04em", textTransform: "uppercase", borderBottom: "1px solid var(--htr-line)" }}>
            <div>档位</div><div style={{ textAlign: "right" }}>建议价位</div><div style={{ textAlign: "right" }}>止损距</div><div style={{ textAlign: "right" }}>测算股数 · 投入</div><div style={{ textAlign: "right" }}>%NAV</div>
          </div>
          {rows.map((r) => (
            <div key={r.e.kind} style={{ display: "grid", gridTemplateColumns: "84px 70px 60px 1fr 56px", gap: 8, padding: "8px 0", alignItems: "baseline", borderBottom: "1px solid var(--htr-line-soft)", fontSize: 12 }}>
              <div style={{ color: "var(--htr-ink)", fontWeight: 600 }}>{r.e.label.replace("买入 · ", "")}</div>
              <div className="htr-num" style={{ textAlign: "right", fontWeight: 600 }}>{fmtY(r.e.price)}</div>
              <div className="htr-num" style={{ textAlign: "right", color: "var(--htr-bear)" }}>{fmtY(r.riskPS)}</div>
              <div className="htr-num" style={{ textAlign: "right" }}>{r.qtyEff} 股 · {fmtY(r.investEff)}{r.capped && <span style={{ color: "var(--htr-warn)", fontSize: 9.5, marginLeft: 4 }}>封顶</span>}</div>
              <div className="htr-num" style={{ textAlign: "right", fontWeight: 700, color: r.capped ? "var(--htr-warn)" : "var(--htr-ink)" }}>{r.pctNav.toFixed(0)}%</div>
            </div>
          ))}
          <div style={{ marginTop: 8, fontSize: 10.5, color: "var(--htr-ink-3)", lineHeight: 1.5 }}>
            {rows.some((r) => r.capped)
              ? <>⚠ 部分档位按 {riskPct}% 风险预算反推的仓位超过 {concCapPct}% NAV 集中度上限，已按上限 {fmtY(concCapY)} 封顶（Rule 12.5 集中度护栏）。</>
              : <>各档仓位均在 {concCapPct}% NAV 集中度上限内。</>}
          </div>
        </div>

        {/* right: R:R visual + key levels */}
        <div style={{ minWidth: 0, borderLeft: "1px solid var(--htr-line-soft)", paddingLeft: 18 }}>
          <div className="htr-eyebrow" style={{ marginBottom: 8 }}>风险 / 收益 · 以均衡档为基准</div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--htr-ink-3)", marginBottom: 4 }}>
            <span>止损 {fmtY(stop.price)}</span><span>均衡 {fmtY(bal.e.price)}</span><span>首止盈 {fmtY(exit1.price)}</span>
          </div>
          <div style={{ display: "flex", height: 12, borderRadius: 6, overflow: "hidden" }}>
            <div style={{ width: `${riskFrac}%`, background: "var(--htr-bear)" }} title="风险段" />
            <div style={{ width: `${100 - riskFrac}%`, background: "var(--htr-bull)" }} title="收益段" />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, marginTop: 4 }}>
            <span style={{ color: "var(--htr-bear)" }}>风险 {fmtY(risk)}/股</span>
            <span style={{ color: "var(--htr-bull)" }}>收益 {fmtY(reward)}/股</span>
          </div>
          <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <V1RRStat label="R:R · 首止盈" value={`1 : ${bal.rr.toFixed(1)}`} accent />
            <V1RRStat label="R:R · 延伸卖出" value={`1 : ${bal.rrS.toFixed(1)}`} />
            <V1RRStat label="均衡档单笔风险" value={fmtY(riskBudget)} />
            <V1RRStat label="止损距现价" value={`${((candidate.price - stop.price) / candidate.price * 100).toFixed(1)}%`} danger />
          </div>
          <div style={{ marginTop: 12, fontSize: 9.5, color: "var(--htr-ink-3)", lineHeight: 1.5, borderTop: "1px solid var(--htr-line-soft)", paddingTop: 8 }}>
            ⚠ advice-only · 测算 = 风险预算的确定性函数，非订单、非建议买入 · 由用户在外部券商手动执行 (Rule 3)。
          </div>
        </div>
      </div>
    </V1Panel>
  );
}

function V1RRStat({ label, value, accent, danger }) {
  const c = danger ? "var(--htr-bear)" : accent ? "var(--htr-accent)" : "var(--htr-ink)";
  return (
    <div style={{ padding: "7px 9px", background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", borderRadius: 6 }}>
      <div className="htr-num" style={{ fontSize: 15, fontWeight: 800, color: c }}>{value}</div>
      <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", marginTop: 2 }}>{label}</div>
    </div>
  );
}

window.V1ProTerminal = V1ProTerminal;
