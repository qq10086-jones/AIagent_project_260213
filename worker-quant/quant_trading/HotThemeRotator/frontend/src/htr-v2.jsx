// htr-v2.jsx — refined "研究备忘录" (Editorial Research Brief).
// Magazine masthead, serif display, generous leading, section rhythm. Single
// lead symbol owns the page; ladder as a sidebar chart; markets as a footer strip.
// Refinements: responsive chart, larger reading type, explicit advice-only note.

function V2EditorialBrief() {
  const data = window.HTR_DATA;
  // P6-B — selectable lead (shared htr_symbol key); no longer hard-locked to [0].
  const [sel, setSel] = React.useState(() => { try { return localStorage.getItem("htr_symbol") || data.candidates[0].symbol; } catch (e) { return data.candidates[0].symbol; } });
  React.useEffect(() => { try { localStorage.setItem("htr_symbol", sel); } catch (e) {} }, [sel]);
  // Rule 11.9.5 — overlay real per-symbol detail from /api/symbol/{T}/*.
  const top = useEnrichedCandidate(data.candidates.find((c) => c.symbol === sel) || data.candidates[0]);
  const livePrice = top.price;
  const others = data.candidates.filter((c) => c.symbol !== top.symbol);
  return (
    <div className="htr" style={{ width: "100%", minHeight: "100%", background: "var(--htr-bg)", display: "flex", justifyContent: "center", alignItems: "flex-start" }}>
      <div className="v2-page" style={{ width: "100%", maxWidth: 1500, minHeight: "100%", background: "var(--htr-surface)", borderLeft: "1px solid var(--htr-line)", borderRight: "1px solid var(--htr-line)", padding: "30px 52px 28px", display: "flex", flexDirection: "column", gap: 22 }}>
        <V2Masthead asof={data.meta.asof} tradeDate={data.meta.tradeDate} calib={data.meta.calibration} />
        <V2SectionLabel num="A" title="今日主线 · Lead" sub={top.one_liner || top.reason || "按未校准研究分排序的今日 #1 候选"} />
        <V2FeatureHero candidate={top} livePrice={livePrice} />
        <div className="v2-body-grid">
          <div style={{ display: "flex", flexDirection: "column", gap: 20, minWidth: 0 }}>
            <V2ChartCard candidate={top} kline={data.kline} />
            <V2ProseTwoCol candidate={top} />
          </div>
          <V2LadderSidebar candidate={top} livePrice={livePrice} />
        </div>
        {/* P6-F — evidence (why #1) sits directly under the §A lead claim */}
        <V2SectionLabel num="B" title="因子构成与校准 · Factors & Calibration" sub="为什么它排前面 + 诚实的不确定性" />
        <div className="v2-tri-grid">
          <V2Pane title="因子构成"><FactorBody candidate={top} /></V2Pane>
          <V2Pane title="历史命中（原始频率）"><OutcomesBody candidate={top} /></V2Pane>
          <V2Pane title="校准状态 · K-fold"><CalibBody /></V2Pane>
        </div>
        <V2SectionLabel num="C" title="多市场温度 · External Factors" sub="外部温度因子驱动今日相对强度" />
        <div className="v2-market-strip">{data.markets.map((m) => <MarketTempCell key={m.id} m={m} variant="tile" />)}</div>
        <V2SectionLabel num="D" title="本日候选清单 · Watchlist" sub="按未校准研究分排序 · 点击切换主线" />
        <V2WatchlistTable candidates={others} onSelect={setSel} />
        {/* Rule 11.11 — historical-candidate cohort review (shared governed card, four-variant parity). */}
        <V2SectionLabel num="D·复盘" title="历史候选复盘 · Cohort Review" sub="过往整批候选 vs 同期大盘 · 只读 PIT 复盘" />
        <V3CandidateHistoryCard />
        {/* Rule 11.17 — Position Exit Discipline Board (shared governed card, four-variant parity). */}
        <ExitBoardCard />
        {/* Section 17 — Owner Risk Mandate sleeve panel (shared governed card, four-variant parity). */}
        <RiskMandateCard />
        {/* Rule 11.16 — Daily Action Board (shared governed card, four-variant parity). */}
        <ActionBoardCard />
        <EventDeskCard />
        <V2SectionLabel num="E" title="新闻催化与决策日志 · Catalyst & Decision Log" />
        <div className="v2-dual-grid">
          <V2Pane title="新闻催化时间线"><NewsTimeline items={data.newsTimeline} max={6} compact /></V2Pane>
          <V2Pane title="§8.6 决策日志（最近）">
            {data.decisionLog.length ? <DecisionLog entries={data.decisionLog} max={6} /> : (
              <div style={{ minHeight: 146, display: "flex", flexDirection: "column", justifyContent: "center", border: "1px dashed var(--htr-line-2)", background: "var(--htr-surface-2)", borderRadius: 6, padding: "18px 20px" }}>
                <div style={{ fontSize: 13, fontWeight: 700 }}>暂无 §8.6 决策日志</div>
                <div style={{ marginTop: 8, fontSize: 11.5, lineHeight: 1.6, color: "var(--htr-ink-3)" }}>候选预测写入 reports/predictions 后会显示最近记录。空状态不代表校准完成，也不开放任何执行动作。</div>
              </div>
            )}
          </V2Pane>
        </div>
        <V2SectionLabel num="F" title="§10 自动化八阶门槛 · Governance" sub="Gate 8 券商执行硬锁定" />
        <div style={{ paddingBottom: 4 }}><GateFlow gates={data.gates} /></div>
        <V2Footer tradeDate={data.meta.tradeDate} />
      </div>
    </div>
  );
}

function V2Masthead({ asof, tradeDate, calib }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", paddingBottom: 12, borderBottom: "2px solid var(--htr-ink)", gap: 16, flexWrap: "wrap" }}>
        <div>
          <div className="htr-eyebrow" style={{ marginBottom: 5 }}>HotThemeRotator · Daily Research Brief</div>
          <div className="htr-serif" style={{ fontSize: 34, fontWeight: 700, letterSpacing: "-0.01em", lineHeight: 1.05 }}>今日机会中心 <span style={{ color: "var(--htr-ink-3)", fontWeight: 400 }}>· Opportunity Memo</span></div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="htr-mono" style={{ fontSize: 11.5, color: "var(--htr-ink-2)" }}>{asof}</div>
          <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginTop: 2 }}>ISSUE · <span className="htr-mono">{tradeDate}</span></div>
          <div style={{ marginTop: 7, display: "flex", justifyContent: "flex-end" }}><CalibPill sample={calib.sample} /></div>
        </div>
      </div>
      <div style={{ display: "flex", gap: 24, paddingTop: 11, fontSize: 11.5, color: "var(--htr-ink-3)", flexWrap: "wrap" }}>
        <span>主市场: <strong style={{ color: "var(--htr-ink)" }}>日本</strong></span>
        <span>外部温度: <strong style={{ color: "var(--htr-ink)" }}>A股 · 美股 · USD/JPY · SOX</strong></span>
        <span>研究模式: <strong style={{ color: "var(--htr-ink)" }}>不自动下单</strong></span>
        <span style={{ flex: 1 }} />
        <span>建议刷新 <strong className="htr-mono" style={{ color: "var(--htr-ink)" }}>120s</strong></span>
      </div>
    </div>
  );
}

function V2SectionLabel({ num, title, sub }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 14, paddingBottom: 9, borderBottom: "1px solid var(--htr-line)", flexWrap: "wrap" }}>
      <span style={{ fontFamily: "var(--htr-font-mono)", fontSize: 10.5, fontWeight: 700, color: "var(--htr-accent)", letterSpacing: "0.06em", background: "var(--htr-accent-soft)", padding: "2px 8px", borderRadius: 3 }}>§ {num}</span>
      <h3 className="htr-serif" style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>{title}</h3>
      {sub && <span style={{ fontSize: 12, color: "var(--htr-ink-3)", fontStyle: "italic" }}>— {sub}</span>}
    </div>
  );
}

function V2FeatureHero({ candidate, livePrice }) {
  return (
    <div className="v2-hero-grid">
      <div>
        <div className="htr-eyebrow" style={{ marginBottom: 6 }}>SYMBOL · LEAD</div>
        <div className="htr-mono" style={{ fontSize: 31, fontWeight: 800, lineHeight: 1, letterSpacing: "-0.02em" }}>{candidate.symbol}</div>
        <div className="htr-serif" style={{ fontSize: 17, fontWeight: 500, marginTop: 7 }}>{candidate.nameJa || candidate.symbol}</div>
        <div style={{ fontSize: 12.5, color: "var(--htr-ink-3)" }}>{[candidate.nameCn, candidate.theme && candidate.theme !== "screener_v2" ? candidate.theme : null].filter(Boolean).join(" · ") || "未命名 · 未分类主题"}</div>
        <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}><CatalystBadges c={candidate} /></div>
      </div>
      <div className="htr-serif" style={{ fontSize: 22, lineHeight: 1.5, color: "var(--htr-ink)", borderLeft: "3px solid var(--htr-accent)", paddingLeft: 18, textWrap: "pretty" }}>“{candidate.one_liner}”</div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 9, minWidth: 160 }}>
        <div style={{ textAlign: "right" }}>
          <div className="htr-mono" style={{ fontSize: 29, fontWeight: 700 }}><AnimatedPrice value={livePrice} decimals={0} /></div>
          <div className={"htr-num " + HTR.changeClass(candidate.chg)} style={{ fontSize: 13, fontWeight: 700 }}>{HTR.arrow(candidate.chg)} {HTR.fmtPct(candidate.chg)} <span style={{ color: "var(--htr-ink-3)", fontWeight: 400 }}>· 今日</span></div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <V2BigStat label="研究分" value={candidate.score} />
          <V2BigStat label="主题热" value={(() => { const t = ((window.HTR_DATA && window.HTR_DATA.themes) || []).find((x) => x.id === candidate.themeId); return t ? `${t.heat}°` : "—"; })()} accent />
        </div>
      </div>
    </div>
  );
}

function V2BigStat({ label, value, accent }) {
  return (
    <div style={{ padding: "9px 13px", border: "1px solid var(--htr-line)", background: accent ? "var(--htr-accent-soft)" : "var(--htr-surface-2)", borderRadius: 6, textAlign: "center", minWidth: 74 }}>
      <div className="htr-num" style={{ fontSize: 19, fontWeight: 700, color: accent ? "var(--htr-accent)" : "var(--htr-ink)" }}>{value}</div>
      <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{label}</div>
    </div>
  );
}

function V2CaptionTitle({ children }) {
  return <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", color: "var(--htr-ink-2)", textTransform: "uppercase", paddingBottom: 6, borderBottom: "1px solid var(--htr-line-soft)" }}>{children}</div>;
}

function V2ChartCard({ candidate, kline }) {
  const [boxRef, { width }] = useElementSize();
  const H = 320;
  // Rule 11.9.5 — prefer the enriched real overlay (candidate.kline from
  // /api/symbol/{T}/kline) over the boot mock baseline prop, matching V1/V3.
  const bars = (Array.isArray(candidate.kline) && candidate.kline.length) ? candidate.kline : kline;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 9, gap: 8, flexWrap: "wrap" }}>
        <V2CaptionTitle>价格走势 · {bars.length} sessions{candidate._status && candidate._status.kline === "failed" ? " · 示例K线" : ""}</V2CaptionTitle>
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)", fontStyle: "italic" }}>fig. 1 · 蓝/绿/红 阶梯为研究档位（建议价位）</span>
      </div>
      <div ref={boxRef} style={{ background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", padding: "10px 8px 4px", borderRadius: 6, height: H }}>
        {width > 0 && <KLineChart data={bars} ladder={candidate.ladder} width={width - 16} height={H - 14} padding={{ top: 14, right: 120, bottom: 22, left: 10 }} withVolume withMA with52wLines />}
      </div>
    </div>
  );
}

function V2ProseTwoCol({ candidate }) {
  const Col = ({ label, body, accent, extra }) => (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}><span style={{ width: 3, height: 13, background: accent, borderRadius: 1 }} /><span style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.02em" }}>{label}</span></div>
      <p className="htr-serif" style={{ margin: 0, fontSize: 14, lineHeight: 1.7, color: "var(--htr-ink)", textWrap: "pretty" }}>{body}</p>
      {extra && <p style={{ margin: "10px 0 0", fontSize: 11, color: "var(--htr-ink-3)", fontStyle: "italic" }}>{extra}</p>}
    </div>
  );
  return (
    <div className="v2-prose-grid">
      <Col label="为什么它排前面" body={candidate.reason} accent="var(--htr-bull)" />
      <Col label="先看清楚的风险" body={candidate.risk} accent="var(--htr-bear)" extra={candidate.dataQuality} />
    </div>
  );
}

function V2LadderSidebar({ candidate, livePrice }) {
  // derive the action summary from the REAL ladder, not hardcoded 8035.T mock prices
  const L = candidate.ladder || [];
  const ft = (k) => (L.find((r) => r.kind === k) || {}).price;
  const eb = ft("entry_balanced"), e1 = ft("exit_1"), sp = ft("stop");
  const fy = (v) => (v != null ? "¥" + Math.round(v).toLocaleString("en-US") : "—");
  const rr = (eb && e1 && sp && eb > sp) ? `1 : ${((e1 - eb) / (eb - sp)).toFixed(1)}` : "—";
  return (
    <div className="v2-ladder-side" style={{ borderLeft: "1px solid var(--htr-line)", paddingLeft: 22 }}>
      <V2CaptionTitle>七档价格阶梯</V2CaptionTitle>
      <div style={{ marginTop: 10 }}><VerticalLadder ladder={candidate.ladder} currentPrice={livePrice} height={420} /></div>
      <div style={{ marginTop: 14, padding: "11px 13px", background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", borderRadius: 6, fontSize: 11.5, lineHeight: 1.6 }}>
        <div style={{ fontWeight: 700, marginBottom: 5 }}>操作摘要 · 建议参考</div>
        <div style={{ color: "var(--htr-ink-2)" }}>均衡档介入 {fy(eb)}；首止盈 {fy(e1)}；止损参考 {fy(sp)}。风险/收益 <strong>{rr}</strong>。</div>
        <div style={{ marginTop: 7, paddingTop: 7, borderTop: "1px solid var(--htr-line)", fontSize: 10, color: "var(--htr-ink-3)" }}>⚠ advice-only · 由用户在外部券商手动执行 (Rule 3)</div>
      </div>
    </div>
  );
}

function V2WatchlistTable({ candidates, onSelect }) {
  const cols = "40px 100px 1fr 110px 96px 76px 56px";
  return (
    <div style={{ border: "1px solid var(--htr-line)", background: "var(--htr-surface)", borderRadius: 8, overflow: "hidden" }}>
      <div style={{ display: "grid", gridTemplateColumns: cols, gap: 10, padding: "8px 16px", background: "var(--htr-surface-2)", borderBottom: "1px solid var(--htr-line)", fontSize: 10, fontWeight: 700, color: "var(--htr-ink-3)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
        <div>#</div><div>SYMBOL</div><div>THESIS</div><div style={{ textAlign: "right" }}>现价</div><div style={{ textAlign: "right" }}>日内</div><div style={{ textAlign: "right" }}>研究分</div><div></div>
      </div>
      {candidates.map((c) => (
        <div key={c.symbol} role="button" tabIndex={0} onClick={() => onSelect && onSelect(c.symbol)} onKeyDown={(e) => { if (onSelect && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onSelect(c.symbol); } }} title="设为今日主线" style={{ display: "grid", gridTemplateColumns: cols, gap: 10, padding: "10px 16px", alignItems: "center", borderBottom: "1px solid var(--htr-line-soft)", fontSize: 12.5, cursor: onSelect ? "pointer" : "default" }}>
          <div className="htr-mono" style={{ color: "var(--htr-ink-3)" }}>#{String(c.rank).padStart(2, "0")}</div>
          <div><div className="htr-mono" style={{ fontWeight: 700 }}>{c.symbol}</div><div style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{c.nameCn}</div></div>
          <div className="htr-serif" style={{ fontSize: 13, color: "var(--htr-ink-2)", lineHeight: 1.45 }}>{c.one_liner}</div>
          <div className="htr-num" style={{ textAlign: "right", fontWeight: 600 }}>{HTR.fmtPrice(c.price, c.price > 1000 ? 0 : 2)}</div>
          <div className={"htr-num " + HTR.changeClass(c.chg)} style={{ textAlign: "right", fontWeight: 600 }}>{HTR.arrow(c.chg)} {HTR.fmtPct(c.chg)}</div>
          <div className="htr-num" style={{ textAlign: "right", fontSize: 13.5, fontWeight: 700 }}>{c.score}</div>
          <ScoreBar value={c.score} />
        </div>
      ))}
    </div>
  );
}

function V2Pane({ title, children }) {
  return (
    <div style={{ background: "var(--htr-surface)", border: "1px solid var(--htr-line)", borderRadius: 8, minHeight: 220, padding: "13px 15px" }}>
      <V2CaptionTitle>{title}</V2CaptionTitle>
      <div style={{ paddingTop: 8 }}>{children}</div>
    </div>
  );
}

function V2Footer({ tradeDate }) {
  return (
    <div style={{ paddingTop: 15, marginTop: 8, borderTop: "2px solid var(--htr-ink)", display: "flex", justifyContent: "space-between", gap: 16, fontSize: 10.5, color: "var(--htr-ink-3)", flexWrap: "wrap" }}>
      <span>本文件仅为研究决策辅助，不构成投资建议。所有分数 <strong>uncalibrated_research_score</strong>，不可解读为真实胜率。</span>
      <span className="htr-mono">HTR · build 0.7.4 · {tradeDate}</span>
    </div>
  );
}

window.V2EditorialBrief = V2EditorialBrief;
