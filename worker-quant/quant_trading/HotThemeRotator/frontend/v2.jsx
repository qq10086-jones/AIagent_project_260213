// V2 — 研究报告编辑风 (Editorial Research Brief)
// Two Sigma 内部研究备忘录的感觉: 衬线头题、宽松行距、纸面感、分隔线、章节式排版
// 单标的占主版面, 价格阶梯作为侧边图表, 市场温度作为页脚 strip

function V2EditorialBrief() {
  const data = window.HTR_DATA;
  const top = data.candidates[0];
  // Q5 fix — useTickingPrice was synthetic jitter, not a live feed. Use the
  // real close until a real intraday_quotes adapter is wired.
  const livePrice = top.price;
  const others = data.candidates.slice(1);

  return (
    <div className="htr" style={{
      width: "100%", height: "100%", background: "var(--htr-bg)",
      display: "flex", justifyContent: "center", alignItems: "flex-start",
    }}>
      <div style={{
        width: "100%", maxWidth: 1380, minHeight: "100%",
        background: "var(--htr-surface)",
        borderLeft: "1px solid var(--htr-line)",
        borderRight: "1px solid var(--htr-line)",
        padding: "26px 44px 24px",
        display: "flex", flexDirection: "column", gap: 20,
      }}>
        {/* Masthead */}
        <V2Masthead asof={data.meta.asof} tradeDate={data.meta.tradeDate} />

        {/* Section A: today's lead */}
        <SectionLabel num="A" title="今日主线 · Lead" sub="news-catalysed semiconductor day" />

        <FeatureHero candidate={top} livePrice={livePrice} />

        <div style={{
          display: "grid", gridTemplateColumns: "1fr 280px",
          gap: 32, paddingTop: 8,
        }}>
          {/* Left: chart + reason/risk in editorial prose */}
          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            <ChartCard candidate={top} />
            <ProseTwoCol candidate={top} />
          </div>

          {/* Right: vertical price ladder */}
          <PriceLadderSidebar candidate={top} livePrice={livePrice} />
        </div>

        {/* Section B: market temperatures */}
        <SectionLabel num="B" title="多市场温度 · External Factors" sub="外部温度因子驱动今日相对强度" />
        <MarketStrip markets={data.markets} />

        {/* Section C: other candidates */}
        <SectionLabel num="C" title="本日候选清单 · Watchlist" sub="按未校准研究分排序" />
        <WatchlistTable candidates={others} />

        {/* Section D: news catalyst + decision log
            P8-16 Cycle 1 — removed §10 GateFlow here; moved to global nav chip. */}
        <SectionLabel num="D" title="新闻催化与决策日志 · Catalyst & Decision Log" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, paddingBottom: 8 }}>
          <V2SurfacePane title="新闻催化时间线">
            <NewsTimeline items={data.newsTimeline} max={5} compact />
          </V2SurfacePane>
          <V2DecisionLogPane entries={data.decisionLog} />
        </div>

        <V2Footer />
      </div>
    </div>
  );
}

function V2Masthead({ asof, tradeDate }) {
  return (
    <div>
      <div style={{
        display: "flex", alignItems: "flex-end", justifyContent: "space-between",
        paddingBottom: 10, borderBottom: "1.5px solid var(--htr-ink)",
      }}>
        <div>
          <div className="htr-eyebrow" style={{ marginBottom: 4 }}>HotThemeRotator · Daily Research Brief</div>
          <div className="htr-serif" style={{ fontSize: 32, fontWeight: 700, letterSpacing: "-0.01em", lineHeight: 1.05 }}>
            今日机会中心 <span style={{ color: "var(--htr-ink-3)", fontWeight: 400 }}>· Opportunity Memo</span>
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="htr-mono" style={{ fontSize: 11, color: "var(--htr-ink-2)" }}>{asof}</div>
          <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginTop: 2 }}>
            ISSUE · <span className="htr-mono">{tradeDate}</span> · vol. 4 issue 113
          </div>
          <div style={{ marginTop: 6 }}>
            <CalibrationBadge />
          </div>
        </div>
      </div>
      <div style={{
        display: "flex", gap: 22, paddingTop: 10,
        fontSize: 11, color: "var(--htr-ink-3)",
      }}>
        <span>主市场: <strong style={{ color: "var(--htr-ink)" }}>日本</strong></span>
        <span>外部温度: <strong style={{ color: "var(--htr-ink)" }}>A股 · 美股 · USD/JPY · SOX</strong></span>
        <span>研究模式: <strong style={{ color: "var(--htr-ink)" }}>不自动下单</strong></span>
        <span style={{ flex: 1 }} />
        <span>建议刷新 <strong className="htr-mono" style={{ color: "var(--htr-ink)" }}>120s</strong></span>
      </div>
    </div>
  );
}

function SectionLabel({ num, title, sub }) {
  return (
    <div style={{
      display: "flex", alignItems: "baseline", gap: 14,
      paddingBottom: 8, borderBottom: "1px solid var(--htr-line)",
    }}>
      <span style={{
        fontFamily: "var(--htr-font-mono)", fontSize: 10.5, fontWeight: 700,
        color: "var(--htr-accent)", letterSpacing: "0.06em",
        background: "var(--htr-accent-soft)", padding: "2px 7px", borderRadius: 2,
      }}>§ {num}</span>
      <h3 className="htr-serif" style={{ margin: 0, fontSize: 19, fontWeight: 700, letterSpacing: "-0.005em" }}>
        {title}
      </h3>
      {sub && (
        <span style={{ fontSize: 11.5, color: "var(--htr-ink-3)", fontStyle: "italic" }}>— {sub}</span>
      )}
    </div>
  );
}

function FeatureHero({ candidate, livePrice }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", gap: 28, alignItems: "center" }}>
      <div>
        <div className="htr-eyebrow" style={{ marginBottom: 6 }}>SYMBOL · LEAD</div>
        <div className="htr-mono" style={{ fontSize: 30, fontWeight: 800, lineHeight: 1, letterSpacing: "-0.02em" }}>
          {candidate.symbol}
        </div>
        <div className="htr-serif" style={{ fontSize: 16, fontWeight: 500, marginTop: 6 }}>
          {candidate.nameJa}
        </div>
        <div style={{ fontSize: 12, color: "var(--htr-ink-3)" }}>
          {candidate.nameCn} · {candidate.theme}
        </div>
      </div>

      <div className="htr-serif" style={{
        fontSize: 22, lineHeight: 1.45, color: "var(--htr-ink)",
        borderLeft: "3px solid var(--htr-accent)", paddingLeft: 18,
        textWrap: "pretty",
      }}>
        “{candidate.one_liner}”
      </div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8, minWidth: 160 }}>
        <div style={{ textAlign: "right" }}>
          <div className="htr-mono" style={{ fontSize: 28, fontWeight: 700 }}>
            <AnimatedPrice value={livePrice} decimals={0} />
          </div>
          <div className={"htr-num " + changeClass(candidate.chg)} style={{ fontSize: 13, fontWeight: 700, marginTop: -2 }}>
            {arrow(candidate.chg)} {fmtPct(candidate.chg)} <span style={{ color: "var(--htr-ink-3)", fontWeight: 400 }}>· 今日</span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <BigStat label="研究分" value={candidate.score} />
          <BigStat label="主题热" value="86°" accent />
        </div>
      </div>
    </div>
  );
}

function BigStat({ label, value, accent }) {
  return (
    <div style={{
      padding: "8px 12px", border: "1px solid var(--htr-line)",
      background: accent ? "var(--htr-accent-soft)" : "var(--htr-surface-2)",
      borderRadius: 4, textAlign: "center", minWidth: 72,
    }}>
      <div className="htr-num" style={{
        fontSize: 18, fontWeight: 700,
        color: accent ? "var(--htr-accent)" : "var(--htr-ink)",
      }}>{value}</div>
      <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{label}</div>
    </div>
  );
}

function ChartCard({ candidate }) {
  const data = window.HTR_DATA;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
        <CaptionTitle>价格走势 · 40 sessions</CaptionTitle>
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)", fontStyle: "italic" }}>
          fig. 1 · 黑线为现价 · 蓝/绿/红 阶梯为研究档位
        </span>
      </div>
      <div style={{
        background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)",
        padding: "10px 8px 4px", borderRadius: 3,
      }}>
        <KLineChart data={data.kline} ladder={candidate.ladder} width={740} height={310}
                    padding={{ top: 14, right: 130, bottom: 22, left: 10 }} />
      </div>
    </div>
  );
}

function ProseTwoCol({ candidate }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28 }}>
      <ProseCol label="为什么它排前面" body={candidate.reason} accent="var(--htr-bull)" />
      <ProseCol label="先看清楚的风险" body={candidate.risk}   accent="var(--htr-bear)" extra={candidate.dataQuality} />
    </div>
  );
}

function ProseCol({ label, body, accent, extra }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ width: 3, height: 13, background: accent, borderRadius: 1 }} />
        <span style={{ fontSize: 11.5, fontWeight: 700, letterSpacing: "0.04em" }}>{label}</span>
      </div>
      <p className="htr-serif" style={{
        margin: 0, fontSize: 13.5, lineHeight: 1.65, color: "var(--htr-ink)",
        textWrap: "pretty",
      }}>{body}</p>
      {extra && (
        <p style={{ margin: "10px 0 0", fontSize: 10.5, color: "var(--htr-ink-3)", fontStyle: "italic" }}>
          {extra}
        </p>
      )}
    </div>
  );
}

function PriceLadderSidebar({ candidate, livePrice }) {
  return (
    <div style={{
      borderLeft: "1px solid var(--htr-line)", paddingLeft: 20,
    }}>
      <CaptionTitle>七档价格阶梯</CaptionTitle>
      <div style={{ marginTop: 8 }}>
        <VerticalLadder ladder={candidate.ladder} currentPrice={livePrice} height={420} />
      </div>
      <div style={{
        marginTop: 14, padding: "10px 12px", background: "var(--htr-surface-2)",
        border: "1px solid var(--htr-line)", borderRadius: 3, fontSize: 11, lineHeight: 1.55,
      }}>
        <div style={{ fontWeight: 700, marginBottom: 5 }}>操作摘要</div>
        <div style={{ color: "var(--htr-ink-2)" }}>
          均衡档介入 ¥29,872；首止盈 ¥31,180；止损 ¥29,051。
          风险/收益 <strong>1 : 2.4</strong>。
        </div>
      </div>
    </div>
  );
}

function CaptionTitle({ children }) {
  return (
    <div style={{
      fontSize: 11, fontWeight: 700, letterSpacing: "0.08em",
      color: "var(--htr-ink-2)", textTransform: "uppercase",
      paddingBottom: 5, borderBottom: "1px solid var(--htr-line-soft)",
    }}>
      {children}
    </div>
  );
}

function MarketStrip({ markets }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 12 }}>
      {markets.map((m) => <MarketTempCell key={m.id} m={m} variant="tile" />)}
    </div>
  );
}

function WatchlistTable({ candidates }) {
  return (
    // Q3 fix — was missing explicit background; body rows looked transparent
    // against the page bg while the header (surface-2 off-white) stood out.
    // Now the whole table consistently sits on white surface like other panels.
    <div style={{
      border: "1px solid var(--htr-line)",
      background: "var(--htr-surface)",
      borderRadius: 4,
    }}>
      <div style={{
        display: "grid", gridTemplateColumns: "40px 100px 1fr 110px 100px 80px 60px",
        gap: 10, padding: "7px 14px", background: "var(--htr-surface-2)",
        borderBottom: "1px solid var(--htr-line)",
        fontSize: 10, fontWeight: 700, color: "var(--htr-ink-3)", letterSpacing: "0.08em",
        textTransform: "uppercase",
      }}>
        <div>#</div><div>SYMBOL</div><div>THESIS</div>
        <div style={{ textAlign: "right" }}>现价</div>
        <div style={{ textAlign: "right" }}>日内</div>
        <div style={{ textAlign: "right" }}>研究分</div>
        <div></div>
      </div>
      {candidates.map((c) => (
        <div key={c.symbol} style={{
          display: "grid", gridTemplateColumns: "40px 100px 1fr 110px 100px 80px 60px",
          gap: 10, padding: "9px 14px", alignItems: "center",
          borderBottom: "1px solid var(--htr-line-soft)",
          fontSize: "var(--htr-fs-sm)",
        }}>
          <div className="htr-mono" style={{ color: "var(--htr-ink-3)" }}>#{String(c.rank).padStart(2, "0")}</div>
          <div>
            <div className="htr-mono" style={{ fontWeight: 700 }}>{c.symbol}</div>
            <div style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{c.nameCn}</div>
          </div>
          <div className="htr-serif" style={{ fontSize: 12.5, color: "var(--htr-ink-2)", lineHeight: 1.4 }}>
            {c.one_liner}
          </div>
          <div className="htr-num" style={{ textAlign: "right", fontWeight: 600 }}>
            {fmtPrice(c.price, c.price > 1000 ? 0 : 2)}
          </div>
          <div className={"htr-num " + changeClass(c.chg)} style={{ textAlign: "right", fontWeight: 600 }}>
            {arrow(c.chg)} {fmtPct(c.chg)}
          </div>
          <div className="htr-num" style={{ textAlign: "right", fontSize: 13, fontWeight: 700 }}>{c.score}</div>
          <ScoreBar value={c.score} />
        </div>
      ))}
    </div>
  );
}

function V2SurfacePane({ title, children }) {
  return (
    <div style={{
      background: "var(--htr-surface)",
      border: "1px solid var(--htr-line)",
      borderRadius: 4,
      minHeight: 220,
      padding: "12px 14px",
    }}>
      <CaptionTitle>{title}</CaptionTitle>
      <div style={{ paddingTop: 8 }}>
        {children}
      </div>
    </div>
  );
}

function V2DecisionLogPane({ entries }) {
  const hasEntries = entries && entries.length > 0;
  return (
    <V2SurfacePane title="§8.6 决策日志（最近）">
      {hasEntries ? (
        <DecisionLog entries={entries} max={5} />
      ) : (
        <div style={{
          minHeight: 146,
          display: "flex", flexDirection: "column", justifyContent: "center",
          border: "1px dashed var(--htr-line)",
          background: "var(--htr-surface-2)",
          borderRadius: 4,
          padding: "18px 20px",
        }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--htr-ink)" }}>
            暂无 §8.6 决策日志
          </div>
          <div style={{ marginTop: 8, fontSize: 11.5, lineHeight: 1.6, color: "var(--htr-ink-3)" }}>
            候选预测写入 reports/predictions 后会显示最近记录。当前仍是研究界面；空状态不代表校准完成，也不开放任何执行动作。
          </div>
        </div>
      )}
    </V2SurfacePane>
  );
}

function V2Footer() {
  return (
    <div style={{
      paddingTop: 14, marginTop: 8, borderTop: "1.5px solid var(--htr-ink)",
      display: "flex", justifyContent: "space-between",
      fontSize: 10, color: "var(--htr-ink-3)",
    }}>
      <span>本文件仅为研究决策辅助，不构成投资建议。所有分数 <strong>uncalibrated_research_score</strong>，不可解读为真实胜率。</span>
      <span className="htr-mono">HTR · build 0.7.4 · {window.HTR_DATA.meta.tradeDate}</span>
    </div>
  );
}

window.V2EditorialBrief = V2EditorialBrief;
