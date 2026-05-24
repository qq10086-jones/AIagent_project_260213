// V3 — 市场温度主导 (Market-Temperature First)
// 首屏首先回答"今天市场什么感觉" → 然后才是"我应该买什么"
// Hero 是六市场温度网格 (大), 主题热力作为二级,候选作为三级

function V3MarketDashboard() {
  const data = window.HTR_DATA;
  // P8-18 — selected symbol drives the leader card; defaults to top1.
  const defaultSymbol = data.candidates[0]?.symbol || "";
  const [selectedSymbol, setSelectedSymbol] = useSelectedSymbol(defaultSymbol);
  const top = data.candidates.find((c) => c.symbol === selectedSymbol) || data.candidates[0];
  const livePrice = top.price;

  return (
    <div className="htr" style={{
      width: "100%", height: "100%", background: "var(--htr-bg)",
      // P8-16 Cycle 1 — removed bottom gate row; §10 now in app nav chip.
      display: "grid", gridTemplateRows: "44px 240px 1fr",
      gridTemplateColumns: "minmax(0, 1fr)",
      padding: "12px 16px", gap: 12,
    }}>
      <V3TopBar />

      {/* ─── Hero: market temperature mosaic ─── */}
      <V3TemperatureHero markets={data.markets} />

      {/* ─── Mid: positions + themes + leader + watchlist + news ─── */}
      <div style={{ display: "grid", gridTemplateColumns: "340px 1fr 380px", gap: 12, minHeight: 0 }}>
        {/* Left column: positions (P8-10 real Project_optimized data) + theme heat */}
        <div style={{ display: "grid", gridTemplateRows: "auto 1fr", gap: 12, minHeight: 0 }}>
          <V3PositionsCard positions={data.positions} />
          <div className="htr-card" style={{ display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
            <CardHead title={<Term>主题热力</Term>} sub="Theme Heat" right={<span className="htr-chip">六板块</span>} />
            <div style={{ padding: "10px 14px 4px" }}>
              <ThemeBubbleMap themes={data.themes} />
            </div>
            <hr className="htr-divider" />
            <div style={{ padding: "10px 14px" }}>
              <div className="htr-eyebrow" style={{ marginBottom: 8 }}>主题排序</div>
              <ThemeHeatBars themes={data.themes} />
            </div>
          </div>
        </div>

        {/* Center: leader + watchlist */}
        <div style={{ display: "grid", gridTemplateRows: "auto 1fr", gap: 12, minHeight: 0 }}>
          <V3LeaderCard candidate={top} livePrice={livePrice} />

          <div className="htr-card" style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
            <CardHead title="候选清单" sub={`点击切换 · 当前 ${top.symbol}`} />
            <div style={{
              display: "grid", gridTemplateColumns: "50px 100px 1fr 90px 90px 80px 64px",
              gap: 10, padding: "6px 12px",
              fontSize: 10, color: "var(--htr-ink-3)", letterSpacing: "0.08em", textTransform: "uppercase",
              borderBottom: "1px solid var(--htr-line)",
            }}>
              <div>#</div><div>Symbol</div><div>Thesis</div>
              <div style={{ textAlign: "right" }}>价格</div>
              <div style={{ textAlign: "right" }}>日内</div>
              <div style={{ textAlign: "right" }}>研究分</div>
              <div></div>
            </div>
            <div style={{ flex: 1, overflow: "auto" }}>
              {data.candidates.map((c) => (
                <CandidateRow
                  key={c.symbol}
                  c={c}
                  dense
                  active={c.symbol === top.symbol}
                  onClick={() => setSelectedSymbol(c.symbol)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Right: news + decision log */}
        <div style={{ display: "grid", gridTemplateRows: "1fr 1fr", gap: 12, minHeight: 0 }}>
          <div className="htr-card" style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
            <CardHead title={<Term>新闻催化</Term>} sub="6h" right={<span className="htr-chip warn">2 high</span>} />
            <div style={{ padding: "6px 14px", overflow: "auto", flex: 1 }}>
              <NewsTimeline items={data.newsTimeline} max={6} compact />
            </div>
          </div>
          <div className="htr-card" style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
            <CardHead title={<span><Term>决策日志</Term> · <Term>§8.6</Term></span>} sub={data.meta.tradeDate} />
            <div style={{ overflow: "auto", flex: 1 }}>
              <DecisionLog entries={data.decisionLog} max={6} />
            </div>
          </div>
        </div>
      </div>

      {/* P8-16 Cycle 1 — bottom gate strip removed; §10 now in app nav chip. */}
    </div>
  );
}

function V3TopBar() {
  const data = window.HTR_DATA;
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "minmax(0,auto) minmax(0,1fr) minmax(0,auto)",
      alignItems: "center", gap: 14, minWidth: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{
          width: 22, height: 22, borderRadius: 4, background: "var(--htr-accent)",
          color: "var(--htr-accent-ink)", display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 11, fontWeight: 800,
        }}>HTR</div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700 }}>市场温度仪表盘</div>
          <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>MARKET TEMPERATURE · JST</div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, fontSize: 11, minWidth: 0, overflow: "hidden" }}>
        <MoodChip label="日股" mood="warm"  delta="+1.2%" />
        <MoodChip label="美股" mood="hot"   delta="SOX +1.8%" />
        <MoodChip label="A股"  mood="cool"  delta="-0.3%" />
        <MoodChip label="FX"   mood="risk"  delta="156.18" />
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-2)", whiteSpace: "nowrap" }}>{data.meta.asof}</span>
        <CalibrationBadge />
      </div>
    </div>
  );
}

function MoodChip({ label, mood, delta }) {
  const map = {
    hot:  { bg: "var(--htr-bear-bg)", fg: "var(--htr-bear)", t: "热" },
    warm: { bg: "var(--htr-warn-bg)", fg: "var(--htr-warn)", t: "暖" },
    cool: { bg: "var(--htr-info-bg)", fg: "var(--htr-info)", t: "凉" },
    risk: { bg: "var(--htr-surface-3)", fg: "var(--htr-ink-2)", t: "·" },
  };
  const c = map[mood];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "3px 9px", borderRadius: 999, background: c.bg, border: `1px solid ${c.fg}33`,
      fontSize: 11,
    }}>
      <span style={{ fontWeight: 700, color: "var(--htr-ink)" }}>{label}</span>
      <span style={{ fontWeight: 700, color: c.fg }}>{c.t}</span>
      <span className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-2)" }}>{delta}</span>
    </span>
  );
}

function V3TemperatureHero({ markets }) {
  return (
    <div className="htr-card" style={{ padding: "12px 16px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <span className="htr-eyebrow">市场温度计 · 今天市场什么感觉</span>
          <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>
            温度 = 多因子合成 (价格动量 · 量能 · 板块共振 · 隐含波动率)
          </span>
        </div>
        <div style={{ display: "flex", gap: 8, fontSize: 10, color: "var(--htr-ink-3)" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <i style={{ width: 8, height: 8, background: "var(--htr-bear)", borderRadius: 2 }} /> ≥ 70 过热
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <i style={{ width: 8, height: 8, background: "var(--htr-warn)", borderRadius: 2 }} /> 50–69 升温
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <i style={{ width: 8, height: 8, background: "var(--htr-info)", borderRadius: 2 }} /> 30–49 平温
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <i style={{ width: 8, height: 8, background: "var(--htr-ink-3)", borderRadius: 2 }} /> &lt; 30 冷
          </span>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10 }}>
        {markets.map((m) => <V3MarketTile key={m.id} m={m} />)}
      </div>
    </div>
  );
}

function V3MarketTile({ m }) {
  const hot = m.temp >= 70, warm = m.temp >= 50, mild = m.temp >= 30;
  const accent = hot ? "var(--htr-bear)" : warm ? "var(--htr-warn)" : mild ? "var(--htr-info)" : "var(--htr-ink-3)";
  const bg     = hot ? "var(--htr-bear-bg)" : warm ? "var(--htr-warn-bg)" : mild ? "var(--htr-info-bg)" : "var(--htr-surface-3)";
  return (
    <div style={{
      background: bg, border: `1px solid ${accent}33`,
      borderRadius: 8, padding: "10px 12px",
      display: "flex", flexDirection: "column", gap: 6, position: "relative", overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", top: 0, right: 0, padding: "2px 6px",
        background: accent, color: "white", fontSize: 9, fontWeight: 700,
        borderBottomLeftRadius: 6, letterSpacing: "0.04em",
      }}>{m.state}</div>
      <div>
        <div style={{ fontSize: 12, fontWeight: 700 }}>{m.label}</div>
        <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", marginTop: 1 }}>{m.sub}</div>
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between" }}>
        <div>
          <div className="htr-num" style={{ fontSize: 30, fontWeight: 800, color: accent, lineHeight: 1 }}>
            {m.temp}<span style={{ fontSize: 14, marginLeft: 1 }}>°</span>
          </div>
          <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em", marginTop: 2 }}>TEMPERATURE</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="htr-num" style={{ fontSize: 13, fontWeight: 700 }}>
            {fmtPrice(m.price, m.id === "USDJPY" ? 2 : 0)}
          </div>
          <div className={"htr-num " + changeClass(m.chg)} style={{ fontSize: 11, fontWeight: 600 }}>
            {arrow(m.chg)} {fmtPct(m.chg)}
          </div>
        </div>
      </div>
      <Sparkline data={m.spark} width={undefined} height={28}
                 stroke={accent} fill={`color-mix(in oklab, ${accent} 18%, transparent)`} />
    </div>
  );
}

function ThemeBubbleMap({ themes }) {
  // Treemap-like: scale by heat
  const total = themes.reduce((a, b) => a + b.heat, 0);
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {themes.map((t, i) => {
        const w = Math.max(20, (t.heat / total) * 100);
        const accent = t.heat >= 70 ? "var(--htr-bear)" : t.heat >= 50 ? "var(--htr-warn)" : "var(--htr-info)";
        const bg     = t.heat >= 70 ? "var(--htr-bear-bg)" : t.heat >= 50 ? "var(--htr-warn-bg)" : "var(--htr-info-bg)";
        return (
          <div key={t.id} style={{
            flexBasis: `calc(${w}% - 6px)`,
            background: bg, border: `1px solid ${accent}33`,
            borderRadius: 5, padding: "10px 10px 8px",
            display: "flex", flexDirection: "column", gap: 4,
            minHeight: t.heat >= 70 ? 86 : 64,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700 }}>{t.label}</div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
              <span className="htr-num" style={{ fontSize: 18, fontWeight: 800, color: accent, lineHeight: 1 }}>{t.heat}°</span>
              <span className={"htr-num " + changeClass(t.mom)} style={{ fontSize: 10 }}>
                {arrow(t.mom)} {Math.abs(t.mom).toFixed(1)}
              </span>
            </div>
            <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)" }}>
              {t.leaders.slice(0, 2).join(" · ")}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function V3LeaderCard({ candidate, livePrice }) {
  return (
    <div className="htr-card" style={{
      padding: "14px 18px",
      background: "linear-gradient(135deg, var(--htr-accent-soft) 0%, var(--htr-surface) 50%)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <span className="htr-eyebrow" style={{ color: "var(--htr-accent)" }}>主题领涨 → 候选 LEADER</span>
        <span className="htr-chip accent">{candidate.theme}</span>
        <span className="htr-chip warn">{candidate.priority}</span>
        <span style={{ flex: 1 }} />
        <CalibrationBadge inline />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "auto minmax(0, 1fr) 220px", gap: 24, alignItems: "center" }}>
        <div>
          <div className="htr-mono" style={{ fontSize: 28, fontWeight: 800 }}>{candidate.symbol}</div>
          <div style={{ fontSize: 12, color: "var(--htr-ink-3)", marginTop: 2 }}>
            {candidate.nameJa} · {candidate.nameCn}
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 8 }}>
            <span className="htr-num" style={{ fontSize: 28, fontWeight: 700 }}>
              <AnimatedPrice value={livePrice} decimals={0} />
            </span>
            <span className={"htr-num " + changeClass(candidate.chg)} style={{ fontSize: 13, fontWeight: 700 }}>
              {arrow(candidate.chg)} {fmtPct(candidate.chg)}
            </span>
          </div>
        </div>

        <div>
          <div style={{ fontSize: 13.5, lineHeight: 1.5, color: "var(--htr-ink)", marginBottom: 10 }}>
            {candidate.one_liner}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 6 }}>
            <MiniStat label="研究分" value={candidate.score} accent />
            <MiniStat label="买入·均衡" value={fmtPrice(candidate.ladder.find(r=>r.kind==="entry_balanced").price, 0)} />
            <MiniStat label="首止盈" value={fmtPrice(candidate.ladder.find(r=>r.kind==="exit_1").price, 0)} />
            <MiniStat label="止损" value={fmtPrice(candidate.ladder.find(r=>r.kind==="stop").price, 0)} danger />
          </div>
        </div>

        <V3LadderMini ladder={candidate.ladder} currentPrice={livePrice} />
      </div>
    </div>
  );
}

function MiniStat({ label, value, accent, danger }) {
  return (
    <div style={{
      padding: "5px 7px", background: "var(--htr-surface)",
      border: "1px solid var(--htr-line)", borderRadius: 4,
      borderLeft: `3px solid ${danger ? "var(--htr-bear)" : accent ? "var(--htr-accent)" : "var(--htr-line-2)"}`,
      minWidth: 0,
    }}>
      <div className="htr-num" style={{
        fontSize: 11.5, fontWeight: 700, whiteSpace: "nowrap",
        color: danger ? "var(--htr-bear)" : accent ? "var(--htr-accent)" : "var(--htr-ink)",
      }}>{value}</div>
      <div style={{ fontSize: 9, color: "var(--htr-ink-3)", letterSpacing: "0.04em", whiteSpace: "nowrap" }}>{label}</div>
    </div>
  );
}

function V3LadderMini({ ladder, currentPrice }) {
  const all = [...ladder.map((r) => r.price), currentPrice];
  const min = Math.min(...all), max = Math.max(...all);
  const pad = (max - min) * 0.10;
  const yMin = min - pad, yMax = max + pad;
  // Increased H 130 -> 220 so 7 ladder rows + current price never crowd.
  const H = 220, W = 220;
  const yOf = (p) => 10 + ((yMax - p) / (yMax - yMin)) * (H - 20);
  // Anti-collision: dots stay at true priceY, labels pushed apart.
  const placed = antiCollideLabels(ladder, yOf, { gap: 22, top: 14, bottom: H - 8 });
  return (
    <div style={{
      background: "var(--htr-surface)", border: "1px solid var(--htr-line)",
      borderRadius: 5, padding: "8px 10px",
    }}>
      <div className="htr-eyebrow" style={{ marginBottom: 4 }}>七档阶梯</div>
      <svg width={W} height={H} style={{ display: "block" }}>
        <line x1={28} y1={4} x2={28} y2={H - 4} stroke="var(--htr-line)" strokeWidth="1" />
        {placed.map((r, i) => {
          const isStop = r.kind === "stop";
          const isExit = r.kind.startsWith("exit");
          const c = isStop ? "var(--htr-bear)" : isExit ? "var(--htr-info)" : "var(--htr-bull)";
          return (
            <g key={r.kind}>
              {/* True price tick on the axis */}
              <line x1={20} y1={r.priceY} x2={36} y2={r.priceY} stroke={c} strokeWidth="2" />
              {/* Connector from price tick to label when they diverge */}
              <path d={`M${36},${r.priceY} L${44},${r.labelY}`} stroke={c} strokeWidth="0.6" fill="none" opacity="0.6" />
              <text x={44} y={r.labelY + 3} fontSize="9.5" fill="var(--htr-ink-2)">{r.label}</text>
              <text x={W - 4} y={r.labelY + 3} fontSize="9.5" textAnchor="end" fill={c} fontWeight="700"
                    style={{ fontFamily: "var(--htr-font-mono)" }}>
                {fmtPrice(r.price, 0)}
              </text>
            </g>
          );
        })}
        {/* current */}
        {(() => {
          const y = yOf(currentPrice);
          return (
            <g>
              <line x1={4} y1={y} x2={W - 4} y2={y} stroke="var(--htr-accent)" strokeWidth="1.3" />
              <rect x={4} y={y - 8} width={36} height={16} fill="var(--htr-accent)" rx="2" />
              <text x={22} y={y + 4} fontSize="10" textAnchor="middle" fontWeight="700" fill="#fff">现价</text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}

function CardHead({ title, sub, right }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "7px 14px", borderBottom: "1px solid var(--htr-line)",
      background: "var(--htr-surface-2)",
    }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <div style={{ fontSize: 11.5, fontWeight: 700 }}>{title}</div>
        {sub && <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{sub}</div>}
      </div>
      {right}
    </div>
  );
}

// P8-10 — real portfolio card backed by Project_optimized/paper_trading_account.json
function V3PositionsCard({ positions }) {
  if (!positions || !positions.available) {
    return (
      <div className="htr-card" style={{ padding: "12px 14px" }}>
        <div className="htr-eyebrow" style={{ marginBottom: 4 }}>持仓 · Portfolio</div>
        <div style={{ fontSize: 11, color: "var(--htr-ink-3)", lineHeight: 1.5 }}>
          {positions && positions.error ? `数据未就绪：${positions.error}` : "持仓数据未就绪"}
        </div>
      </div>
    );
  }
  const fmtY = (v, d = 0) => "¥" + Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  return (
    <div className="htr-card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <CardHead
        title="持仓 · Portfolio"
        sub={`${positions.strategy_id || "—"} · ${positions.positions_asof || positions.asof}`}
        right={<span className="htr-chip">{positions.holdings.length} 标的</span>}
      />
      <div style={{ padding: "8px 14px 6px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div>
          <div className="htr-eyebrow">NAV</div>
          <div className="htr-num" style={{ fontSize: 16, fontWeight: 700 }}>{fmtY(positions.nav)}</div>
        </div>
        <div>
          <div className="htr-eyebrow">现金</div>
          <div className="htr-num" style={{ fontSize: 16, fontWeight: 700 }}>{fmtY(positions.cash)}</div>
        </div>
      </div>
      <div style={{ borderTop: "1px solid var(--htr-line)" }}>
        {positions.holdings.length === 0 && (
          <div style={{ padding: "10px 14px", fontSize: 11, color: "var(--htr-ink-3)" }}>
            当前无持仓（仅现金）
          </div>
        )}
        {positions.holdings.map((h) => {
          const isUp = h.unrealized_pnl >= 0;
          const pnlColor = isUp ? "var(--htr-bull)" : "var(--htr-bear)";
          return (
            <div key={h.symbol} style={{
              padding: "8px 14px",
              display: "grid", gridTemplateColumns: "1fr auto", gap: 6, alignItems: "center",
              borderBottom: "1px solid var(--htr-line-soft)",
            }}>
              <div>
                <div className="htr-mono" style={{ fontSize: 12.5, fontWeight: 700 }}>{h.symbol}</div>
                <div style={{ fontSize: 10, color: "var(--htr-ink-3)", marginTop: 1 }}>
                  {h.qty.toFixed(0)} 股 @ 均价 ¥{h.avg_cost.toFixed(0)}
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="htr-num" style={{ fontSize: 12.5, fontWeight: 700 }}>¥{h.market_price.toFixed(0)}</div>
                <div className="htr-num" style={{ fontSize: 10.5, fontWeight: 700, color: pnlColor }}>
                  {isUp ? "+" : ""}{h.unrealized_pnl.toLocaleString("en-US")} ({isUp ? "+" : ""}{h.unrealized_return_pct.toFixed(1)}%)
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

window.V3MarketDashboard = V3MarketDashboard;
