// V1 — 三栏专业终端 (Pro Terminal)
// 左栏：市场温度计 + 主题热力 / 中央：K线+七档价格阶梯（贴右边）+ 操作区 / 右栏：新闻+决策日志
// 底部：§10 八阶门槛 horizontal flow

function V1ProTerminal() {
  const data = window.HTR_DATA;
  // P8-18 — selected symbol comes from localStorage; default to top1.
  const defaultSymbol = data.candidates[0]?.symbol || "";
  const [selectedSymbol, setSelectedSymbol] = useSelectedSymbol(defaultSymbol);
  // Resolve the selected candidate; fall back to top1 if the persisted symbol
  // isn't in today's candidate set (e.g., user previously selected a holding).
  const selectedCandidate = data.candidates.find((c) => c.symbol === selectedSymbol) || data.candidates[0];
  const top = selectedCandidate;
  const livePrice = top.price;
  // Live K-line for the selected symbol. The dashboard fallback only belongs
  // to the initial top symbol; using it for another symbol would mislabel the
  // chart if /api/symbol/* is unavailable.
  const fallbackKline = top.symbol === defaultSymbol ? data.kline : [];
  const kline = useSymbolKline(top.symbol, { sessions: 252, fallback: fallbackKline });

  return (
    <div className="htr" style={{
      width: "100%", height: "100%", background: "var(--htr-bg)",
      // P8-16 Cycle 1 — bottom gate strip removed; §10 now in app nav chip.
      // Middle row gets the freed 70px.
      display: "grid", gridTemplateRows: "44px 1fr",
      gridTemplateColumns: "minmax(0, 1fr)",
      padding: "12px 14px", gap: 10,
    }}>
      {/* ───── Top bar ───── */}
      <V1TopBar asof={data.meta.asof} refresh={data.meta.refreshLabel} />

      {/* ───── Main 3-column ───── */}
      <div style={{ display: "grid", gridTemplateColumns: "278px 1fr 360px", gap: 10, minHeight: 0 }}>

        {/* Left rail */}
        <div style={{ display: "grid", gridTemplateRows: "auto 1fr", gap: 10, minHeight: 0 }}>
          <Panel title={<span><Term k="市场温度">多市场温度</Term></span>} sub="External Temperature">
            <div style={{ background: "var(--htr-surface)" }}>
              {data.markets.map((m) => <MarketTempCell key={m.id} m={m} variant="row" />)}
            </div>
          </Panel>
          <Panel title={<Term>主题热力</Term>} sub="Theme Heat · 今日">
            <div style={{ padding: "6px 12px 10px" }}>
              <ThemeHeatBars themes={data.themes} />
            </div>
          </Panel>
        </div>

        {/* Center — hero + chart + action */}
        <div style={{ display: "grid", gridTemplateRows: "auto 1fr auto", gap: 10, minHeight: 0 }}>
          <HeroHeader candidate={top} livePrice={livePrice} />

          <V1KLineLadderPanel data={data} top={top} chartLadder={top.ladder} klineBars={kline.bars} klineLoading={kline.loading} klineError={kline.error} />

          <ActionZone candidate={top} livePrice={livePrice} />
        </div>

        {/* Right rail */}
        <div style={{ display: "grid", gridTemplateRows: "auto auto 1fr auto", gap: 10, minHeight: 0 }}>
          <Panel title="新闻催化" sub="News Catalyst · 6h">
            <div style={{ padding: "4px 12px 8px", maxHeight: 240, overflow: "auto" }}>
              <NewsTimeline items={data.newsTimeline} max={5} compact />
            </div>
          </Panel>
          <V1DailyCockpitPanel cockpit={data.dailyCockpit} />
          <Panel title="候选清单" sub={`点击切换 · 当前 ${top.symbol}`}>
            <div style={{ background: "var(--htr-surface)", overflow: "auto" }}>
              {data.candidates.map((c) => (
                <CandidateRowMini
                  key={c.symbol}
                  c={c}
                  active={c.symbol === top.symbol}
                  onClick={() => setSelectedSymbol(c.symbol)}
                />
              ))}
            </div>
          </Panel>
          <Panel title="决策日志 · §8.6" sub={data.meta.tradeDate}>
            <div style={{ overflow: "auto", maxHeight: 160 }}>
              <DecisionLog entries={data.decisionLog} max={4} />
            </div>
          </Panel>
        </div>
      </div>

      {/* P8-16 Cycle 1 — §10 gate strip moved to global nav chip. */}
    </div>
  );
}

// ─── helpers used only in V1 ───────────────────────────────────────────────

function V1TopBar({ asof, refresh }) {
  const data = window.HTR_DATA;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "minmax(0,auto) minmax(0,1fr) minmax(0,auto)",
      alignItems: "center", gap: 14, padding: "0 4px", minWidth: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 22, height: 22, borderRadius: 4, background: "var(--htr-accent)",
            color: "var(--htr-accent-ink)", display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 11, fontWeight: 800, letterSpacing: "0.04em",
          }}>HTR</div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.01em" }}>HotThemeRotator</div>
            <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.08em" }}>RESEARCH · v0.7.4 · JST</div>
          </div>
        </div>
        <span className="htr-chip">日股主</span>
        <span className="htr-chip info">A股/美股 温度因子</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 14, fontSize: 11 }}>
        <SessionStrip />
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)", whiteSpace: "nowrap" }}>
          刷新 <strong className="htr-num" style={{ color: "var(--htr-ink)" }}>{refresh}</strong>
        </span>
        <span style={{ fontFamily: "var(--htr-font-mono)", fontSize: 10.5, color: "var(--htr-ink-2)", whiteSpace: "nowrap" }}>{asof}</span>
        <CalibrationBadge inline={false} />
      </div>
    </div>
  );
}

function V1DailyCockpitPanel({ cockpit }) {
  const rows = cockpit?.watchlist || [];
  const summary = cockpit?.summary || {};
  const stage = cockpit ? cockpit.activationStage : "stage_0_pull_only";
  const blocked = cockpit && (cockpit.notificationsInvoked || cockpit.execution?.orders || cockpit.execution?.broker);
  return (
    <Panel
      title="Daily Cockpit"
      sub={stage}
      right={<span className={blocked ? "htr-chip bear" : "htr-chip bull"}>{blocked ? "blocked" : "pull only"}</span>}
    >
      <div style={{ padding: "8px 12px", display: "grid", gap: 7 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6 }}>
          <V1CockpitMetric label="watch" value={summary.watchlistCount ?? rows.length} />
          <V1CockpitMetric label="quote" value={summary.symbolsWithQuotes ?? 0} />
          <V1CockpitMetric label="silent" value={summary.silentQueueCount ?? 0} />
        </div>
        <div style={{ display: "grid", gap: 4, maxHeight: 86, overflow: "auto" }}>
          {rows.slice(0, 4).map((row) => (
            <div key={row.symbol} style={{
              display: "grid", gridTemplateColumns: "58px 1fr auto", gap: 6,
              alignItems: "center", fontSize: 10.5,
            }}>
              <span className="htr-mono" style={{ fontWeight: 700 }}>{row.symbol}</span>
              <span style={{ color: "var(--htr-ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {row.quoteStatus || "unavailable"}
              </span>
              <span className={row.quoteStatus === "ok" ? "htr-chip bull" : "htr-chip warn"}>
                TDnet {row.tdnetCount || 0}
              </span>
            </div>
          ))}
          {rows.length === 0 && (
            <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>watchlist empty</div>
          )}
        </div>
      </div>
    </Panel>
  );
}

function V1CockpitMetric({ label, value }) {
  return (
    <div style={{ background: "var(--htr-surface-2)", border: "1px solid var(--htr-line)", borderRadius: 4, padding: "5px 6px" }}>
      <div className="htr-num" style={{ fontSize: 14, fontWeight: 800 }}>{value}</div>
      <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{label}</div>
    </div>
  );
}

function SessionStrip() {
  const sessions = [
    { label: "Tokyo", state: "OPEN",   t: "09:00–11:30 · 12:30–15:00" },
    { label: "HK",    state: "OPEN",   t: "09:30–16:00" },
    { label: "London",state: "PRE",    t: "16:00 JST 开盘" },
    { label: "NY",    state: "CLOSED", t: "22:30 JST 开盘" },
  ];
  return (
    <div style={{ display: "flex", gap: 14 }}>
      {sessions.map((s) => (
        <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            width: 6, height: 6, borderRadius: "50%",
            background: s.state === "OPEN" ? "var(--htr-bull)" : s.state === "PRE" ? "var(--htr-warn)" : "var(--htr-ink-4)",
          }} />
          <span style={{ fontSize: 10.5, fontWeight: 600 }}>{s.label}</span>
          <span style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{s.state}</span>
        </div>
      ))}
    </div>
  );
}

// P8-17 — K-line panel with a dedicated ladder rail. The chart no longer
// spends right padding on seven text labels; the rail owns ladder readability.
// P8-18 — klineBars + klineLoading flow from the symbol-switch hook in the parent.
function V1KLineLadderPanel({ data, top, chartLadder, klineBars, klineLoading, klineError }) {
  const [boxRef, { width, height }] = useElementSize();
  const ready = width > 240 && height > 200;
  const bars = (klineBars && klineBars.length) ? klineBars : [];
  const statusText = klineLoading ? " - loading" : klineError ? ` - K-line unavailable: ${klineError}` : "";
  return (
    <Panel
      title={<span>价格走势 · <Term>七档阶梯</Term></span>}
      sub={`${bars.length} sessions · ${top.symbol}${klineLoading ? " · 加载中" : ""}`}
      right={<span className="htr-chip">研究模式 · <Term>MA20</Term>/<Term>MA60</Term> · <Term k="52w 高">52w 高低</Term></span>}
    >
      <div style={{
        width: "100%", height: "100%", padding: 6,
        display: "grid", gridTemplateColumns: "minmax(0, 1fr) 184px",
        gap: 8, minHeight: 0,
      }}>
        <div ref={boxRef} style={{
          minWidth: 0, minHeight: 0,
          display: "flex", alignItems: "stretch", justifyContent: "stretch",
        }}>
          {ready && bars.length > 0 && (
            <KLineChart
              key={top.symbol}
              data={bars}
              ladder={null}
              width={width - 12}
              height={height - 12}
              padding={{ top: 18, right: 28, bottom: 22, left: 12 }}
              withVolume withMA with52wLines
            />
          )}
          {ready && bars.length === 0 && (
            <div style={{
              flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
              border: "1px dashed var(--htr-line)", background: "var(--htr-surface-2)",
              color: "var(--htr-ink-3)", fontSize: 12,
            }}>
              {top.symbol} K-line data unavailable
            </div>
          )}
        </div>
        <V1LadderRail ladder={chartLadder} currentPrice={top.price} />
      </div>
    </Panel>
  );
}

function V1LadderRail({ ladder, currentPrice }) {
  const exits = ladder.filter((r) => r.kind.startsWith("exit"));
  const entries = ladder.filter((r) => r.kind.startsWith("entry"));
  const stop = ladder.find((r) => r.kind === "stop");
  return (
    <div style={{
      minHeight: 0, overflow: "auto",
      borderLeft: "1px solid var(--htr-line)",
      padding: "4px 0 4px 8px",
      display: "grid", gridTemplateRows: "auto 1fr auto", gap: 8,
    }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto", alignItems: "baseline", gap: 6 }}>
        <span style={{ fontSize: 10.5, fontWeight: 700, color: "var(--htr-ink-2)" }}>七档阶梯</span>
        <span className="htr-num" style={{ fontSize: 10.5, color: "var(--htr-accent)", fontWeight: 700 }}>
          {fmt(currentPrice, 0)}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5, justifyContent: "center" }}>
        <V1LadderGroup label="卖出区" levels={exits} tone="info" />
        <V1LadderGroup label="买入区" levels={entries} tone="bull" />
      </div>
      {stop && (
        <V1LadderLevel level={stop} tone="bear" compact />
      )}
    </div>
  );
}

function V1LadderGroup({ label, levels, tone }) {
  return (
    <div>
      <div style={{
        fontSize: 9.5, fontWeight: 700, color: "var(--htr-ink-3)",
        letterSpacing: "0.08em", marginBottom: 3,
      }}>
        {label}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {levels.map((level) => (
          <V1LadderLevel key={level.kind} level={level} tone={tone} />
        ))}
      </div>
    </div>
  );
}

function V1LadderLevel({ level, tone, compact = false }) {
  const color = tone === "bull" ? "var(--htr-bull)" : tone === "bear" ? "var(--htr-bear)" : "var(--htr-info)";
  const bg = tone === "bull" ? "var(--htr-bull-bg)" : tone === "bear" ? "var(--htr-bear-bg)" : "var(--htr-info-bg)";
  return (
    <div style={{
      border: `1px solid ${color}`, background: bg, borderRadius: 4,
      padding: compact ? "6px 7px" : "5px 7px",
      display: "grid", gridTemplateColumns: "1fr auto", gap: 6,
      alignItems: "baseline",
    }}>
      <span style={{
        fontSize: 10.5, color, fontWeight: 700,
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
      }}>
        {level.label.replace("买入 · ", "")}
      </span>
      <span className="htr-num" style={{ fontSize: 12, color, fontWeight: 800 }}>
        {fmt(level.price, 0)}
      </span>
      <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)" }}>{level.kind}</span>
      <span className="htr-num" style={{ fontSize: 10, color: "var(--htr-ink-3)", textAlign: "right" }}>
        {window.HTR.fmtPct(level.pct, 1)}
      </span>
    </div>
  );
}

function Panel({ title, sub, right, children, pad = true }) {
  return (
    <div className="htr-card" style={{ display: "flex", flexDirection: "column", overflow: "hidden", minHeight: 0 }}>
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "7px 12px 7px", borderBottom: "1px solid var(--htr-line)",
        background: "var(--htr-surface-2)",
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <div style={{ fontSize: 11.5, fontWeight: 700, color: "var(--htr-ink)" }}>{title}</div>
          {sub && <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{sub}</div>}
        </div>
        {right}
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>{children}</div>
    </div>
  );
}

function HeroHeader({ candidate, livePrice }) {
  return (
    <div className="htr-card" style={{ padding: "14px 18px", display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span className="htr-eyebrow">今日最值得盯 · #01</span>
        <span className="htr-chip warn">{candidate.priority}</span>
        <span className="htr-chip accent">{candidate.theme}</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>
          决策时刻 <strong className="htr-num" style={{ color: "var(--htr-ink)" }}>{candidate.decisionCutoff}</strong>
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto auto auto", alignItems: "baseline", gap: 18 }}>
        <div>
          <div className="htr-mono" style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.01em", color: "var(--htr-ink)" }}>
            {candidate.symbol}
          </div>
          <div style={{ fontSize: 11, color: "var(--htr-ink-3)" }}>{candidate.nameJa} · {candidate.nameCn}</div>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{ fontSize: 30, fontWeight: 700, color: "var(--htr-ink)" }}>
            <AnimatedPrice value={livePrice} decimals={0} />
          </span>
          <span className={"htr-num " + changeClass(candidate.chg)} style={{ fontSize: 14, fontWeight: 700 }}>
            {arrow(candidate.chg)} {fmtPct(candidate.chg)}
          </span>
        </div>
        <MetricInline label="机会分" value={candidate.score} />
        <MetricInline label="风险/收益" value="1 : 2.4" />
        <MetricInline label="主题热力" value={`86°`} accent />
      </div>

      <div style={{
        fontSize: "var(--htr-fs-md)", color: "var(--htr-ink)", lineHeight: 1.5,
        padding: "6px 0 0", borderTop: "1px solid var(--htr-line-soft)", marginTop: 2,
      }}>
        <span style={{ color: "var(--htr-ink-3)", marginRight: 8 }}>判断</span>
        {candidate.one_liner}
      </div>
    </div>
  );
}

function MetricInline({ label, value, accent }) {
  return (
    <div style={{ textAlign: "right" }}>
      <div className="htr-num" style={{
        fontSize: 17, fontWeight: 700,
        color: accent ? "var(--htr-accent)" : "var(--htr-ink)",
      }}>{value}</div>
      <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.06em" }}>{label}</div>
    </div>
  );
}

function fmt(n, d = 2) { return window.HTR.fmtPrice(n, d); }

function ActionZone({ candidate, livePrice }) {
  const buyR = candidate.ladder.filter((r) => r.kind.startsWith("entry"));
  const sellR = candidate.ladder.filter((r) => r.kind.startsWith("exit"));
  const stop = candidate.ladder.find((r) => r.kind === "stop");
  return (
    <div className="htr-card" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", padding: 0 }}>
      <ActionBlock title="买入区" tone="bull" current={livePrice}>
        {buyR.map((r) => (
          <ActionRow key={r.kind} label={r.label.replace("买入 · ", "")} price={r.price} pct={r.pct} tone="bull" />
        ))}
      </ActionBlock>
      <ActionBlock title="卖出区" tone="info" current={livePrice}>
        {sellR.map((r) => (
          <ActionRow key={r.kind} label={r.label} price={r.price} pct={r.pct} tone="info" />
        ))}
      </ActionBlock>
      <ActionBlock title="风险线" tone="bear" current={livePrice} note="超过止损线 → 平仓">
        <ActionRow label={stop.label} price={stop.price} pct={stop.pct} tone="bear" />
        <div style={{
          marginTop: 6, padding: "6px 8px", background: "var(--htr-bear-bg)",
          borderRadius: 4, fontSize: 10.5, color: "var(--htr-bear)", lineHeight: 1.4,
        }}>
          USD/JPY 跌破 156 取消激进档
        </div>
      </ActionBlock>
    </div>
  );
}

function ActionBlock({ title, tone, children }) {
  const color = tone === "bull" ? "var(--htr-bull)" : tone === "bear" ? "var(--htr-bear)" : "var(--htr-info)";
  return (
    <div style={{
      padding: "10px 14px",
      borderRight: "1px solid var(--htr-line)",
      display: "flex", flexDirection: "column", gap: 5,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
        <span style={{ width: 4, height: 12, background: color, borderRadius: 1 }} />
        <span style={{ fontSize: 11.5, fontWeight: 700, color: "var(--htr-ink)" }}>{title}</span>
      </div>
      {children}
    </div>
  );
}

function ActionRow({ label, price, pct, tone }) {
  const color = tone === "bull" ? "var(--htr-bull)" : tone === "bear" ? "var(--htr-bear)" : "var(--htr-info)";
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "1fr auto auto", alignItems: "baseline", gap: 8,
      padding: "3px 0",
    }}>
      <span style={{ fontSize: 11, color: "var(--htr-ink-2)" }}>{label}</span>
      <span className="htr-num" style={{ fontSize: 13, fontWeight: 700, color: color }}>{fmt(price, 0)}</span>
      <span className="htr-num" style={{ fontSize: 10.5, color: "var(--htr-ink-3)", minWidth: 38, textAlign: "right" }}>
        {window.HTR.fmtPct(pct, 1)}
      </span>
    </div>
  );
}

function CandidateRowMini({ c, active = false, onClick }) {
  // P8-18 — clickable row switches the selected symbol. Rule 11.1 allowed
  // interaction: drill into any candidate. The click only writes localStorage
  // (user-state); nothing hits the decision log.
  const interactive = typeof onClick === "function";
  return (
    <div
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={interactive ? onClick : undefined}
      onKeyDown={interactive ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } } : undefined}
      style={{
        display: "grid", gridTemplateColumns: "20px 64px 1fr 46px 32px",
        gap: 8, alignItems: "center",
        padding: "6px 10px",
        borderBottom: "1px solid var(--htr-line-soft)",
        borderLeft: active ? "3px solid var(--htr-accent)" : "3px solid transparent",
        background: active ? "var(--htr-accent-soft)"
                  : c.rank === 1 ? "var(--htr-bg)" : "transparent",
        fontSize: 11, cursor: interactive ? "pointer" : "default",
        userSelect: "none",
      }}>
      <span style={{ fontFamily: "var(--htr-font-mono)", color: "var(--htr-ink-3)" }}>#{c.rank}</span>
      <span style={{ fontFamily: "var(--htr-font-mono)", fontWeight: 700 }}>{c.symbol}</span>
      <span style={{ color: "var(--htr-ink-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {c.theme}
      </span>
      <span className={"htr-num " + window.HTR.changeClass(c.chg)} style={{ fontSize: 10.5, textAlign: "right" }}>
        {window.HTR.fmtPct(c.chg, 2)}
      </span>
      <span className="htr-num" style={{ fontSize: 11, fontWeight: 700, textAlign: "right" }}>{c.score}</span>
    </div>
  );
}

window.V1ProTerminal = V1ProTerminal;
