// V4 — 决策日志为脊 (Workflow / Timeline Spine)
// 中心是一条今日时间线: 新闻 → 候选浮现 → 决策动作 → 风险事件 全部按时间穿成一条
// 这一版强调 "今天发生了什么 + 为什么这样判断", 而不是静态状态

function V4WorkflowSpine() {
  const data = window.HTR_DATA;
  const top = data.candidates[0];
  // Q5 fix — useTickingPrice was synthetic jitter, not a live feed. Use the
  // real close until a real intraday_quotes adapter is wired.
  const livePrice = top.price;

  // Merge news + decisions + macro into one unified stream
  const events = useMemo(() => buildEventStream(data), [data]);

  return (
    <div className="htr" style={{
      width: "100%", height: "100%", background: "var(--htr-bg)",
      display: "grid", gridTemplateRows: "44px 1fr",
      gridTemplateColumns: "minmax(0, 1fr)",
      padding: "12px 16px", gap: 10,
    }}>
      <V4Header />

      <div style={{ display: "grid", gridTemplateColumns: "256px 1fr 380px", gap: 12, minHeight: 0 }}>
        {/* Left rail: macro context */}
        <V4LeftRail markets={data.markets} themes={data.themes} />

        {/* Center: unified spine */}
        <div className="htr-card" style={{ display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
          <V4SpineHeader trackedSymbol={top.symbol} score={top.score} />
          <div style={{ flex: 1, overflow: "auto", padding: "8px 20px 18px" }}>
            <Spine events={events} hero={top} livePrice={livePrice} />
          </div>
        </div>

        {/* Right rail: hero card + ladder + governance */}
        <V4RightRail candidate={top} livePrice={livePrice} />
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────

function V4Header() {
  const data = window.HTR_DATA;
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "minmax(0,auto) minmax(0,1fr) minmax(0,auto)",
      alignItems: "center", gap: 14, padding: "0 4px", minWidth: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 22, height: 22, borderRadius: 4, background: "var(--htr-accent)",
          color: "var(--htr-accent-ink)", display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 11, fontWeight: 800,
        }}>HTR</div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700 }}>今日叙事 · Story of {data.meta.tradeDate}</div>
          <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", letterSpacing: "0.08em" }}>
            UNIFIED EVENT STREAM · §8.6 decision log · §10 governance
          </div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, minWidth: 0, overflow: "hidden" }}>
        <V4FilterChip label="新闻"     count="7"  active />
        <V4FilterChip label="候选浮现" count="6"  active />
        <V4FilterChip label="决策动作" count="8"  active />
        <V4FilterChip label="宏观事件" count="3"  active />
        <V4FilterChip label="风险触发" count="0"  />
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-2)", whiteSpace: "nowrap" }}>{data.meta.asof}</span>
        <CalibrationBadge />
      </div>
    </div>
  );
}

function V4FilterChip({ label, count, active }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "3px 9px", borderRadius: 999,
      border: "1px solid " + (active ? "var(--htr-accent)" : "var(--htr-line-2)"),
      background: active ? "var(--htr-accent-soft)" : "var(--htr-surface-2)",
      fontSize: 11, color: active ? "var(--htr-accent)" : "var(--htr-ink-3)",
    }}>
      <span style={{ fontWeight: 600 }}>{label}</span>
      <span className="htr-mono" style={{ fontWeight: 700, fontSize: 10.5 }}>{count}</span>
    </span>
  );
}

// ──────────────────────────────────────────────────────────────────────────

function V4LeftRail({ markets, themes }) {
  return (
    <div style={{ display: "grid", gridTemplateRows: "auto auto 1fr", gap: 10, minHeight: 0 }}>
      <div className="htr-card">
        <CardHead title="多市场温度" sub="External" />
        <div style={{ background: "var(--htr-surface)" }}>
          {markets.slice(0, 6).map((m) => <MarketTempCell key={m.id} m={m} variant="row" />)}
        </div>
      </div>
      <div className="htr-card">
        <CardHead title="主题热力" sub="Theme" />
        <div style={{ padding: "8px 12px 10px" }}>
          <ThemeHeatBars themes={themes} />
        </div>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────

function V4SpineHeader({ trackedSymbol, score }) {
  return (
    <div style={{
      padding: "10px 20px 10px", background: "var(--htr-surface-2)",
      borderBottom: "1px solid var(--htr-line)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
    }}>
      <div>
        <div style={{ fontSize: 13.5, fontWeight: 700 }}>今天发生了什么 · 为什么 {trackedSymbol} 跑出来</div>
        <div style={{ fontSize: 11, color: "var(--htr-ink-3)", marginTop: 1 }}>
          按时间倒序 · 新闻 + 决策日志 + 宏观 + 候选浮现 全部穿成一线
        </div>
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>{trackedSymbol} 研究分</span>
        <span className="htr-num" style={{ fontSize: 18, fontWeight: 700, color: "var(--htr-accent)" }}>{score}</span>
        <Sparkline data={[42, 45, 48, 52, 58, 62, 68, 72, 75, 78]} width={70} height={24}
                   stroke="var(--htr-accent)" fill="var(--htr-accent-soft)" />
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────

function Spine({ events, hero, livePrice }) {
  return (
    <div style={{ position: "relative", paddingLeft: 100 }}>
      {/* Main vertical line */}
      <div style={{
        position: "absolute", left: 80, top: 6, bottom: 6,
        width: 2, background: "var(--htr-line)",
      }} />
      {events.map((e, i) => <SpineRow key={i} ev={e} hero={hero} livePrice={livePrice} />)}
    </div>
  );
}

function SpineRow({ ev, hero, livePrice }) {
  const dotColor = ({
    news_high:   "var(--htr-bear)",
    news_med:    "var(--htr-warn)",
    news_low:    "var(--htr-ink-3)",
    candidate:   "var(--htr-accent)",
    decision:    "var(--htr-info)",
    macro:       "var(--htr-warn)",
    session:     "var(--htr-ink-2)",
    leader:      "var(--htr-bull)",
  })[ev.kind] || "var(--htr-ink-3)";

  // The "leader" event uses an expanded card with mini chart
  const expanded = ev.kind === "leader";

  return (
    <div style={{ position: "relative", paddingBottom: expanded ? 18 : 12 }}>
      {/* Time label */}
      <div style={{
        position: "absolute", left: -100, top: 2, width: 76,
        textAlign: "right",
      }}>
        <div className="htr-mono" style={{ fontSize: 11, fontWeight: 600, color: "var(--htr-ink-2)" }}>{ev.ts}</div>
        <div style={{ fontSize: 9, color: "var(--htr-ink-4)", letterSpacing: "0.08em", marginTop: 1 }}>
          {ev.kindLabel}
        </div>
      </div>

      {/* Dot */}
      <div style={{
        position: "absolute", left: -22, top: 6,
        width: 16, height: 16, borderRadius: "50%",
        background: dotColor, border: "3px solid var(--htr-surface)",
        boxShadow: `0 0 0 1px ${dotColor}`,
      }} />

      {/* Card */}
      {expanded ? (
        <LeaderEmergeCard ev={ev} hero={hero} livePrice={livePrice} />
      ) : (
        <SpineCard ev={ev} dotColor={dotColor} />
      )}
    </div>
  );
}

function SpineCard({ ev, dotColor }) {
  return (
    <div style={{
      padding: "8px 12px", background: "var(--htr-surface)",
      border: "1px solid var(--htr-line)", borderLeft: `3px solid ${dotColor}`,
      borderRadius: 4,
    }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 2 }}>
        {ev.src && <span className="htr-chip" style={{ fontSize: 9.5 }}>{ev.src}</span>}
        {ev.action && (
          <code style={{
            fontFamily: "var(--htr-font-mono)", fontSize: 10,
            background: "var(--htr-surface-3)", padding: "1px 6px", borderRadius: 2,
            color: "var(--htr-ink-2)",
          }}>{ev.action}</code>
        )}
        {ev.weight && <span style={{ fontSize: 9.5, color: "var(--htr-ink-3)" }}>weight {ev.weight}</span>}
        <span style={{ flex: 1 }} />
        {ev.symbols && ev.symbols.map((s) => (
          <span key={s} className="htr-chip" style={{ fontSize: 9.5 }}>{s}</span>
        ))}
      </div>
      <div style={{ fontSize: 12.5, color: "var(--htr-ink)", lineHeight: 1.45 }}>{ev.text}</div>
      {ev.note && (
        <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginTop: 4 }}>{ev.note}</div>
      )}
    </div>
  );
}

function LeaderEmergeCard({ ev, hero, livePrice }) {
  return (
    <div style={{
      padding: "12px 14px",
      background: "linear-gradient(180deg, var(--htr-accent-soft) 0%, var(--htr-surface) 100%)",
      border: "1px solid var(--htr-accent)", borderRadius: 6,
      display: "flex", flexDirection: "column", gap: 10,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{
          padding: "3px 9px", background: "var(--htr-accent)", color: "var(--htr-accent-ink)",
          fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", borderRadius: 3,
        }}>LEADER 浮现</span>
        <span style={{ fontSize: 12.5, color: "var(--htr-ink)" }}>
          {ev.text}
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr 1fr", gap: 14, alignItems: "center" }}>
        <div>
          <div className="htr-mono" style={{ fontSize: 18, fontWeight: 800 }}>{hero.symbol}</div>
          <div style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{hero.nameCn}</div>
        </div>
        <div>
          <div className="htr-num" style={{ fontSize: 17, fontWeight: 700 }}>
            <AnimatedPrice value={livePrice} decimals={0} />
          </div>
          <div className={"htr-num " + changeClass(hero.chg)} style={{ fontSize: 10.5 }}>
            {arrow(hero.chg)} {fmtPct(hero.chg)}
          </div>
        </div>
        <div>
          <div className="htr-num" style={{ fontSize: 17, fontWeight: 800, color: "var(--htr-accent)" }}>{hero.score}</div>
          <div style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>研究分 (+12)</div>
        </div>
        <Sparkline data={[42, 48, 52, 58, 62, 68, 72, 75, 78]} width={140} height={32}
                   stroke="var(--htr-accent)" fill="var(--htr-accent-soft)" />
      </div>
      <div style={{
        fontSize: 11.5, color: "var(--htr-ink-2)", lineHeight: 1.5,
        padding: "8px 10px", background: "var(--htr-surface)", borderRadius: 4,
        border: "1px solid var(--htr-line-soft)",
      }}>
        <strong style={{ color: "var(--htr-ink)" }}>判断: </strong>{hero.one_liner}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────

function V4RightRail({ candidate, livePrice }) {
  return (
    <div style={{ display: "grid", gridTemplateRows: "auto 1fr auto", gap: 10, minHeight: 0 }}>
      {/* Hero compact */}
      <div className="htr-card" style={{ padding: "12px 14px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <span className="htr-eyebrow" style={{ color: "var(--htr-accent)" }}>当前主标的</span>
          <span style={{ flex: 1 }} />
          <span className="htr-chip warn">{candidate.priority}</span>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <div className="htr-mono" style={{ fontSize: 20, fontWeight: 800 }}>{candidate.symbol}</div>
          <div style={{ fontSize: 11, color: "var(--htr-ink-3)" }}>{candidate.nameCn}</div>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 6 }}>
          <span className="htr-num" style={{ fontSize: 24, fontWeight: 700 }}>
            <AnimatedPrice value={livePrice} decimals={0} />
          </span>
          <span className={"htr-num " + changeClass(candidate.chg)} style={{ fontSize: 12, fontWeight: 700 }}>
            {arrow(candidate.chg)} {fmtPct(candidate.chg)}
          </span>
          <span style={{ flex: 1 }} />
          <div style={{ textAlign: "right" }}>
            <div className="htr-num" style={{ fontSize: 18, fontWeight: 800, color: "var(--htr-accent)" }}>{candidate.score}</div>
            <div style={{ fontSize: 9, color: "var(--htr-ink-3)" }}>研究分</div>
          </div>
        </div>
      </div>

      {/* Vertical ladder + actions */}
      <div className="htr-card" style={{ display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
        <CardHead title="七档价格阶梯" sub="Price Ladder" />
        <div style={{ flex: 1, overflow: "auto", padding: "10px 14px" }}>
          <VerticalLadder ladder={candidate.ladder} currentPrice={livePrice} height={340} />
        </div>
        <div style={{
          padding: "10px 14px", borderTop: "1px solid var(--htr-line)",
          background: "var(--htr-surface-2)",
        }}>
          <div className="htr-eyebrow" style={{ marginBottom: 6 }}>操作 · 研究模式</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            <ActButton tone="bull">分批介入</ActButton>
            <ActButton tone="info">设条件单</ActButton>
            <ActButton tone="warn">仅观察</ActButton>
            <ActButton tone="bear">放弃</ActButton>
          </div>
          <div style={{ fontSize: 10, color: "var(--htr-ink-3)", marginTop: 8, lineHeight: 1.4 }}>
            按钮仅写入<Term>决策日志</Term>, 不连券商. 自动执行需走完 <Term k="§10">§10 八阶门槛</Term>.
          </div>
        </div>
      </div>

      {/* P8-16 Cycle 1 — §10 gate strip moved to global nav chip. */}
    </div>
  );
}

function ActButton({ tone, children }) {
  const map = {
    bull: { bg: "var(--htr-bull-bg)", fg: "var(--htr-bull)" },
    bear: { bg: "var(--htr-bear-bg)", fg: "var(--htr-bear)" },
    info: { bg: "var(--htr-info-bg)", fg: "var(--htr-info)" },
    warn: { bg: "var(--htr-warn-bg)", fg: "var(--htr-warn)" },
  };
  const c = map[tone];
  return (
    <button style={{
      padding: "7px 10px", border: `1px solid ${c.fg}33`, background: c.bg, color: c.fg,
      borderRadius: 4, fontSize: 11.5, fontWeight: 700, cursor: "pointer",
      fontFamily: "inherit", letterSpacing: "0.02em",
    }}>{children}</button>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// Event stream builder

function buildEventStream(data) {
  const evs = [];

  // News
  data.newsTimeline.forEach((n) => {
    evs.push({
      ts: n.ts,
      tsKey: parseTs(n.ts),
      kind: n.weight === "high" ? "news_high" : n.weight === "medium" ? "news_med" : "news_low",
      kindLabel: "NEWS · " + n.weight.toUpperCase(),
      src: n.src,
      weight: n.weight,
      text: n.text,
      symbols: n.linkedSymbols,
    });
  });

  // Decision log
  data.decisionLog.forEach((d) => {
    if (d.action === "session_open") {
      evs.push({
        ts: d.ts.slice(0, 8),
        tsKey: parseTs(d.ts.slice(0, 8)),
        kind: "session",
        kindLabel: "SESSION",
        action: d.action,
        text: "JST 09:00 开盘",
      });
    } else if (d.action === "macro_change") {
      evs.push({
        ts: d.ts.slice(0, 8),
        tsKey: parseTs(d.ts.slice(0, 8)),
        kind: "macro",
        kindLabel: "MACRO",
        action: d.action,
        text: d.note,
        symbols: [d.symbol],
      });
    } else if (d.action === "scan_completed") {
      evs.push({
        ts: d.ts.slice(0, 8),
        tsKey: parseTs(d.ts.slice(0, 8)),
        kind: "decision",
        kindLabel: "DECISION LOG",
        action: d.action,
        text: d.note,
      });
    }
  });

  // Synthetic "leader emerged" pinned event
  evs.push({
    ts: "10:30 JST",
    tsKey: parseTs("10:30 JST"),
    kind: "leader",
    kindLabel: "LEADER 浮现",
    text: "8035.T 研究分从 66 → 78, 跨过 75 阈值, 进入今日机会中心首位",
  });

  // Sort desc
  evs.sort((a, b) => b.tsKey - a.tsKey);
  return evs;
}

function parseTs(ts) {
  // crude — extract hh:mm
  const m = ts.match(/(\d{1,2}):(\d{2})/);
  if (!m) return 0;
  return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
}

window.V4WorkflowSpine = V4WorkflowSpine;
