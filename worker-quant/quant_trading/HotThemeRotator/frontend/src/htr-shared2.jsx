// htr-shared2.jsx — extra shared components used by V1 / V2 / V4.
// Refined versions of MarketTempCell (row/tile), VerticalLadder, ScoreBar.

function ScoreBar({ value, max = 100 }) {
  return (
    <div style={{ width: 50, height: 5, background: "var(--htr-line-soft)", borderRadius: 3, overflow: "hidden" }}>
      <div style={{ width: `${(value / max) * 100}%`, height: "100%", background: HTR.heatColor(value) }} />
    </div>
  );
}

const TEMP_GRAD = "linear-gradient(90deg,var(--htr-info) 0%,var(--htr-warn) 55%,var(--htr-heat-hot) 100%)";
function MarketTempCell({ m, variant = "row" }) {
  const hasSpark = Array.isArray(m.spark) && m.spark.length >= 2;
  const tempPct = Math.max(3, Math.min(100, Number(m.temp) || 0));
  if (variant === "tile") {
    return (
      <div style={{ border: "1px solid var(--htr-line)", borderRadius: 10, padding: "12px 14px", background: "var(--htr-surface)", display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600 }}>{m.label}</div>
            <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginTop: 1 }}>{m.sub}</div>
          </div>
          <span className="htr-chip" style={{ fontSize: 9.5 }}>{m.state}</span>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 8 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <span className="htr-num" style={{ fontSize: 16, fontWeight: 600 }}>{HTR.fmtPrice(m.price, m.id === "USDJPY" ? 2 : 0)}</span>
            <span className={"htr-num " + HTR.changeClass(m.chg)} style={{ fontSize: 12 }}>{HTR.arrow(m.chg)} {HTR.fmtPct(m.chg)}</span>
          </div>
          <span className="htr-num" style={{ fontSize: 15, fontWeight: 700, color: HTR.heatColor(m.temp) }}>{m.temp}°</span>
        </div>
        {/* temperature gauge — real 0-100 reading, fills the tile */}
        <div style={{ height: 6, borderRadius: 3, background: "var(--htr-surface-3)", overflow: "hidden", position: "relative" }}>
          <div style={{ position: "absolute", inset: 0, background: TEMP_GRAD, opacity: 0.22 }} />
          <div style={{ width: tempPct + "%", height: "100%", background: TEMP_GRAD, borderRadius: 3 }} />
        </div>
        {hasSpark
          ? <Sparkline data={m.spark} width={undefined} height={24} />
          : <div style={{ fontSize: 9.5, color: "var(--htr-ink-4)", fontFamily: "var(--htr-font-mono)" }}>无盘中走势数据</div>}
      </div>
    );
  }
  // row variant — flexible so it never overflows a narrow rail (V1 286 / V4 256)
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 12px", borderBottom: "1px solid var(--htr-line-soft)" }}>
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ fontSize: 12.5, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.label}</div>
        <div style={{ fontSize: 9.5, color: "var(--htr-ink-4)", letterSpacing: "0.04em" }}>{m.state}</div>
      </div>
      {/* sparkline when we have intraday data, else a compact temp gauge (never an empty gap) */}
      <div style={{ flexShrink: 0, width: 48 }}>
        {hasSpark
          ? <Sparkline data={m.spark} width={48} height={20} />
          : <div style={{ height: 5, borderRadius: 3, background: "var(--htr-surface-3)", overflow: "hidden" }}>
              <div style={{ width: tempPct + "%", height: "100%", background: TEMP_GRAD, borderRadius: 3 }} /></div>}
      </div>
      <div className="htr-num" style={{ textAlign: "right", flexShrink: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 600 }}>{HTR.fmtPrice(m.price, m.id === "USDJPY" ? 2 : 0)}</div>
        <div style={{ fontSize: 10 }} className={HTR.changeClass(m.chg)}>{HTR.fmtPct(m.chg, 2)}</div>
      </div>
      <div style={{ flexShrink: 0, width: 26, height: 26, borderRadius: 5, background: HTR.heatBg(m.temp), display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "var(--htr-font-mono)", fontSize: 11.5, fontWeight: 700, color: HTR.heatColor(m.temp) }}>{m.temp}</div>
    </div>
  );
}

function VerticalLadder({ ladder, currentPrice, height = 360 }) {
  if (!ladder || !ladder.length) return null;
  const allP = [...ladder.map((r) => r.price), currentPrice];
  const min = Math.min(...allP), max = Math.max(...allP);
  const pad = (max - min) * 0.10, yMin = min - pad, yMax = max + pad;
  const W = 244, axisX = 64, labelX = 78;
  const yOf = (p) => 14 + ((yMax - p) / (yMax - yMin)) * (height - 28);
  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} preserveAspectRatio="xMidYMid meet" style={{ display: "block" }}>
      <line x1={axisX} y1={8} x2={axisX} y2={height - 8} stroke="var(--htr-line)" strokeWidth="1" />
      {(() => { const y = yOf(currentPrice); return (
        <g>
          <line x1={axisX - 6} y1={y} x2={W - 4} y2={y} stroke="var(--htr-accent)" strokeWidth="1.5" />
          <rect x={4} y={y - 9} width={56} height={18} fill="var(--htr-accent)" rx="3" />
          <text x={32} y={y + 4} fontSize="10.5" fontWeight="700" textAnchor="middle" fill="var(--htr-accent-ink)" style={{ fontFamily: "var(--htr-font-mono)" }}>现价</text>
          <text x={W - 6} y={y - 5} fontSize="10.5" fontWeight="700" textAnchor="end" fill="var(--htr-accent)" style={{ fontFamily: "var(--htr-font-mono)" }}>{HTR.fmtPrice(currentPrice, 0)}</text>
        </g>
      ); })()}
      {antiCollideLabels(ladder, yOf, { gap: 27, top: 18, bottom: height - 10 }).map((row, i) => {
        const c = ladderColor(row.kind);
        return (
          <g key={i}>
            <circle cx={axisX} cy={row.priceY} r="4.5" fill={c} stroke="var(--htr-surface)" strokeWidth="1.8" />
            <path d={`M${axisX + 5},${row.priceY} L${labelX - 4},${row.labelY}`} stroke={c} strokeWidth="0.7" fill="none" opacity="0.7" />
            <text x={labelX} y={row.labelY - 3} fontSize="10.5" fontWeight="600" fill={c}>{row.label}</text>
            <text x={labelX} y={row.labelY + 10} fontSize="11" fontWeight="700" fill="var(--htr-ink)" style={{ fontFamily: "var(--htr-font-mono)" }}>
              {HTR.fmtPrice(row.price, 0)}<tspan dx="6" fill="var(--htr-ink-3)" fontWeight="500">{HTR.fmtPct(row.pct, 1)}</tspan>
            </text>
            <text x={axisX - 9} y={row.priceY + 4} fontSize="9.5" fill="var(--htr-ink-4)" textAnchor="end">{String(i + 1).padStart(2, "0")}</text>
          </g>
        );
      })}
    </svg>
  );
}

Object.assign(window, { ScoreBar, MarketTempCell, VerticalLadder });
