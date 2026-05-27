// shared.jsx — design tokens + reusable components for the HotThemeRotator redesign.
// All four variations consume these. Tokens are CSS variables on :root so the
// Tweaks panel can re-skin everything live (light/dark, primary color, density,
// font pairing).

const { useState, useEffect, useRef, useMemo, Fragment } = React;

// ─────────────────────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────────────────────

const TOKEN_STYLE = `
:root {
  /* Surfaces — light, warm research paper */
  --htr-bg:       #F6F4ED;
  --htr-surface:  #FFFFFF;
  --htr-surface-2:#FBFAF5;
  --htr-surface-3:#EFEDE5;
  --htr-line:     #E3DFD3;
  --htr-line-2:   #D5D0BF;
  --htr-line-soft:#EDEAE0;

  /* Text */
  --htr-ink:      #1A1817;
  --htr-ink-2:    #4B4843;
  --htr-ink-3:    #7A766C;
  --htr-ink-4:    #ACA89C;

  /* Semantic */
  --htr-bull:     #1F7A4D;
  --htr-bull-bg:  #E6F1E9;
  --htr-bear:     #B23A3A;
  --htr-bear-bg:  #F6E5E2;
  --htr-warn:     #B5772E;
  --htr-warn-bg:  #FAEEDB;
  --htr-info:     #355D8C;
  --htr-info-bg:  #E5ECF4;

  /* Accent (driven by tweak) */
  --htr-accent:        #1F3A5F;
  --htr-accent-soft:   #E4E9F0;
  --htr-accent-ink:    #FFFFFF;

  /* Typography */
  --htr-font-sans: "IBM Plex Sans", "Noto Sans JP", system-ui, sans-serif;
  --htr-font-serif:"IBM Plex Serif", "Noto Serif JP", Georgia, serif;
  --htr-font-mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --htr-font-body: var(--htr-font-sans);
  --htr-font-num:  var(--htr-font-mono);

  /* Density */
  --htr-row-h:     30px;
  --htr-pad-x:     14px;
  --htr-pad-y:     10px;
  --htr-gap:       14px;
  --htr-radius:    6px;
  --htr-radius-lg: 10px;

  /* Text sizes */
  --htr-fs-xs:  10.5px;
  --htr-fs-sm:  11.5px;
  --htr-fs-md:  13px;
  --htr-fs-lg:  15px;
  --htr-fs-xl:  19px;
  --htr-fs-2xl: 26px;
  --htr-fs-3xl: 38px;
  --htr-fs-num: 14px;
}

:root[data-htr-dark="true"] {
  --htr-bg:       #14130F;
  --htr-surface:  #1B1A16;
  --htr-surface-2:#1F1E1A;
  --htr-surface-3:#262520;
  --htr-line:     #2E2C26;
  --htr-line-2:   #3A382F;
  --htr-line-soft:#23221D;
  --htr-ink:      #ECE9DF;
  --htr-ink-2:    #BFBBAE;
  --htr-ink-3:    #8A867A;
  --htr-ink-4:    #5A5750;
  --htr-bull:     #5BB57E;
  --htr-bull-bg:  #1B2C22;
  --htr-bear:     #DC6F6F;
  --htr-bear-bg:  #2D1E1E;
  --htr-warn:     #D9A45E;
  --htr-warn-bg:  #2A2218;
  --htr-info:     #7AA8D9;
  --htr-info-bg:  #1B232D;
  --htr-accent-soft: #1F2A38;
  --htr-accent-ink:  #ECE9DF;
}

:root[data-htr-density="compact"] {
  --htr-row-h: 24px; --htr-pad-x: 10px; --htr-pad-y: 6px; --htr-gap: 9px;
  --htr-fs-sm: 11px; --htr-fs-md: 12px; --htr-fs-lg: 13.5px; --htr-fs-num: 13px;
}
:root[data-htr-density="comfy"] {
  --htr-row-h: 36px; --htr-pad-x: 18px; --htr-pad-y: 14px; --htr-gap: 18px;
  --htr-fs-sm: 12px; --htr-fs-md: 14px; --htr-fs-lg: 16.5px; --htr-fs-num: 15px;
}

/* Base */
.htr { font-family: var(--htr-font-body); color: var(--htr-ink);
  font-feature-settings: "tnum", "ss01"; -webkit-font-smoothing: antialiased; }
.htr, .htr * { box-sizing: border-box; }
.htr-num { font-family: var(--htr-font-num); font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em; }
.htr-mono { font-family: var(--htr-font-mono); }
.htr-serif{ font-family: var(--htr-font-serif); }

/* Colors helpers */
.htr-up   { color: var(--htr-bull); }
.htr-down { color: var(--htr-bear); }
.htr-flat { color: var(--htr-ink-3); }

/* Headers & labels */
.htr-eyebrow { font-size: var(--htr-fs-xs); letter-spacing: 0.14em; color: var(--htr-ink-3);
  text-transform: uppercase; font-weight: 600; }
.htr-divider { border: 0; border-top: 1px solid var(--htr-line); margin: 0; }

/* Card */
.htr-card { background: var(--htr-surface); border: 1px solid var(--htr-line);
  border-radius: var(--htr-radius); }
.htr-card-flush { background: var(--htr-surface); border-top: 1px solid var(--htr-line);
  border-bottom: 1px solid var(--htr-line); }

/* Buttons / chip */
.htr-chip { display: inline-flex; align-items: center; gap: 5px; padding: 2px 7px;
  font-size: var(--htr-fs-xs); border-radius: 3px; border: 1px solid var(--htr-line-2);
  background: var(--htr-surface-2); color: var(--htr-ink-2); letter-spacing: 0.04em; }
.htr-chip.bull { color: var(--htr-bull); border-color: color-mix(in oklab, var(--htr-bull) 30%, var(--htr-line-2)); background: var(--htr-bull-bg); }
.htr-chip.bear { color: var(--htr-bear); border-color: color-mix(in oklab, var(--htr-bear) 30%, var(--htr-line-2)); background: var(--htr-bear-bg); }
.htr-chip.warn { color: var(--htr-warn); border-color: color-mix(in oklab, var(--htr-warn) 30%, var(--htr-line-2)); background: var(--htr-warn-bg); }
.htr-chip.info { color: var(--htr-info); border-color: color-mix(in oklab, var(--htr-info) 30%, var(--htr-line-2)); background: var(--htr-info-bg); }
.htr-chip.accent { color: var(--htr-accent); border-color: color-mix(in oklab, var(--htr-accent) 30%, var(--htr-line-2)); background: var(--htr-accent-soft); }

/* Pulse highlight */
@keyframes htr-pulse-up   { 0% { background: rgba(31,122,77,0.18); } 100% { background: transparent; } }
@keyframes htr-pulse-down { 0% { background: rgba(178,58,58,0.18); } 100% { background: transparent; } }
.htr-pulse-up   { animation: htr-pulse-up 1.6s ease-out; }
.htr-pulse-down { animation: htr-pulse-down 1.6s ease-out; }

/* Scrollbars in artboards */
.htr ::-webkit-scrollbar { width: 8px; height: 8px; }
.htr ::-webkit-scrollbar-thumb { background: var(--htr-line-2); border-radius: 4px; }
.htr ::-webkit-scrollbar-track { background: transparent; }
`;

// Inject once
if (!document.getElementById('htr-tokens')) {
  const s = document.createElement('style');
  s.id = 'htr-tokens';
  s.textContent = TOKEN_STYLE;
  document.head.appendChild(s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function fmtPrice(p, decimals = 2) {
  if (p == null || isNaN(p)) return "—";
  return p.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}
function fmtPct(p, decimals = 2, withSign = true) {
  if (p == null || isNaN(p)) return "—";
  const sign = withSign && p > 0 ? "+" : "";
  return `${sign}${p.toFixed(decimals)}%`;
}
function changeClass(n) {
  if (n == null || isNaN(n)) return "htr-flat";
  if (n > 0) return "htr-up";
  if (n < 0) return "htr-down";
  return "htr-flat";
}
function arrow(n) {
  if (n > 0) return "▲";
  if (n < 0) return "▼";
  return "·";
}

window.HTR_HELPERS = { fmtPrice, fmtPct, changeClass, arrow };

// ─────────────────────────────────────────────────────────────────────────────
// AnimatedPrice — pulses background on change
// ─────────────────────────────────────────────────────────────────────────────

function AnimatedPrice({ value, decimals = 2, style }) {
  const [v, setV] = useState(value);
  const [dir, setDir] = useState(0);
  useEffect(() => {
    if (value === v) return;
    setDir(value > v ? 1 : -1);
    setV(value);
    const t = setTimeout(() => setDir(0), 1600);
    return () => clearTimeout(t);
  }, [value]);
  const cls = dir > 0 ? "htr-pulse-up" : dir < 0 ? "htr-pulse-down" : "";
  return (
    <span className={"htr-num " + cls} key={dir + "-" + v}
          style={{ display: "inline-block", padding: "0 4px", borderRadius: 3, ...style }}>
      {fmtPrice(v, decimals)}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparkline
// ─────────────────────────────────────────────────────────────────────────────

function Sparkline({ data, width = 80, height = 22, stroke, fill, area = true }) {
  if (!data || data.length === 0) return null;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const dx = data.length > 1 ? width / (data.length - 1) : 0;
  const points = data.map((v, i) => [i * dx, height - ((v - min) / range) * height]);
  const path = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const areaPath = path + ` L${width},${height} L0,${height} Z`;
  const last = data[data.length - 1], first = data[0];
  const up = last >= first;
  const s = stroke || (up ? "var(--htr-bull)" : "var(--htr-bear)");
  const f = fill   || (up ? "rgba(31,122,77,0.10)" : "rgba(178,58,58,0.10)");
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {area && <path d={areaPath} fill={f} />}
      <path d={path} fill="none" stroke={s} strokeWidth="1.4" />
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TempGauge — semi-circular temperature 0..100
// ─────────────────────────────────────────────────────────────────────────────

function TempGauge({ value, size = 64, label }) {
  const r = size / 2 - 6;
  const cx = size / 2, cy = size / 2;
  const start = Math.PI, end = 2 * Math.PI;
  const t = Math.max(0, Math.min(100, value)) / 100;
  const angle = start + (end - start) * t;
  const x = cx + r * Math.cos(angle), y = cy + r * Math.sin(angle);
  const large = t > 0.5 ? 1 : 0;
  const arc = `M ${cx - r},${cy} A ${r} ${r} 0 ${large} 1 ${x.toFixed(2)} ${y.toFixed(2)}`;
  const arcBg = `M ${cx - r},${cy} A ${r} ${r} 0 1 1 ${cx + r},${cy}`;
  const heatColor = value >= 70 ? "var(--htr-bear)" : value >= 50 ? "var(--htr-warn)" : value >= 30 ? "var(--htr-info)" : "var(--htr-ink-3)";
  return (
    <svg width={size} height={size / 2 + 6} viewBox={`0 0 ${size} ${size / 2 + 6}`}>
      <path d={arcBg} fill="none" stroke="var(--htr-line)" strokeWidth="4" strokeLinecap="round" />
      <path d={arc}   fill="none" stroke={heatColor}      strokeWidth="4" strokeLinecap="round" />
      <text x={cx} y={cy - 2} textAnchor="middle" fontSize="13" fontWeight="600"
            fill="var(--htr-ink)" style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>
        {Math.round(value)}
      </text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MarketTempCell — one entry in the multi-market temperature panel
// ─────────────────────────────────────────────────────────────────────────────

function MarketTempCell({ m, variant = "row" }) {
  if (variant === "tile") {
    return (
      <div style={{
        border: "1px solid var(--htr-line)", borderRadius: "var(--htr-radius)",
        padding: "12px 14px", background: "var(--htr-surface)",
        display: "flex", flexDirection: "column", gap: 8,
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: "var(--htr-fs-md)", fontWeight: 600, color: "var(--htr-ink)" }}>{m.label}</div>
            <div style={{ fontSize: "var(--htr-fs-xs)", color: "var(--htr-ink-3)", marginTop: 1 }}>{m.sub}</div>
          </div>
          <span className="htr-chip" style={{ fontSize: 9.5 }}>{m.state}</span>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span className="htr-num" style={{ fontSize: "var(--htr-fs-lg)", fontWeight: 600 }}>{fmtPrice(m.price)}</span>
          <span className={"htr-num " + changeClass(m.chg)} style={{ fontSize: "var(--htr-fs-sm)" }}>
            {arrow(m.chg)} {fmtPct(m.chg)}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Sparkline data={m.spark} width={90} height={26} />
          <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ flex: 1, height: 4, background: "var(--htr-line-soft)", borderRadius: 2, overflow: "hidden" }}>
              <div style={{
                width: `${m.temp}%`, height: "100%",
                background: m.temp >= 70 ? "var(--htr-bear)" : m.temp >= 50 ? "var(--htr-warn)" : m.temp >= 30 ? "var(--htr-info)" : "var(--htr-ink-3)",
              }} />
            </div>
            <span className="htr-num" style={{ fontSize: "var(--htr-fs-xs)", color: "var(--htr-ink-3)", width: 28, textAlign: "right" }}>
              {m.temp}°
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Row variant (left rail in V1)
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "60px 1fr 60px 38px",
      alignItems: "center", gap: 8, padding: "8px 12px",
      borderBottom: "1px solid var(--htr-line-soft)",
    }}>
      <div>
        <div style={{ fontSize: "var(--htr-fs-sm)", fontWeight: 600 }}>{m.label}</div>
        <div style={{ fontSize: 9.5, color: "var(--htr-ink-4)", letterSpacing: "0.04em" }}>{m.state}</div>
      </div>
      <Sparkline data={m.spark} width={70} height={20} />
      <div className="htr-num" style={{ textAlign: "right" }}>
        <div style={{ fontSize: "var(--htr-fs-sm)", fontWeight: 600 }}>{fmtPrice(m.price, m.id === "USDJPY" ? 2 : 0)}</div>
        <div style={{ fontSize: 10 }} className={changeClass(m.chg)}>{fmtPct(m.chg, 2)}</div>
      </div>
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <div style={{
          width: 26, height: 26, borderRadius: 4,
          background: m.temp >= 70 ? "var(--htr-bear-bg)" : m.temp >= 50 ? "var(--htr-warn-bg)" : m.temp >= 30 ? "var(--htr-info-bg)" : "var(--htr-surface-3)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontFamily: "var(--htr-font-mono)", fontSize: 11, fontWeight: 600,
          color: m.temp >= 70 ? "var(--htr-bear)" : m.temp >= 50 ? "var(--htr-warn)" : m.temp >= 30 ? "var(--htr-info)" : "var(--htr-ink-3)",
        }}>{m.temp}</div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CalibrationBadge — top-of-page banner / inline
// ─────────────────────────────────────────────────────────────────────────────

function CalibrationBadge({ inline = false }) {
  const c = window.HTR_DATA.meta.calibration;
  return (
    <div className="htr-chip warn" style={{
      padding: inline ? "2px 7px" : "3px 9px",
      fontSize: inline ? "var(--htr-fs-xs)" : "var(--htr-fs-xs)",
      letterSpacing: "0.02em",
    }}>
      <span style={{ width: 6, height: 6, background: "var(--htr-warn)", borderRadius: "50%", display: "inline-block" }} />
      <span style={{ fontWeight: 600 }}>未校准 · 研究分</span>
      {!inline && <span style={{ color: "var(--htr-ink-3)", marginLeft: 3 }}>
        n={c.sample}
      </span>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// KLineChart — SVG candles with optional right-anchored price ladder
// ─────────────────────────────────────────────────────────────────────────────

function KLineChart({
  data, width = 520, height = 280, ladder = null,
  padding = { top: 14, right: 110, bottom: 22, left: 8 },
  withVolume = true, withMA = true, with52wLines = true,
}) {
  // Crosshair hover state — broker-app-style: vertical line snaps to bar,
  // horizontal line follows cursor Y, fixed OHLCV box in top-right inside
  // chart area. Pure read-only UI (Rule 11 — no writeback, no decision_log,
  // no calibration mutation). The realized historical change % in the tooltip
  // is structural data display, not LLM narrative, so Rule 8.3.1 regex does
  // NOT apply (that rule guards probability/win-rate language in narrative
  // output only).
  const [hover, setHover] = React.useState(null); // {idx, yPx} | null
  const svgRef = React.useRef(null);

  if (!data || !data.length) return null;
  const innerW = width - padding.left - padding.right;
  const innerH_total = height - padding.top - padding.bottom;
  // P8-16 C2: bottom 25% reserved for volume sub-chart when enabled.
  const volH = withVolume ? Math.round(innerH_total * 0.25) : 0;
  const priceGap = withVolume ? 4 : 0;  // small gap between price and volume
  const innerH = innerH_total - volH - priceGap;

  // Y range for price area (top portion) — include ladder prices if provided
  const candleVals = data.flatMap((c) => [c.high, c.low]);
  const ladderVals = ladder ? ladder.map((r) => r.price) : [];
  const allVals = [...candleVals, ...ladderVals];
  const min = Math.min(...allVals), max = Math.max(...allVals);
  const pad = (max - min) * 0.06;
  const yMin = min - pad, yMax = max + pad;
  const yScale = (v) => padding.top + innerH - ((v - yMin) / (yMax - yMin)) * innerH;

  const candleW = Math.max(2, innerW / data.length * 0.7);
  const slot = innerW / data.length;

  // Crosshair handlers — compute bar index by inverse-mapping mouseX through
  // the same `slot` geometry used to lay out candles. yScale^-1 gives the
  // price the cursor is currently pointing at (for the horizontal line label).
  const onChartMouseMove = (e) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    // Scale coords from CSS pixels to SVG user units when sizing differs.
    const mouseX = (e.clientX - rect.left) * (width / rect.width);
    const mouseY = (e.clientY - rect.top) * (height / rect.height);
    if (mouseX < padding.left || mouseX > padding.left + innerW
        || mouseY < padding.top || mouseY > padding.top + innerH_total) {
      setHover(null);
      return;
    }
    let idx = Math.floor((mouseX - padding.left) / slot);
    idx = Math.max(0, Math.min(data.length - 1, idx));
    setHover({ idx, yPx: mouseY });
  };
  const onChartMouseLeave = () => setHover(null);
  const yInv = (yPx) => yMax - ((yPx - padding.top) / innerH) * (yMax - yMin);

  // Y-axis gridlines (5) in price area
  const gridSteps = 5;
  const grid = [];
  for (let i = 0; i <= gridSteps; i++) {
    const v = yMin + (yMax - yMin) * (i / gridSteps);
    grid.push({ v, y: yScale(v) });
  }

  // P8-16 C2 — moving averages computed inline from `data` (close series).
  const rollingMean = (closes, window) => {
    const out = new Array(closes.length).fill(null);
    if (closes.length < window) return out;
    let sum = 0;
    for (let i = 0; i < closes.length; i++) {
      sum += closes[i];
      if (i >= window) sum -= closes[i - window];
      if (i >= window - 1) out[i] = sum / window;
    }
    return out;
  };
  const closes = data.map((c) => c.close);
  const ma20 = withMA && closes.length >= 20 ? rollingMean(closes, 20) : null;
  const ma60 = withMA && closes.length >= 60 ? rollingMean(closes, 60) : null;
  const polyline = (series) => series
    .map((v, i) => v == null ? null : `${padding.left + slot * i + slot / 2},${yScale(v)}`)
    .filter(Boolean)
    .join(" ");

  // P8-16 C2 — 52w (252-session) high / low reference lines.
  // Uses whatever look-back is in `data`; label notes session count if < 252.
  const lookback = data.slice(-Math.min(252, data.length));
  const w52High = with52wLines ? Math.max(...lookback.map((c) => c.high)) : null;
  const w52Low  = with52wLines ? Math.min(...lookback.map((c) => c.low))  : null;
  const w52Label = data.length >= 252 ? "52w" : `${data.length}D`;

  // P8-16 C2 — Volume sub-chart geometry.
  const vols = data.map((c) => Number(c.volume ?? c.vol ?? 0));
  const vMax = Math.max(1, ...vols);
  const volTopY = padding.top + innerH + priceGap;
  const volScale = (v) => volTopY + volH - (v / vMax) * volH;

  return (
    <svg
      ref={svgRef}
      width={width} height={height}
      style={{ display: "block", overflow: "visible", cursor: "crosshair" }}
      onMouseMove={onChartMouseMove}
      onMouseLeave={onChartMouseLeave}
    >
      {/* Grid */}
      {grid.map((g, i) => (
        <line key={i} x1={padding.left} y1={g.y} x2={padding.left + innerW} y2={g.y}
              stroke="var(--htr-line-soft)" strokeWidth="0.6" strokeDasharray="2 3" />
      ))}

      {/* Candles */}
      {data.map((c, i) => {
        const x = padding.left + slot * i + slot / 2;
        const up = c.close >= c.open;
        const top = yScale(Math.max(c.open, c.close));
        const bot = yScale(Math.min(c.open, c.close));
        const high = yScale(c.high), low = yScale(c.low);
        const color = up ? "var(--htr-bull)" : "var(--htr-bear)";
        return (
          <g key={i}>
            <line x1={x} y1={high} x2={x} y2={low} stroke={color} strokeWidth="1" />
            <rect x={x - candleW / 2} y={top} width={candleW} height={Math.max(1, bot - top)}
                  fill={up ? color : color} opacity={up ? 0.9 : 1}
                  stroke={color} strokeWidth="0.6" />
          </g>
        );
      })}

      {/* P8-16 C2 — MA20 / MA60 polylines over candles */}
      {ma20 && (
        <polyline points={polyline(ma20)} fill="none"
                  stroke="var(--htr-info)" strokeWidth="1.2" opacity="0.85" />
      )}
      {ma60 && (
        <polyline points={polyline(ma60)} fill="none"
                  stroke="var(--htr-warn)" strokeWidth="1.2" opacity="0.85"
                  strokeDasharray="4 3" />
      )}
      {/* MA legend top-left */}
      {(ma20 || ma60) && (
        <g transform={`translate(${padding.left + 6}, ${padding.top + 8})`}>
          {ma20 && (
            <g>
              <line x1={0} y1={4} x2={14} y2={4} stroke="var(--htr-info)" strokeWidth="1.4" />
              <text x={18} y={7} fontSize="9" fill="var(--htr-ink-2)" fontWeight="600">MA20</text>
            </g>
          )}
          {ma60 && (
            <g transform="translate(54, 0)">
              <line x1={0} y1={4} x2={14} y2={4} stroke="var(--htr-warn)" strokeWidth="1.4" strokeDasharray="3 2" />
              <text x={18} y={7} fontSize="9" fill="var(--htr-ink-2)" fontWeight="600">MA60</text>
            </g>
          )}
        </g>
      )}

      {/* P8-16 C2 — 52w (or NwD) high/low reference horizontals + labels */}
      {with52wLines && w52High != null && w52Low != null && (
        <g>
          <line x1={padding.left} y1={yScale(w52High)} x2={padding.left + innerW}
                y2={yScale(w52High)} stroke="var(--htr-ink-3)" strokeWidth="0.7"
                strokeDasharray="6 4" opacity="0.55" />
          <text x={padding.left + innerW - 4} y={yScale(w52High) - 3} fontSize="9"
                textAnchor="end" fill="var(--htr-ink-3)" fontWeight="600">
            {w52Label} 高 {fmtPrice(w52High, 0)}
          </text>
          <line x1={padding.left} y1={yScale(w52Low)} x2={padding.left + innerW}
                y2={yScale(w52Low)} stroke="var(--htr-ink-3)" strokeWidth="0.7"
                strokeDasharray="6 4" opacity="0.55" />
          <text x={padding.left + innerW - 4} y={yScale(w52Low) + 10} fontSize="9"
                textAnchor="end" fill="var(--htr-ink-3)" fontWeight="600">
            {w52Label} 低 {fmtPrice(w52Low, 0)}
          </text>
        </g>
      )}

      {/* P8-16 C2 — Volume sub-chart at bottom 25% */}
      {withVolume && (
        <g>
          <line x1={padding.left} y1={volTopY} x2={padding.left + innerW} y2={volTopY}
                stroke="var(--htr-line)" strokeWidth="0.6" />
          <text x={padding.left + 4} y={volTopY + 9} fontSize="8.5"
                fill="var(--htr-ink-3)" fontWeight="600">VOL</text>
          {data.map((c, i) => {
            const v = Number(c.volume ?? c.vol ?? 0);
            if (!v) return null;
            const x = padding.left + slot * i + slot / 2;
            const y = volScale(v);
            const up = c.close >= c.open;
            return (
              <rect key={`v${i}`} x={x - candleW / 2} y={y} width={candleW}
                    height={Math.max(0.5, (volTopY + volH) - y)}
                    fill={up ? "var(--htr-bull)" : "var(--htr-bear)"} opacity="0.55" />
            );
          })}
        </g>
      )}

      {/* Price ladder labels on the right — labels anti-collide while price
          line stays at true y. Connector path bridges any vertical offset. */}
      {ladder && antiCollideLabels(ladder, yScale, {
        gap: 22, top: padding.top + 12, bottom: padding.top + innerH - 6,
      }).map((row, i) => {
        const isStop  = row.kind === "stop";
        const isExit  = row.kind.startsWith("exit");
        const isEntry = row.kind.startsWith("entry");
        const color = isStop ? "var(--htr-bear)" : isExit ? "var(--htr-info)" : isEntry ? "var(--htr-bull)" : "var(--htr-ink-2)";
        const bg    = isStop ? "var(--htr-bear-bg)" : isExit ? "var(--htr-info-bg)" : isEntry ? "var(--htr-bull-bg)" : "var(--htr-surface-2)";
        const labelBoxX = padding.left + innerW + 6;
        return (
          <g key={i}>
            {/* True price line across the candle area */}
            <line x1={padding.left} y1={row.priceY} x2={padding.left + innerW + 4} y2={row.priceY}
                  stroke={color} strokeWidth="0.7" strokeDasharray={isStop ? "3 3" : "0"} opacity="0.55" />
            {/* Connector from price line to label box when they diverge */}
            <path d={`M${padding.left + innerW + 4},${row.priceY} L${labelBoxX},${row.labelY}`}
                  stroke={color} strokeWidth="0.6" fill="none" opacity="0.7" />
            <rect x={labelBoxX} y={row.labelY - 9} width={padding.right - 12} height={18}
                  fill={bg} stroke={color} strokeWidth="0.7" rx="2" />
            <text x={labelBoxX + 5} y={row.labelY + 4} fontSize="9.5" fontWeight="600" fill={color}
                  style={{ letterSpacing: "0.03em" }}>
              {row.label}
            </text>
            <text x={padding.left + innerW + padding.right - 8} y={row.labelY + 4} fontSize="10" fontWeight="700"
                  textAnchor="end" fill={color}
                  style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>
              {fmtPrice(row.price, 0)}
            </text>
          </g>
        );
      })}

      {/* Current price line + label (uses last close) */}
      {(() => {
        const last = data[data.length - 1].close;
        const y = yScale(last);
        return (
          <g>
            <line x1={padding.left} y1={y} x2={padding.left + innerW} y2={y}
                  stroke="var(--htr-accent)" strokeWidth="1.2" strokeDasharray="0" />
            <rect x={padding.left + innerW - 56} y={y - 9} width={54} height={18}
                  fill="var(--htr-accent)" rx="2" />
            <text x={padding.left + innerW - 6} y={y + 4} fontSize="10.5" fontWeight="700"
                  textAnchor="end" fill="#fff"
                  style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>
              {fmtPrice(last, 0)}
            </text>
          </g>
        );
      })()}

      {/* X-axis tick labels (just session indices) */}
      <text x={padding.left} y={height - 6} fontSize="9" fill="var(--htr-ink-4)">D-40</text>
      <text x={padding.left + innerW / 2} y={height - 6} fontSize="9" fill="var(--htr-ink-4)" textAnchor="middle">D-20</text>
      <text x={padding.left + innerW} y={height - 6} fontSize="9" fill="var(--htr-ink-4)" textAnchor="end">今日</text>

      {/* Crosshair overlay (rendered last so it sits on top) */}
      {hover && (() => {
        const bar = data[hover.idx];
        const prev = hover.idx > 0 ? data[hover.idx - 1] : null;
        const barX = padding.left + slot * hover.idx + slot / 2;
        const cursorY = Math.max(padding.top, Math.min(padding.top + innerH, hover.yPx));
        const cursorPrice = yInv(cursorY);
        const chgPct = prev ? (bar.close - prev.close) / prev.close * 100 : null;
        const volStr = bar.volume != null
          ? bar.volume >= 1e8 ? (bar.volume / 1e8).toFixed(2) + '亿'
            : bar.volume >= 1e4 ? (bar.volume / 1e4).toFixed(1) + '万'
            : String(Math.round(bar.volume))
          : '—';
        const boxW = 102, boxH = 110;
        const boxX = padding.left + 6;
        const boxY = padding.top + 4;
        const upColor = "var(--htr-bull)";
        const dnColor = "var(--htr-bear)";
        const chgColor = chgPct == null ? "var(--htr-ink-2)" : chgPct >= 0 ? upColor : dnColor;
        return (
          <g pointerEvents="none">
            {/* Vertical crosshair line — spans price + volume areas */}
            <line x1={barX} y1={padding.top} x2={barX} y2={height - padding.bottom}
                  stroke="var(--htr-ink-3)" strokeWidth="0.7" strokeDasharray="3 3" opacity="0.7" />
            {/* Horizontal crosshair line — only inside price area */}
            <line x1={padding.left} y1={cursorY} x2={padding.left + innerW} y2={cursorY}
                  stroke="var(--htr-ink-3)" strokeWidth="0.7" strokeDasharray="3 3" opacity="0.7" />
            {/* Cursor-price tag on right axis edge */}
            <rect x={padding.left + innerW - 50} y={cursorY - 8} width={48} height={16}
                  fill="var(--htr-ink-2)" rx="2" />
            <text x={padding.left + innerW - 4} y={cursorY + 3} fontSize="9.5" fontWeight="700"
                  textAnchor="end" fill="#fff"
                  style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>
              {fmtPrice(cursorPrice, 0)}
            </text>
            {/* OHLCV tooltip box — fixed top-left inside chart */}
            <g transform={`translate(${boxX}, ${boxY})`}>
              <rect width={boxW} height={boxH} fill="var(--htr-surface-2)"
                    stroke="var(--htr-line)" strokeWidth="0.8" rx="3" opacity="0.95" />
              <text x={6} y={13} fontSize="10" fontWeight="700" fill="var(--htr-ink)"
                    style={{ fontFamily: "var(--htr-font-mono)" }}>
                {bar.date || bar.asof || `D-${data.length - 1 - hover.idx}`}
              </text>
              <text x={6} y={28} fontSize="9.5" fill="var(--htr-ink-2)">开 {fmtPrice(bar.open, 0)}</text>
              <text x={6} y={42} fontSize="9.5" fill={upColor}>高 {fmtPrice(bar.high, 0)}</text>
              <text x={6} y={56} fontSize="9.5" fill={dnColor}>低 {fmtPrice(bar.low, 0)}</text>
              <text x={6} y={70} fontSize="9.5" fontWeight="700" fill="var(--htr-ink)">收 {fmtPrice(bar.close, 0)}</text>
              <text x={6} y={84} fontSize="9.5" fill="var(--htr-ink-2)">量 {volStr}</text>
              {chgPct !== null && (
                <text x={6} y={100} fontSize="9.5" fontWeight="700" fill={chgColor}
                      style={{ fontFamily: "var(--htr-font-mono)" }}>
                  涨跌 {chgPct >= 0 ? '+' : ''}{chgPct.toFixed(2)}%
                </text>
              )}
            </g>
          </g>
        );
      })()}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// antiCollideLabels — push overlapping label y positions apart while keeping
// each item's true `priceY`. Used by VerticalLadder / V3LadderMini / KLineChart
// to render price dots at true positions but separate the text labels.
// ─────────────────────────────────────────────────────────────────────────────

function antiCollideLabels(items, yOfPrice, { gap = 20, top = 0, bottom = null } = {}) {
  const out = items.map((r) => ({ ...r, priceY: yOfPrice(r.price), labelY: yOfPrice(r.price) }));
  out.sort((a, b) => a.priceY - b.priceY);
  // Top-down: push each label at least `gap` below the previous.
  for (let i = 1; i < out.length; i++) {
    if (out[i].labelY < out[i - 1].labelY + gap) {
      out[i].labelY = out[i - 1].labelY + gap;
    }
  }
  // Bottom-up: if we exceeded bottom, walk back up to respect bound.
  if (bottom != null) {
    for (let i = out.length - 1; i >= 0; i--) {
      const limit = bottom - gap * (out.length - 1 - i);
      if (out[i].labelY > limit) out[i].labelY = limit;
    }
  }
  // Top-down: respect top bound.
  for (let i = 0; i < out.length; i++) {
    const limit = top + gap * i;
    if (out[i].labelY < limit) out[i].labelY = limit;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// VerticalLadder — standalone vertical price ladder (V2, V4)
// ─────────────────────────────────────────────────────────────────────────────

function VerticalLadder({ ladder, currentPrice, height = 360 }) {
  if (!ladder || !ladder.length) return null;
  const allP = [...ladder.map((r) => r.price), currentPrice];
  const min = Math.min(...allP), max = Math.max(...allP);
  const pad = (max - min) * 0.10;
  const yMin = min - pad, yMax = max + pad;
  const W = 240;
  const axisX = 64;
  const labelX = 78;
  const yOf = (p) => 14 + ((yMax - p) / (yMax - yMin)) * (height - 28);
  return (
    <svg width={W} height={height} style={{ display: "block" }}>
      {/* axis */}
      <line x1={axisX} y1={8} x2={axisX} y2={height - 8} stroke="var(--htr-line)" strokeWidth="1" />
      {/* current price marker */}
      <g>
        <line x1={axisX - 6} y1={yOf(currentPrice)} x2={W - 4} y2={yOf(currentPrice)}
              stroke="var(--htr-accent)" strokeWidth="1.4" />
        <rect x={4} y={yOf(currentPrice) - 9} width={56} height={18} fill="var(--htr-accent)" rx="3" />
        <text x={32} y={yOf(currentPrice) + 4} fontSize="10.5" fontWeight="700" textAnchor="middle" fill="#fff"
              style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>
          现价
        </text>
        <text x={W - 6} y={yOf(currentPrice) - 4} fontSize="10.5" fontWeight="700" textAnchor="end" fill="var(--htr-accent)"
              style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>
          {fmtPrice(currentPrice, 0)}
        </text>
      </g>
      {antiCollideLabels(ladder, yOf, { gap: 26, top: 18, bottom: height - 10 }).map((row, i) => {
        const isStop  = row.kind === "stop";
        const isExit  = row.kind.startsWith("exit");
        const isEntry = row.kind.startsWith("entry");
        const color = isStop ? "var(--htr-bear)" : isExit ? "var(--htr-info)" : isEntry ? "var(--htr-bull)" : "var(--htr-ink-2)";
        // Connector line: from price dot on axis to label anchor (handles
        // non-trivial vertical offset when prices cluster).
        return (
          <g key={i}>
            <circle cx={axisX} cy={row.priceY} r="4" fill={color} stroke="var(--htr-surface)" strokeWidth="1.6" />
            <path d={`M${axisX + 5},${row.priceY} L${labelX - 4},${row.labelY}`}
                  stroke={color} strokeWidth="0.7" fill="none" opacity="0.7" />
            <text x={labelX} y={row.labelY - 3} fontSize="10" fontWeight="600" fill={color}>{row.label}</text>
            <text x={labelX} y={row.labelY + 9} fontSize="10.5" fontWeight="700" fill="var(--htr-ink)"
                  style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>
              {fmtPrice(row.price, 0)}
              <tspan dx="6" fill="var(--htr-ink-3)" fontWeight="500">{fmtPct(row.pct, 1)}</tspan>
            </text>
            <text x={axisX - 8} y={row.priceY + 4} fontSize="9" fill="var(--htr-ink-4)" textAnchor="end">
              {String(i + 1).padStart(2, "0")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// NewsTimeline — vertical chrono list with weight indicators
// ─────────────────────────────────────────────────────────────────────────────

function NewsTimeline({ items, max = 10, compact = false }) {
  const rows = items.slice(0, max);
  return (
    <div>
      {rows.map((n, i) => {
        const dot = n.weight === "high" ? "var(--htr-bear)" : n.weight === "medium" ? "var(--htr-warn)" : "var(--htr-ink-3)";
        return (
          <div key={i} style={{
            display: "grid",
            gridTemplateColumns: "70px 8px 1fr",
            gap: 10, padding: compact ? "7px 0" : "10px 0",
            borderBottom: i < rows.length - 1 ? "1px solid var(--htr-line-soft)" : "none",
            alignItems: "start",
          }}>
            <div style={{ fontFamily: "var(--htr-font-mono)", fontSize: 10, color: "var(--htr-ink-3)", paddingTop: 2 }}>
              {n.ts}
            </div>
            <div style={{ position: "relative", height: "100%", paddingTop: 5 }}>
              <span style={{
                position: "absolute", left: 0, top: 5,
                width: 8, height: 8, borderRadius: "50%",
                background: dot, boxShadow: "0 0 0 2px var(--htr-surface)",
              }} />
              {i < rows.length - 1 && (
                <span style={{
                  position: "absolute", left: 3.5, top: 14, bottom: -14,
                  width: 1, background: "var(--htr-line)",
                }} />
              )}
            </div>
            <div>
              <div style={{ fontSize: "var(--htr-fs-sm)", color: "var(--htr-ink)", lineHeight: 1.4 }}>
                {n.text}
              </div>
              <div style={{ fontSize: 10, color: "var(--htr-ink-3)", marginTop: 3, display: "flex", gap: 6, flexWrap: "wrap" }}>
                <span>{n.src}</span>
                <span>·</span>
                {(n.linkedSymbols || []).slice(0, 3).map((s) => (
                  <span key={s} className="htr-chip" style={{ padding: "0px 5px", fontSize: 9.5 }}>{s}</span>
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GateFlow — §10 八阶门槛 horizontal step flow (better than chips)
// ─────────────────────────────────────────────────────────────────────────────

function GateFlow({ gates, compact = false }) {
  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: 0, width: "100%" }}>
      {gates.map((g, i) => {
        const isDone = g.status === "done";
        const isCur  = g.status === "in_progress";
        const isBlk  = g.status === "blocked";
        const bg = isDone ? "var(--htr-bull-bg)" : isCur ? "var(--htr-warn-bg)" : isBlk ? "var(--htr-bear-bg)" : "var(--htr-surface-2)";
        const fg = isDone ? "var(--htr-bull)" : isCur ? "var(--htr-warn)" : isBlk ? "var(--htr-bear)" : "var(--htr-ink-3)";
        return (
          <div key={g.id} style={{
            flex: 1, padding: compact ? "6px 8px" : "9px 10px",
            background: bg, borderTop: "1px solid var(--htr-line)", borderBottom: "1px solid var(--htr-line)",
            borderLeft: i === 0 ? "1px solid var(--htr-line)" : "none",
            borderRight: "1px solid var(--htr-line)",
            position: "relative",
            clipPath: i < gates.length - 1
              ? "polygon(0 0, calc(100% - 8px) 0, 100% 50%, calc(100% - 8px) 100%, 0 100%, 8px 50%)"
              : "polygon(0 0, 100% 0, 100% 100%, 0 100%, 8px 50%)",
            marginLeft: i === 0 ? 0 : -8,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{
                width: 14, height: 14, borderRadius: "50%", display: "inline-flex",
                alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700,
                background: fg, color: "var(--htr-surface)",
              }}>{g.glyph}</span>
              <span style={{ fontSize: 10, fontWeight: 700, color: fg, letterSpacing: "0.04em" }}>{g.id}</span>
            </div>
            <div style={{ fontSize: compact ? 10.5 : "var(--htr-fs-sm)", fontWeight: 600, color: "var(--htr-ink)", marginTop: 2 }}>
              {g.label}
            </div>
            {!compact && (
              <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", marginTop: 1 }}>{g.next}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ThemeHeatBars
// ─────────────────────────────────────────────────────────────────────────────

function ThemeHeatBars({ themes }) {
  const maxHeat = Math.max(...themes.map((t) => t.heat));
  return (
    <div>
      {themes.map((t, i) => (
        <div key={t.id} style={{
          display: "grid", gridTemplateColumns: "98px 1fr 50px 60px",
          gap: 10, alignItems: "center", padding: "7px 0",
          borderBottom: i < themes.length - 1 ? "1px solid var(--htr-line-soft)" : "none",
        }}>
          <div style={{ fontSize: "var(--htr-fs-sm)", fontWeight: 600 }}>{t.label}</div>
          <div style={{ height: 6, background: "var(--htr-line-soft)", borderRadius: 2, overflow: "hidden" }}>
            <div style={{
              width: `${(t.heat / 100) * 100}%`, height: "100%",
              background: t.heat >= 70 ? "var(--htr-bear)" : t.heat >= 50 ? "var(--htr-warn)" : "var(--htr-info)",
            }} />
          </div>
          <div className="htr-num" style={{ fontSize: "var(--htr-fs-sm)", fontWeight: 700, textAlign: "right" }}>
            {t.heat}°
          </div>
          <div className={"htr-num " + changeClass(t.mom)} style={{ fontSize: 11, textAlign: "right" }}>
            {arrow(t.mom)} {Math.abs(t.mom).toFixed(1)}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CandidateRow — used in lists
// ─────────────────────────────────────────────────────────────────────────────

function CandidateRow({ c, dense = false, onClick, active = false }) {
  // P8-18 — `active` highlights the user's currently selected symbol (drives
  // the leader card / K-line drill). Keyboard accessible when interactive.
  const interactive = typeof onClick === "function";
  return (
    <div
      onClick={interactive ? onClick : undefined}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onKeyDown={interactive ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } } : undefined}
      style={{
        display: "grid",
        gridTemplateColumns: "28px 76px 1fr 80px 70px 56px",
        gap: 10, alignItems: "center",
        padding: dense ? "6px 10px" : "9px 12px",
        borderBottom: "1px solid var(--htr-line-soft)",
        borderLeft: active ? "3px solid var(--htr-accent)" : "3px solid transparent",
        cursor: interactive ? "pointer" : "default",
        background: active ? "var(--htr-accent-soft)"
                  : c.rank === 1 ? "var(--htr-surface-2)" : "transparent",
        userSelect: "none",
      }}>
      <div style={{ fontFamily: "var(--htr-font-mono)", fontSize: 11, color: "var(--htr-ink-3)" }}>
        #{String(c.rank).padStart(2, "0")}
      </div>
      <div>
        <div style={{ fontFamily: "var(--htr-font-mono)", fontSize: "var(--htr-fs-sm)", fontWeight: 700, color: "var(--htr-ink)" }}>
          {c.symbol}
        </div>
        <div style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{c.nameCn}</div>
      </div>
      <div>
        <div style={{ fontSize: "var(--htr-fs-sm)", color: "var(--htr-ink)" }}>{c.theme}</div>
        <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)", marginTop: 1 }}>{c.priority}</div>
      </div>
      <div className="htr-num" style={{ textAlign: "right" }}>
        <div style={{ fontSize: "var(--htr-fs-sm)", fontWeight: 600 }}>{fmtPrice(c.price, c.price > 1000 ? 0 : 2)}</div>
        <div style={{ fontSize: 10 }} className={changeClass(c.chg)}>{fmtPct(c.chg, 2)}</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div className="htr-num" style={{ fontSize: "var(--htr-fs-md)", fontWeight: 700, color: "var(--htr-ink)" }}>{c.score}</div>
        <div style={{ fontSize: 9.5, color: "var(--htr-ink-3)" }}>研究分</div>
      </div>
      <ScoreBar value={c.score} />
    </div>
  );
}

function ScoreBar({ value, max = 100 }) {
  return (
    <div style={{ width: 50, height: 4, background: "var(--htr-line-soft)", borderRadius: 2, overflow: "hidden" }}>
      <div style={{
        width: `${(value / max) * 100}%`, height: "100%",
        background: value >= 70 ? "var(--htr-bull)" : value >= 50 ? "var(--htr-warn)" : "var(--htr-info)",
      }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionLog
// ─────────────────────────────────────────────────────────────────────────────

function DecisionLog({ entries, max = 8 }) {
  const items = entries.slice(0, max);
  return (
    <div>
      {items.map((e, i) => (
        <div key={i} style={{
          display: "grid",
          gridTemplateColumns: "92px 56px 1fr 38px",
          gap: 8, alignItems: "center",
          padding: "6px 10px",
          fontSize: "var(--htr-fs-sm)",
          borderBottom: "1px solid var(--htr-line-soft)",
        }}>
          <span style={{ fontFamily: "var(--htr-font-mono)", fontSize: 10, color: "var(--htr-ink-3)" }}>{e.ts}</span>
          <span style={{ fontFamily: "var(--htr-font-mono)", fontSize: 11, fontWeight: 600 }}>{e.symbol}</span>
          <span style={{ color: "var(--htr-ink-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            <code style={{ background: "var(--htr-surface-3)", padding: "1px 5px", borderRadius: 3, fontSize: 10, marginRight: 6, color: "var(--htr-ink-2)" }}>
              {e.action}
            </code>
            {e.note}
          </span>
          <span className="htr-num" style={{ fontSize: 11, fontWeight: 700, textAlign: "right", color: e.score == null ? "var(--htr-ink-4)" : "var(--htr-ink)" }}>
            {e.score == null ? "—" : e.score}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook — auto-pulse hero price every few seconds for "live" feel
// ─────────────────────────────────────────────────────────────────────────────

function useTickingPrice(base, opts = {}) {
  const { intervalMs = 3200, amplitude = 0.0008 } = opts;
  const [price, setPrice] = useState(base);
  useEffect(() => {
    const id = setInterval(() => {
      setPrice((p) => {
        const delta = (Math.random() - 0.5) * 2 * amplitude * base;
        return Math.round((p + delta) * 100) / 100;
      });
    }, intervalMs);
    return () => clearInterval(id);
  }, [base]);
  return price;
}

// ─────────────────────────────────────────────────────────────────────────────
// P8-16 Cycle 3 — Term: hoverable jargon tooltip. `<Term k="七档阶梯">七档阶梯</Term>`
// (k optional; defaults to children text). Definitions live in GLOSSARY below.
// §9.4 constraint: definitions never say "high score = high win rate".
// ─────────────────────────────────────────────────────────────────────────────

const GLOSSARY = {
  // Dashboard core concepts
  "七档阶梯":           "7 个建议价位 = 3 个买入档（激进/均衡/保守）+ 止损 + 3 个卖出档（卖1/卖2/延伸）。从 ATR 或日内波动算出，仅作研究参考，非订单。",
  "未校准研究分":       "0-100 的研究分数。还没经过 ≥100 次「预测 → 实际结果」配对校准，所以不能当胜率读。",
  "校准":               "把研究分数和真实结果对照，得到 Brier score / log loss / 校准箱。样本不达标前一律标 insufficient。",
  "校准样本不足":       "P9-03 校准引擎要求至少 100 对（预测 + 结果）才能输出校准胜率。当前样本 < 100，所有分数仍是研究分。",
  "Brier score":        "校准准确度，0 是完美 / 0.25 = 全猜 50%。越低越好。仅在样本 ≥100 时报告。",
  "Log loss":           "另一种校准指标，对极端错判惩罚更重。越低越好。仅在样本 ≥100 时报告。",
  "calibrated_probability": "经校准的胜率概率。当前 0/100，未达阈值，不会显示此标签。",

  // Alpha factors (P8-12 themes)
  "mom_20":             "20 日动量：过去 20 个交易日的累计收益率（z-score 化）。",
  "mom_60":             "60 日动量。",
  "mom_consist":        "动量持续性：动量的稳定程度。",
  "mom_12_1":           "12-1 动量：过去 12 个月收益减去最近 1 个月（避免 reversal）。",
  "vol_adj_mom20":      "波动率调整后的 20 日动量。控制波动率后的动量强弱。",
  "sharpe_20":          "20 日 Sharpe：收益 / 波动率，越大越好（风险调整后的收益）。",
  "sharpe_60":          "60 日 Sharpe。",
  "Sortino":            "类似 Sharpe，但只罚下行波动（上涨不算波动）。",
  "sortino_60":         "60 日 Sortino。",
  "vol_z":              "成交量异动 z-score：今日量比 N 日均量高几个标准差。",
  "vol_stability":      "波动率稳定度：波动率本身的标准差，越小越稳。",
  "high52w":            "距 52 周新高的距离。常用 RS 因子。",
  "ma_gap":             "均线 gap：现价相对均线的偏离度。",
  "value_bp":           "估值 B/P 比 (book/price)，越大越价值。",
  "ADV":                "Average Dollar Volume，日均成交额，流动性指标。",
  "adv_rank":           "ADV 在全部股票里的百分位排名 (0..1)。",
  "z-score":            "标准化分数：(值 - 均值) / 标准差。|z| ≥ 2 算极端。",

  // Markets (P8-11)
  "TOPIX":              "东京证券交易所综合指数。这里用 1306.T ETF 现价当代理（DB 无原生 TOPIX 列）。",
  "1306.T":             "野村东证 ETF（NEXT FUNDS TOPIX ETF），跟踪 TOPIX。也是用户当前 Path A live 持仓。",
  "Nikkei Futures":     "日经 225 期货。这里用作 N225 现货的开盘前指引代理。",
  "SOX":                "费城半导体指数。隔夜走势对日股半导体板块有引领作用。",
  "USDJPY":             "美元/日元汇率。日元贬值（数字↑）通常利好日本出口股，所以仪表盘对此做了反向温度映射。",
  "USDJPY inverse temp":"USD/JPY 跌（日元升）→ 出口股利空 → 温度调降；涨 → 温度调升。",
  "VIX":                "S&P 500 隐含波动率指数。VIX↑通常是避险信号。",

  // Positions / portfolio
  "NAV":                "Net Asset Value = 现金 + 持仓市值。账户总价值。",
  "浮动盈亏":           "Unrealized P&L = (现价 - 均价) × 数量。还没卖出锁定的纸面盈亏。",
  "均价":               "Average Cost = 多次加仓的成本均值。",
  "etf_buyhold":        "策略 ID：被动持有 ETF。用户 Path A 当前用此策略。",
  "sprint":             "策略 ID：已下线的 sprint 短线策略（历史 3041.T 持仓快照所在）。",

  // Governance / pipeline
  "§10":                "治理规则第 10 节 — 自动化分 8 阶门槛，按顺序解锁：候选 → 阶梯 → 决策日志 → 反馈 → 校准 → 提醒 → 纸面 → 实盘。",
  "§10 gate":           "§10 八阶门槛的某一阶，状态 done / pending / blocked。",
  "§8.6":               "治理规则第 8.6 节 — 每条预测都必须落盘到 decision log，含 prediction_id / model_version / decision_cutoff 等可重现字段。",
  "§9.4":               "治理规则第 9.4 节 — 校准前所有分数必须标 uncalibrated_research_score 或 insufficient_calibration，不得当胜率展示。",
  "decision_cutoff":    "决策时间锚点。所有用作 ex-ante 决策的输入必须 available_ts ≤ decision_cutoff，否则视为 lookahead 而拒绝。",
  "prediction_id":      "每条预测的唯一稳定哈希，sha256(snapshot|model|cutoff|symbol)[:16]，重跑同样输入永远同 id。",
  "advice-only":        "项目硬约束：未通过 paper gate 前不得自动下单，仪表盘只输出人工可读的研究建议。",

  // Screener
  "screener_v2":        "Project_optimized 的每日选股器输出，含 alpha 分数 + 流动性 + 基本面分。",
  "selected_tickers":   "当日 screener 的 top-N 短名单（默认 10 个）。",
  "alpha_score":        "screener 综合分（0..1），加权多个 alpha 因子。",
  "hard_fail":          "screener 把流动性/估值等硬条件不达标的标的拒掉。",
  "fundamental_score":  "基本面分数（0..1），来自财务质量 / 增长 / 估值合成。",

  // Dashboard sections (P8-16 Cycle 4)
  "主题热力":           "把市场分为若干主题（这里用 alpha factor 当主题），按强度排序。热度越高代表该主题当下越被资金推动。",
  "决策日志":           "§8.6 强制：每次扫描的候选、价位、模型版本都按日落盘到 reports/predictions/。",
  "新闻催化":           "最近 N 小时新闻摘要，按重要程度（urgency + sentiment）标 high/medium/low。",
  "市场温度":           "把多个市场（日股/美股/SOX/USDJPY 等）的近期表现合成 0-100 温度计，> 70 过热 / 30-50 平温 / < 30 冷。",
  "MA20":               "20 日收盘价简单移动平均线。短期趋势参考。",
  "MA60":               "60 日收盘价简单移动平均线。中期趋势参考。",
  "52w 高":             "过去 252 个交易日的最高价。突破 = 强势信号。",
  "52w 低":             "过去 252 个交易日的最低价。跌破 = 弱势信号。",
};

// P8-16 C2 follow-up — let consumers size SVG charts dynamically to fill
// their parent container. Returns [ref, {width, height}]. Attach `ref` to the
// containing element. `width/height` update on resize.
function useElementSize() {
  const ref = React.useRef(null);
  const [size, setSize] = React.useState({ width: 0, height: 0 });
  React.useEffect(() => {
    if (!ref.current) return;
    const measure = (entry) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.round(width), height: Math.round(height) });
    };
    const ro = new ResizeObserver((entries) => measure(entries[0]));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

function Term({ children, k }) {
  const key = k || (typeof children === "string" ? children : "");
  const def = GLOSSARY[key];
  if (!def) return children;
  return (
    <span className="htr-term" tabIndex={0}>
      {children}
      <span className="htr-term-pop" role="tooltip">
        <span className="htr-term-pop-title">{key}</span>
        <span className="htr-term-pop-body">{def}</span>
      </span>
    </span>
  );
}

// ─── P8-18 Interactive Exploration Hooks (Rule 11) ──────────────────────────
// All three hooks are pure read-only — they GET from /api/symbol/{ticker}/*
// and never POST. Selection lives in localStorage (user-state, Rule 11.3).

function useSelectedSymbol(defaultSymbol) {
  // Read once from localStorage on mount; if absent, use the prop default
  // (typically `candidates[0].symbol`). Never persist a falsy value.
  const [symbol, setSymbol] = React.useState(() => {
    try {
      const stored = typeof localStorage !== "undefined" && localStorage.getItem("htr_symbol");
      return stored || defaultSymbol || "";
    } catch (_e) {
      return defaultSymbol || "";
    }
  });
  React.useEffect(() => {
    if (!symbol) return;
    try { localStorage.setItem("htr_symbol", symbol); } catch (_e) { /* ignore */ }
  }, [symbol]);
  return [symbol, setSymbol];
}

function useSymbolKline(symbol, { sessions = 252, fallback = null } = {}) {
  // Fetches /api/symbol/{ticker}/kline?sessions=N on every `symbol` change.
  // While loading or on error, returns `fallback` (typically the dashboard's
  // initial kline) so the chart never goes blank during transitions.
  const [state, setState] = React.useState({
    bars: fallback || [], loading: !!symbol, error: null, source: "fallback",
  });
  React.useEffect(() => {
    if (!symbol) {
      setState({ bars: fallback || [], loading: false, error: null, source: "fallback" });
      return;
    }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetch(`/api/symbol/${encodeURIComponent(symbol)}/kline?sessions=${sessions}`, { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      })
      .then((payload) => {
        if (cancelled) return;
        setState({ bars: payload.bars, loading: false, error: null, source: "api" });
      })
      .catch((err) => {
        if (cancelled) return;
        setState({ bars: fallback || [], loading: false, error: err.message, source: "fallback" });
      });
    return () => { cancelled = true; };
  }, [symbol, sessions]);
  return state;
}

function useSymbolProfile(symbol) {
  const [state, setState] = React.useState({
    profile: null, loading: !!symbol, error: null,
  });
  React.useEffect(() => {
    if (!symbol) { setState({ profile: null, loading: false, error: null }); return; }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetch(`/api/symbol/${encodeURIComponent(symbol)}/profile`, { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      })
      .then((profile) => {
        if (cancelled) return;
        setState({ profile, loading: false, error: null });
      })
      .catch((err) => {
        if (cancelled) return;
        setState({ profile: null, loading: false, error: err.message });
      });
    return () => { cancelled = true; };
  }, [symbol]);
  return state;
}

// P10-06 — LLM brief hook. On-demand only: caller invokes fetch() because
// cold-start can take ~30s and we do not want to spam Ollama on every symbol
// click. Resets to idle when symbol changes (prevents stale brief from a
// previous ticker leaking onto a new selection).
function useLlmBrief(symbol) {
  const [state, setState] = React.useState({
    brief: null, loading: false, error: null,
  });
  React.useEffect(() => {
    setState({ brief: null, loading: false, error: null });
  }, [symbol]);
  const fetchBrief = React.useCallback(() => {
    if (!symbol) return;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetch(`/api/symbol/${encodeURIComponent(symbol)}/llm_brief`, { cache: "no-store" })
      .then(async (r) => {
        if (!r.ok) {
          let body = null;
          try { body = await r.json(); } catch (e) {}
          const reason = body && body.detail && body.detail.reason
            ? body.detail.reason
            : `HTTP ${r.status}`;
          const msg = body && body.detail && body.detail.message
            ? `${reason}: ${body.detail.message}`
            : reason;
          throw new Error(msg);
        }
        return r.json();
      })
      .then((brief) => setState({ brief, loading: false, error: null }))
      .catch((err) => setState({ brief: null, loading: false, error: err.message }));
  }, [symbol]);
  return { ...state, fetch: fetchBrief };
}

// Export to window for cross-file Babel use
Object.assign(window, {
  AnimatedPrice, Sparkline, TempGauge, MarketTempCell,
  CalibrationBadge, KLineChart, VerticalLadder, NewsTimeline,
  GateFlow, ThemeHeatBars, CandidateRow, ScoreBar, DecisionLog,
  useTickingPrice, Term, GLOSSARY, useElementSize,
  useSelectedSymbol, useSymbolKline, useSymbolProfile, useLlmBrief,
  HTR: { fmtPrice, fmtPct, changeClass, arrow },
});
