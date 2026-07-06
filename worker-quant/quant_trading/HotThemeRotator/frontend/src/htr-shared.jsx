// htr-shared.jsx — refined design tokens + reusable components for the V3 redesign.
// Visual refinements over the handoff baseline:
//   • Legibility floor (Rule 11.7.4): default body ≥ 12px, numerics ≥ 11.5px;
//     even "compact" density stays ≥ 11px / 10.5px.
//   • Calmer hierarchy: fewer hard borders, more whitespace + surface tints,
//     softer eyebrow letter-spacing, larger headline scale.
//   • Same warm-paper palette, IBM Plex family, semantic up/down/warn colors.

const { useState, useEffect, useRef, useMemo, Fragment } = React;

const TOKEN_STYLE = `
:root {
  --htr-bg:#F4F1EA; --htr-surface:#FFFFFF; --htr-surface-2:#FAF8F2; --htr-surface-3:#EEEBE1;
  --htr-line:#E6E1D4; --htr-line-2:#D4CEBC; --htr-line-soft:#F0EDE3;

  /* P5-D — ink-3 darkened #827D71→#71695C for WCAG AA on muted text over cream */
  --htr-ink:#1A1815; --htr-ink-2:#4C4842; --htr-ink-3:#71695C; --htr-ink-4:#ABA597;

  --htr-bull:#1F7A4D; --htr-bull-bg:#E7F1EA; --htr-bear:#B23A3A; --htr-bear-bg:#F7E7E3;
  --htr-warn:#B0742C; --htr-warn-bg:#FAEDD9; --htr-info:#355D8C; --htr-info-bg:#E6ECF4;
  /* P3-F — "hot" market temperature (>=70) gets its own warm vermilion ramp so an
     overheated tile never shares the loss/stop red (--htr-bear). */
  --htr-heat-hot:#D2451E; --htr-heat-hot-bg:#FBE8DD;

  --htr-accent:#1F3A5F; --htr-accent-soft:#E5EAF1; --htr-accent-ink:#FFFFFF;

  --htr-font-sans:"IBM Plex Sans","Noto Sans JP",system-ui,sans-serif;
  --htr-font-serif:"IBM Plex Serif","Noto Serif JP",Georgia,serif;
  --htr-font-mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
  --htr-font-body:var(--htr-font-sans);
  --htr-font-num:var(--htr-font-mono);

  --htr-pad-x:16px; --htr-pad-y:12px; --htr-gap:14px;
  --htr-radius:8px; --htr-radius-lg:12px;

  /* type scale — readable default */
  --htr-fs-xs:11px; --htr-fs-sm:12px; --htr-fs-md:13.5px; --htr-fs-lg:16px;
  --htr-fs-xl:20px; --htr-fs-2xl:27px; --htr-fs-3xl:40px; --htr-fs-num:14.5px;

  --htr-shadow:0 1px 2px rgba(26,24,21,.04), 0 1px 1px rgba(26,24,21,.03);
  --htr-shadow-lg:0 8px 30px rgba(26,24,21,.12);
}
:root[data-htr-dark="true"] {
  --htr-bg:#13120F; --htr-surface:#1B1A16; --htr-surface-2:#201E19; --htr-surface-3:#272520;
  --htr-line:#302D27; --htr-line-2:#3D3A32; --htr-line-soft:#211F1A;
  --htr-ink:#ECE9DF; --htr-ink-2:#C2BEB1; --htr-ink-3:#8C887C; --htr-ink-4:#5C5950;
  --htr-bull:#5BB57E; --htr-bull-bg:#1B2C22; --htr-bear:#DC6F6F; --htr-bear-bg:#2D1E1E;
  --htr-warn:#D9A45E; --htr-warn-bg:#2A2218; --htr-info:#7AA8D9; --htr-info-bg:#1B232D;
  --htr-heat-hot:#E8794A; --htr-heat-hot-bg:#2E1F16;
  --htr-accent-soft:#1F2A38; --htr-accent-ink:#ECE9DF;
  --htr-shadow:0 1px 2px rgba(0,0,0,.4);
  --htr-shadow-lg:0 8px 30px rgba(0,0,0,.5);
}
/* Compact still respects the Rule 11.7.4 legibility floor */
:root[data-htr-density="compact"] {
  --htr-pad-x:11px; --htr-pad-y:8px; --htr-gap:10px;
  --htr-fs-xs:11px; --htr-fs-sm:11px; --htr-fs-md:12px; --htr-fs-lg:14px;
  --htr-fs-num:12.5px;
}
:root[data-htr-density="comfy"] {
  --htr-pad-x:20px; --htr-pad-y:16px; --htr-gap:18px;
  --htr-fs-sm:13px; --htr-fs-md:15px; --htr-fs-lg:18px; --htr-fs-num:16px;
}

.htr { font-family:var(--htr-font-body); color:var(--htr-ink);
  font-feature-settings:"tnum","ss01"; -webkit-font-smoothing:antialiased; }
.htr, .htr * { box-sizing:border-box; }
.htr-num { font-family:var(--htr-font-num); font-variant-numeric:tabular-nums; letter-spacing:-0.01em; }
.htr-mono { font-family:var(--htr-font-mono); }
.htr-serif { font-family:var(--htr-font-serif); }
.htr-up { color:var(--htr-bull); } .htr-down { color:var(--htr-bear); } .htr-flat { color:var(--htr-ink-3); }

.htr-eyebrow { font-size:var(--htr-fs-xs); letter-spacing:0.1em; color:var(--htr-ink-3);
  text-transform:uppercase; font-weight:600; }
.htr-divider { border:0; border-top:1px solid var(--htr-line); margin:0; }

.htr-card { background:var(--htr-surface); border:1px solid var(--htr-line);
  border-radius:var(--htr-radius); box-shadow:var(--htr-shadow); }

.htr-chip { display:inline-flex; align-items:center; gap:5px; padding:2px 8px;
  font-size:var(--htr-fs-xs); border-radius:999px; border:1px solid var(--htr-line-2);
  background:var(--htr-surface-2); color:var(--htr-ink-2); letter-spacing:0.01em; white-space:nowrap; }
.htr-chip.bull { color:var(--htr-bull); border-color:color-mix(in oklab,var(--htr-bull) 30%,var(--htr-line-2)); background:var(--htr-bull-bg); }
.htr-chip.bear { color:var(--htr-bear); border-color:color-mix(in oklab,var(--htr-bear) 30%,var(--htr-line-2)); background:var(--htr-bear-bg); }
.htr-chip.warn { color:var(--htr-warn); border-color:color-mix(in oklab,var(--htr-warn) 30%,var(--htr-line-2)); background:var(--htr-warn-bg); }
.htr-chip.info { color:var(--htr-info); border-color:color-mix(in oklab,var(--htr-info) 30%,var(--htr-line-2)); background:var(--htr-info-bg); }
.htr-chip.accent { color:var(--htr-accent); border-color:color-mix(in oklab,var(--htr-accent) 30%,var(--htr-line-2)); background:var(--htr-accent-soft); }

@keyframes htr-pulse-up { 0% { background:rgba(31,122,77,0.16); } 100% { background:transparent; } }
@keyframes htr-pulse-down { 0% { background:rgba(178,58,58,0.16); } 100% { background:transparent; } }
.htr-pulse-up { animation:htr-pulse-up 1.6s ease-out; }
.htr-pulse-down { animation:htr-pulse-down 1.6s ease-out; }

.htr ::-webkit-scrollbar { width:9px; height:9px; }
.htr ::-webkit-scrollbar-thumb { background:var(--htr-line-2); border-radius:5px; border:2px solid var(--htr-surface); }
.htr ::-webkit-scrollbar-track { background:transparent; }

.htr-term { border-bottom:1px dotted var(--htr-ink-4); cursor:help; position:relative; }
.htr-term:focus { outline:none; }
/* P5-A (WCAG 2.4.7) — visible keyboard focus ring on every interactive element.
   :focus-visible keeps mouse clicks clean while restoring a ring for keyboard nav
   (the blanket outline:none above is overridden for keyboard focus). */
.htr-term:focus-visible,
.htr [role="button"]:focus-visible,
.htr [tabindex="0"]:focus-visible,
.nav-btn:focus-visible,
.htr-chip:focus-visible,
button:focus-visible { outline:2px solid var(--htr-accent); outline-offset:2px; border-radius:3px; }
.htr-term-pop { display:none; position:absolute; bottom:calc(100% + 6px); left:0; width:280px;
  padding:9px 12px 10px; background:var(--htr-ink); color:var(--htr-bg); border-radius:6px;
  box-shadow:var(--htr-shadow-lg); font-size:11.5px; line-height:1.5; z-index:300;
  white-space:normal; pointer-events:none; font-family:var(--htr-font-body); }
.htr-term-pop-title { display:block; font-family:var(--htr-font-mono); font-size:10.5px; font-weight:700;
  color:var(--htr-warn); margin-bottom:3px; }
.htr-term:hover .htr-term-pop, .htr-term:focus .htr-term-pop { display:block; }
`;

if (!document.getElementById('htr-tokens')) {
  const s = document.createElement('style');
  s.id = 'htr-tokens'; s.textContent = TOKEN_STYLE;
  document.head.appendChild(s);
}

// ── helpers ──
function fmtPrice(p, decimals = 2) {
  if (p == null || isNaN(p)) return "—";
  return p.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}
function fmtPct(p, decimals = 2, withSign = true) {
  if (p == null || isNaN(p)) return "—";
  const sign = withSign && p > 0 ? "+" : "";
  return `${sign}${p.toFixed(decimals)}%`;
}
function changeClass(n) { if (n == null || isNaN(n)) return "htr-flat"; if (n > 0) return "htr-up"; if (n < 0) return "htr-down"; return "htr-flat"; }
function arrow(n) { if (n > 0) return "▲"; if (n < 0) return "▼"; return "·"; }
function heatColor(t) { return t >= 70 ? "var(--htr-heat-hot)" : t >= 50 ? "var(--htr-warn)" : t >= 30 ? "var(--htr-info)" : "var(--htr-ink-3)"; }
function heatBg(t) { return t >= 70 ? "var(--htr-heat-hot-bg)" : t >= 50 ? "var(--htr-warn-bg)" : t >= 30 ? "var(--htr-info-bg)" : "var(--htr-surface-3)"; }

// P3-C — canonical research-score / calibration wording. Single source so the
// no-probability qualifier never drifts across variants (the divergence that
// produces fabrications like the V4 "(+12)"). Rule 8.3 / 9.4.
const HTR_LABELS = {
  scoreUncalibrated: "研究分 · 未校准",
  uncalibrated: "未校准",
  rule94Note: "Rule 9.4 — 分数未经 K-fold OOS 验证，仅作 ranking 信号，不可视作概率。",
};

// ── AnimatedPrice ──
function AnimatedPrice({ value, decimals = 2, style }) {
  const [v, setV] = useState(value);
  const [dir, setDir] = useState(0);
  useEffect(() => {
    if (value === v) return;
    setDir(value > v ? 1 : -1); setV(value);
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

// ── Sparkline ──
function Sparkline({ data, width = 80, height = 22, stroke, fill, area = true }) {
  if (!data || data.length === 0) return null;
  const w = width || 100;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const dx = data.length > 1 ? w / (data.length - 1) : 0;
  const points = data.map((v, i) => [i * dx, height - ((v - min) / range) * height]);
  const path = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const areaPath = path + ` L${w},${height} L0,${height} Z`;
  const last = data[data.length - 1], first = data[0];
  const up = last >= first;
  const s = stroke || (up ? "var(--htr-bull)" : "var(--htr-bear)");
  const f = fill || (up ? "rgba(31,122,77,0.10)" : "rgba(178,58,58,0.10)");
  return (
    <svg viewBox={`0 0 ${w} ${height}`} width={width || "100%"} height={height}
         preserveAspectRatio="none" style={{ display: "block" }}>
      {area && <path d={areaPath} fill={f} />}
      <path d={path} fill="none" stroke={s} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// ── antiCollideLabels (shared by ladder + chart) ──
function antiCollideLabels(items, yOfPrice, { gap = 20, top = 0, bottom = null } = {}) {
  const out = items.map((r) => ({ ...r, priceY: yOfPrice(r.price), labelY: yOfPrice(r.price) }));
  out.sort((a, b) => a.priceY - b.priceY);
  for (let i = 1; i < out.length; i++) {
    if (out[i].labelY < out[i - 1].labelY + gap) out[i].labelY = out[i - 1].labelY + gap;
  }
  if (bottom != null) {
    for (let i = out.length - 1; i >= 0; i--) {
      const limit = bottom - gap * (out.length - 1 - i);
      if (out[i].labelY > limit) out[i].labelY = limit;
    }
  }
  for (let i = 0; i < out.length; i++) {
    const limit = top + gap * i;
    if (out[i].labelY < limit) out[i].labelY = limit;
  }
  return out;
}

function ladderColor(kind) {
  if (kind === "stop") return "var(--htr-bear)";
  if (kind.startsWith("exit")) return "var(--htr-info)";
  if (kind.startsWith("entry")) return "var(--htr-bull)";
  return "var(--htr-ink-2)";
}

// ── KLineChart with crosshair + right-anchored ladder labels ──
function KLineChart({
  data: bars, width = 520, height = 280, ladder = null,
  padding = { top: 16, right: 118, bottom: 22, left: 10 },
  withVolume = true, withMA = true, with52wLines = true,
}) {
  const [hover, setHover] = React.useState(null);
  // Rule 11.1 + 11.4 — mouse-wheel zoom / pan over EXISTING bars only (never
  // fabricates data or precision). `view` = visible window {start,count}; null = full.
  const [view, setView] = React.useState(null);
  const svgRef = React.useRef(null);
  const stateRef = React.useRef({});
  React.useEffect(() => {
    const svg = svgRef.current; if (!svg) return;
    const onWheel = (e) => {
      e.preventDefault();
      const s = stateRef.current; if (!s.innerW) return;
      const rect = svg.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (s.width / rect.width);
      const frac = Math.max(0, Math.min(1, (mx - s.padLeft) / s.innerW));
      const nc = Math.max(20, Math.min(s.total, Math.round(s.count * (e.deltaY < 0 ? 0.8 : 1.25))));
      if (nc === s.count) return;
      const cur = s.start + Math.round(frac * (s.count - 1));
      const ns = Math.max(0, Math.min(s.total - nc, Math.round(cur - frac * (nc - 1))));
      setView(nc >= s.total ? null : { start: ns, count: nc });
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, []);
  if (!bars || !bars.length) return null;
  const total = bars.length;
  const vCount = view ? Math.max(20, Math.min(view.count, total)) : total;
  const vStart = view ? Math.max(0, Math.min(view.start, total - vCount)) : 0;
  const data = bars.slice(vStart, vStart + vCount);   // visible slice — all rendering below uses this
  // Rule 11.7 — responsive right gutter + positive innerW floor so a narrow
  // viewport never yields a non-positive plotting width (broken/blank chart).
  const padRight = Math.max(40, Math.min(padding.right, width - padding.left - 80));
  const innerW = Math.max(40, width - padding.left - padRight);
  const innerH_total = height - padding.top - padding.bottom;
  const volH = withVolume ? Math.round(innerH_total * 0.24) : 0;
  const priceGap = withVolume ? 6 : 0;
  const innerH = innerH_total - volH - priceGap;

  const candleVals = data.flatMap((c) => [c.high, c.low]);
  const ladderVals = ladder ? ladder.map((r) => r.price) : [];
  const allVals = [...candleVals, ...ladderVals];
  const min = Math.min(...allVals), max = Math.max(...allVals);
  const pad = (max - min) * 0.06;
  const yMin = min - pad, yMax = max + pad;
  const yScale = (v) => padding.top + innerH - ((v - yMin) / (yMax - yMin)) * innerH;
  const candleW = Math.max(2, innerW / data.length * 0.68);
  const slot = innerW / data.length;
  stateRef.current = { start: vStart, count: vCount, total, innerW, padLeft: padding.left, width };

  const onMove = (e) => {
    const svg = svgRef.current; if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (width / rect.width);
    const my = (e.clientY - rect.top) * (height / rect.height);
    if (mx < padding.left || mx > padding.left + innerW || my < padding.top || my > padding.top + innerH_total) { setHover(null); return; }
    let idx = Math.floor((mx - padding.left) / slot);
    idx = Math.max(0, Math.min(data.length - 1, idx));
    setHover({ idx, yPx: my });
  };
  const yInv = (yPx) => yMax - ((yPx - padding.top) / innerH) * (yMax - yMin);

  const grid = [];
  for (let i = 0; i <= 5; i++) { const v = yMin + (yMax - yMin) * (i / 5); grid.push({ v, y: yScale(v) }); }

  const rollingMean = (closes, win) => {
    const out = new Array(closes.length).fill(null);
    if (closes.length < win) return out;
    let sum = 0;
    for (let i = 0; i < closes.length; i++) { sum += closes[i]; if (i >= win) sum -= closes[i - win]; if (i >= win - 1) out[i] = sum / win; }
    return out;
  };
  const closes = data.map((c) => c.close);
  // MA computed over the FULL series then sliced to the view, so MA20/MA60 stay
  // the true moving averages when zoomed (not recomputed over only visible bars).
  const fullCloses = bars.map((c) => c.close);
  const ma20 = withMA && fullCloses.length >= 20 ? rollingMean(fullCloses, 20).slice(vStart, vStart + vCount) : null;
  const ma60 = withMA && fullCloses.length >= 60 ? rollingMean(fullCloses, 60).slice(vStart, vStart + vCount) : null;
  const polyline = (series) => series.map((v, i) => v == null ? null : `${padding.left + slot * i + slot / 2},${yScale(v)}`).filter(Boolean).join(" ");

  // 52w window from the FULL series (bars), not the zoomed slice, so the
  // reference high/low lines stay put when the user wheel-zooms (code-review fix).
  const lookback = bars.slice(-Math.min(252, bars.length));
  const w52High = with52wLines ? Math.max(...lookback.map((c) => c.high)) : null;
  const w52Low = with52wLines ? Math.min(...lookback.map((c) => c.low)) : null;
  const w52Label = bars.length >= 252 ? "52w" : `${bars.length}D`;

  const vols = data.map((c) => Number(c.volume ?? c.vol ?? 0));
  const vMax = Math.max(1, ...vols);
  const volTopY = padding.top + innerH + priceGap;
  const volScale = (v) => volTopY + volH - (v / vMax) * volH;

  return (
    <svg ref={svgRef} width={width} height={height} style={{ display: "block", overflow: "visible", cursor: "crosshair" }}
         onMouseMove={onMove} onMouseLeave={() => setHover(null)}
         onDoubleClick={() => setView(null)}>
      {view && (
        <text x={padding.left} y={padding.top - 4} fontSize="9" fill="var(--htr-ink-4)">
          {`显示 ${data.length}/${total} 根 · 滚轮缩放 · 双击复位`}
        </text>
      )}
      {grid.map((g, i) => (
        <line key={i} x1={padding.left} y1={g.y} x2={padding.left + innerW} y2={g.y}
              stroke="var(--htr-line-soft)" strokeWidth="0.7" strokeDasharray="2 3" />
      ))}
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
                  fill={color} opacity={up ? 0.92 : 1} stroke={color} strokeWidth="0.6" />
          </g>
        );
      })}
      {ma20 && <polyline points={polyline(ma20)} fill="none" stroke="var(--htr-info)" strokeWidth="1.3" opacity="0.85" />}
      {ma60 && <polyline points={polyline(ma60)} fill="none" stroke="var(--htr-warn)" strokeWidth="1.3" opacity="0.85" strokeDasharray="4 3" />}
      {(ma20 || ma60) && (
        <g transform={`translate(${padding.left + 6}, ${padding.top + 6})`}>
          {ma20 && <g><line x1={0} y1={4} x2={14} y2={4} stroke="var(--htr-info)" strokeWidth="1.5" /><text x={18} y={7} fontSize="9.5" fill="var(--htr-ink-2)" fontWeight="600">MA20</text></g>}
          {ma60 && <g transform="translate(56,0)"><line x1={0} y1={4} x2={14} y2={4} stroke="var(--htr-warn)" strokeWidth="1.5" strokeDasharray="3 2" /><text x={18} y={7} fontSize="9.5" fill="var(--htr-ink-2)" fontWeight="600">MA60</text></g>}
        </g>
      )}
      {with52wLines && w52High != null && (
        <g>
          <line x1={padding.left} y1={yScale(w52High)} x2={padding.left + innerW} y2={yScale(w52High)} stroke="var(--htr-ink-3)" strokeWidth="0.7" strokeDasharray="6 4" opacity="0.5" />
          <text x={padding.left + innerW - 4} y={yScale(w52High) - 3} fontSize="9.5" textAnchor="end" fill="var(--htr-ink-3)" fontWeight="600">{w52Label} 高 {fmtPrice(w52High, 0)}</text>
          <line x1={padding.left} y1={yScale(w52Low)} x2={padding.left + innerW} y2={yScale(w52Low)} stroke="var(--htr-ink-3)" strokeWidth="0.7" strokeDasharray="6 4" opacity="0.5" />
          <text x={padding.left + innerW - 4} y={yScale(w52Low) + 11} fontSize="9.5" textAnchor="end" fill="var(--htr-ink-3)" fontWeight="600">{w52Label} 低 {fmtPrice(w52Low, 0)}</text>
        </g>
      )}
      {withVolume && (
        <g>
          <line x1={padding.left} y1={volTopY} x2={padding.left + innerW} y2={volTopY} stroke="var(--htr-line)" strokeWidth="0.7" />
          <text x={padding.left + 4} y={volTopY + 10} fontSize="9" fill="var(--htr-ink-3)" fontWeight="600">VOL</text>
          {data.map((c, i) => {
            const v = Number(c.volume ?? c.vol ?? 0); if (!v) return null;
            const x = padding.left + slot * i + slot / 2; const y = volScale(v);
            const up = c.close >= c.open;
            return <rect key={`v${i}`} x={x - candleW / 2} y={y} width={candleW} height={Math.max(0.5, (volTopY + volH) - y)} fill={up ? "var(--htr-bull)" : "var(--htr-bear)"} opacity="0.5" />;
          })}
        </g>
      )}
      {ladder && antiCollideLabels(ladder, yScale, { gap: 21, top: padding.top + 11, bottom: padding.top + innerH - 6 }).map((row, i) => {
        const color = ladderColor(row.kind);
        const bg = row.kind === "stop" ? "var(--htr-bear-bg)" : row.kind.startsWith("exit") ? "var(--htr-info-bg)" : row.kind.startsWith("entry") ? "var(--htr-bull-bg)" : "var(--htr-surface-2)";
        const labelBoxX = padding.left + innerW + 6;
        return (
          <g key={i}>
            <line x1={padding.left} y1={row.priceY} x2={padding.left + innerW + 4} y2={row.priceY} stroke={color} strokeWidth="0.7" strokeDasharray={row.kind === "stop" ? "3 3" : "0"} opacity="0.5" />
            <path d={`M${padding.left + innerW + 4},${row.priceY} L${labelBoxX},${row.labelY}`} stroke={color} strokeWidth="0.6" fill="none" opacity="0.7" />
            <rect x={labelBoxX} y={row.labelY - 9} width={padRight - 12} height={18} fill={bg} stroke={color} strokeWidth="0.7" rx="3" />
            <text x={labelBoxX + 6} y={row.labelY + 4} fontSize="9.5" fontWeight="600" fill={color}>{row.label}</text>
            <text x={padding.left + innerW + padRight - 8} y={row.labelY + 4} fontSize="10" fontWeight="700" textAnchor="end" fill={color} style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>{fmtPrice(row.price, 0)}</text>
          </g>
        );
      })}
      {(() => {
        const last = data[data.length - 1].close; const y = yScale(last);
        return (
          <g>
            <line x1={padding.left} y1={y} x2={padding.left + innerW} y2={y} stroke="var(--htr-accent)" strokeWidth="1.3" />
            <rect x={padding.left + innerW - 56} y={y - 9} width={54} height={18} fill="var(--htr-accent)" rx="3" />
            <text x={padding.left + innerW - 6} y={y + 4} fontSize="10.5" fontWeight="700" textAnchor="end" fill="var(--htr-accent-ink)" style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>{fmtPrice(last, 0)}</text>
          </g>
        );
      })()}
      {(() => {
        // Rule 11.9.2 — axis ticks show the REAL bar date (MM-DD), never a
        // hardcoded "今日": the rightmost bar is the last bar in the series, which
        // may be a stale trading day, not the current wall-clock date.
        const axDate = (b, fb) => { const d = b && b.date; return d ? String(d).slice(5) : fb; };
        const mid = Math.floor((data.length - 1) / 2);
        return (
          <g>
            <text x={padding.left} y={height - 6} fontSize="9.5" fill="var(--htr-ink-3)">{axDate(data[0], `D-${data.length - 1}`)}</text>
            <text x={padding.left + innerW / 2} y={height - 6} fontSize="9.5" fill="var(--htr-ink-3)" textAnchor="middle">{axDate(data[mid], `D-${mid}`)}</text>
            <text x={padding.left + innerW} y={height - 6} fontSize="9.5" fill="var(--htr-ink-3)" textAnchor="end">{axDate(data[data.length - 1], "末根")}</text>
          </g>
        );
      })()}
      {hover && (() => {
        // clamp idx — a stale hover from a wider view can exceed the zoomed slice
        const hIdx = Math.max(0, Math.min(hover.idx, data.length - 1));
        const bar = data[hIdx];
        if (!bar) return null;
        const prev = hIdx > 0 ? data[hIdx - 1] : null;
        const barX = padding.left + slot * hIdx + slot / 2;
        const cursorY = Math.max(padding.top, Math.min(padding.top + innerH, hover.yPx));
        const cursorPrice = yInv(cursorY);
        const chgPct = prev ? (bar.close - prev.close) / prev.close * 100 : null;
        const bvol = bar.volume ?? bar.vol;  // API bars carry `vol`, mock carries `volume`
        const volStr = bvol != null ? (bvol >= 1e8 ? (bvol / 1e8).toFixed(2) + '亿' : bvol >= 1e4 ? (bvol / 1e4).toFixed(1) + '万' : String(Math.round(bvol))) : '—';
        const boxW = 104, boxH = 110, boxX = padding.left + 6, boxY = padding.top + 22;
        const chgColor = chgPct == null ? "var(--htr-ink-2)" : chgPct >= 0 ? "var(--htr-bull)" : "var(--htr-bear)";
        return (
          <g pointerEvents="none">
            <line x1={barX} y1={padding.top} x2={barX} y2={height - padding.bottom} stroke="var(--htr-ink-3)" strokeWidth="0.7" strokeDasharray="3 3" opacity="0.7" />
            <line x1={padding.left} y1={cursorY} x2={padding.left + innerW} y2={cursorY} stroke="var(--htr-ink-3)" strokeWidth="0.7" strokeDasharray="3 3" opacity="0.7" />
            <rect x={padding.left + innerW - 50} y={cursorY - 8} width={48} height={16} fill="var(--htr-ink-2)" rx="3" />
            <text x={padding.left + innerW - 4} y={cursorY + 3} fontSize="9.5" fontWeight="700" textAnchor="end" fill="var(--htr-bg)" style={{ fontFamily: "var(--htr-font-mono)", fontVariantNumeric: "tabular-nums" }}>{fmtPrice(cursorPrice, 0)}</text>
            <g transform={`translate(${boxX}, ${boxY})`}>
              <rect width={boxW} height={boxH} fill="var(--htr-surface)" stroke="var(--htr-line)" strokeWidth="0.8" rx="4" opacity="0.97" />
              <text x={7} y={14} fontSize="10" fontWeight="700" fill="var(--htr-ink)" style={{ fontFamily: "var(--htr-font-mono)" }}>{bar.date || `D-${data.length - 1 - hIdx}`}</text>
              <text x={7} y={29} fontSize="9.5" fill="var(--htr-ink-2)">开 {fmtPrice(bar.open, 0)}</text>
              <text x={7} y={43} fontSize="9.5" fill="var(--htr-bull)">高 {fmtPrice(bar.high, 0)}</text>
              <text x={7} y={57} fontSize="9.5" fill="var(--htr-bear)">低 {fmtPrice(bar.low, 0)}</text>
              <text x={7} y={71} fontSize="9.5" fontWeight="700" fill="var(--htr-ink)">收 {fmtPrice(bar.close, 0)}</text>
              <text x={7} y={85} fontSize="9.5" fill="var(--htr-ink-2)">量 {volStr}</text>
              {chgPct !== null && <text x={7} y={101} fontSize="9.5" fontWeight="700" fill={chgColor} style={{ fontFamily: "var(--htr-font-mono)" }}>涨跌 {chgPct >= 0 ? '+' : ''}{chgPct.toFixed(2)}%</text>}
            </g>
          </g>
        );
      })()}
    </svg>
  );
}

// ── NewsTimeline ──
function NewsTimeline({ items, max = 10, compact = false }) {
  const rows = items.slice(0, max);
  return (
    <div>
      {rows.map((n, i) => {
        const dot = n.weight === "high" ? "var(--htr-bear)" : n.weight === "medium" ? "var(--htr-warn)" : "var(--htr-ink-3)";
        return (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "62px 10px 1fr", gap: 10, padding: compact ? "8px 0" : "11px 0", borderBottom: i < rows.length - 1 ? "1px solid var(--htr-line-soft)" : "none", alignItems: "start" }}>
            <div className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-3)", paddingTop: 2 }}>{n.ts}</div>
            <div style={{ position: "relative", height: "100%", paddingTop: 5 }}>
              <span style={{ position: "absolute", left: 0, top: 5, width: 9, height: 9, borderRadius: "50%", background: dot, boxShadow: "0 0 0 2px var(--htr-surface)" }} />
              {i < rows.length - 1 && <span style={{ position: "absolute", left: 4, top: 15, bottom: -15, width: 1.5, background: "var(--htr-line)" }} />}
            </div>
            <div>
              <div style={{ fontSize: 12.5, color: "var(--htr-ink)", lineHeight: 1.45 }}>{n.text}</div>
              <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)", marginTop: 4, display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                <span>{n.src}</span><span>·</span>
                {(n.linkedSymbols || []).slice(0, 3).map((s) => <span key={s} className="htr-chip" style={{ padding: "0px 6px", fontSize: 9.5 }}>{s}</span>)}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── GateFlow ──
function GateFlow({ gates, compact = false }) {
  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: 0, width: "100%", flexWrap: "wrap" }}>
      {gates.map((g, i) => {
        const isDone = g.status === "done", isCur = g.status === "in_progress", isBlk = g.status === "blocked";
        const bg = isDone ? "var(--htr-bull-bg)" : isCur ? "var(--htr-warn-bg)" : isBlk ? "var(--htr-bear-bg)" : "var(--htr-surface-2)";
        const fg = isDone ? "var(--htr-bull)" : isCur ? "var(--htr-warn)" : isBlk ? "var(--htr-bear)" : "var(--htr-ink-3)";
        return (
          <div key={g.id} style={{ flex: "1 1 110px", padding: compact ? "7px 9px" : "10px 12px", background: bg, borderTop: "1px solid var(--htr-line)", borderBottom: "1px solid var(--htr-line)", borderLeft: i === 0 ? "1px solid var(--htr-line)" : "none", borderRight: "1px solid var(--htr-line)", position: "relative", clipPath: i < gates.length - 1 ? "polygon(0 0, calc(100% - 8px) 0, 100% 50%, calc(100% - 8px) 100%, 0 100%, 8px 50%)" : "polygon(0 0, 100% 0, 100% 100%, 0 100%, 8px 50%)", marginLeft: i === 0 ? 0 : -8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 15, height: 15, borderRadius: "50%", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, background: fg, color: "var(--htr-surface)" }}>{g.glyph}</span>
              <span style={{ fontSize: 10.5, fontWeight: 700, color: fg }}>{g.id}</span>
            </div>
            <div style={{ fontSize: compact ? 11 : 12.5, fontWeight: 600, color: "var(--htr-ink)", marginTop: 3 }}>{g.label}</div>
            {!compact && <div style={{ fontSize: 10, color: "var(--htr-ink-3)", marginTop: 2 }}>{g.next}</div>}
          </div>
        );
      })}
    </div>
  );
}

// ── ThemeHeatBars ──
function ThemeHeatBars({ themes }) {
  return (
    <div>
      {themes.map((t, i) => (
        <div key={t.id} style={{ display: "grid", gridTemplateColumns: "96px 1fr 44px 52px", gap: 10, alignItems: "center", padding: "8px 0", borderBottom: i < themes.length - 1 ? "1px solid var(--htr-line-soft)" : "none" }}>
          <div style={{ fontSize: 12.5, fontWeight: 600 }}>{t.label}</div>
          <div style={{ height: 7, background: "var(--htr-line-soft)", borderRadius: 4, overflow: "hidden" }}>
            <div style={{ width: `${t.heat}%`, height: "100%", background: heatColor(t.heat) }} />
          </div>
          <div className="htr-num" style={{ fontSize: 12.5, fontWeight: 700, textAlign: "right", color: heatColor(t.heat) }}>{t.heat}°</div>
          <div className={"htr-num " + changeClass(t.mom)} style={{ fontSize: 11.5, textAlign: "right" }}>{arrow(t.mom)} {Math.abs(t.mom).toFixed(1)}</div>
        </div>
      ))}
    </div>
  );
}

// ── DecisionLog (Rule 9.6: explicit empty state) ──
function DecisionLog({ entries, max = 8 }) {
  const items = entries.slice(0, max);
  if (items.length === 0) {
    return (
      <div style={{ padding: "20px 16px", textAlign: "center", border: "1px dashed var(--htr-line-2)", borderRadius: 8, margin: "10px 14px", background: "var(--htr-surface-2)" }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--htr-ink-2)" }}>今日尚无决策日志</div>
        <div style={{ fontSize: 11, color: "var(--htr-ink-3)", marginTop: 4, lineHeight: 1.5 }}>扫描运行后，每条候选与价位会按 §8.6 落盘到此处。<br/>空日志不代表校准或纸面交易已就绪。</div>
      </div>
    );
  }
  return (
    <div>
      {items.map((e, i) => (
        <div key={i} style={{ display: "grid", gridTemplateColumns: "92px 56px 1fr 38px", gap: 8, alignItems: "center", padding: "8px 14px", fontSize: 12, borderBottom: "1px solid var(--htr-line-soft)" }}>
          <span className="htr-mono" style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>{e.ts}</span>
          <span className="htr-mono" style={{ fontSize: 11.5, fontWeight: 600 }}>{e.symbol}</span>
          <span title={(e.action || "") + (e.note ? " · " + e.note : "")} style={{ color: "var(--htr-ink-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            <code style={{ background: "var(--htr-surface-3)", padding: "1px 6px", borderRadius: 4, fontSize: 10, marginRight: 6, color: "var(--htr-ink-2)" }}>{e.action}</code>{e.note}
          </span>
          <span className="htr-num" style={{ fontSize: 11.5, fontWeight: 700, textAlign: "right", color: e.score == null ? "var(--htr-ink-4)" : "var(--htr-ink)" }}>{e.score == null ? "—" : e.score}</span>
        </div>
      ))}
    </div>
  );
}

// ── live-feel ticking price ──
// Rule 11.9 — display data honesty. The research panel MUST NOT simulate price
// movement. This previously ran a setInterval random walk (±0.07%/3.4s) on the
// real close, which fabricated intra-second "live" motion (and red/green pulse)
// even on a closed market / non-trading day — a PIT (Rule 8.2) + anti-FOMO
// (§12) violation the user caught on a Saturday. It now returns the real backend
// close UNCHANGED. Kept as a hook so call sites are unaffected; a genuine
// real-time quote feed may only re-introduce motion when gated on a true
// market-open signal (see marketFreshness / market[].state once calendar-correct).
function useTickingPrice(base) {
  return base;
}

// Rule 11.9 — derive an honest freshness / session label from the real backend
// signals (meta.asof = wall-clock now, meta.tradeDate = last trading day). When
// the data's trade date is not today (weekend / holiday / pre-open), say so
// instead of implying a live quote.
function marketFreshness(meta) {
  if (!meta || !meta.asof || !meta.tradeDate) return null;
  const asofDate = String(meta.asof).slice(0, 10);
  const tradeDate = String(meta.tradeDate);
  let dow = null;
  // Weekday of the JST calendar date, independent of the browser's timezone:
  // anchor at JST noon (= 03:00Z same day) and read getUTCDay (code-review fix).
  try { dow = new Date(asofDate + "T12:00:00+09:00").getUTCDay(); } catch (e) {}
  const isWeekend = dow === 0 || dow === 6;
  const sameDay = asofDate === tradeDate;
  const closed = !sameDay || isWeekend;
  return {
    closed, sameDay, isWeekend, tradeDate, asofDate,
    label: closed
      ? `${isWeekend ? "周末休市" : "非交易时段"} · 数据为 ${tradeDate} 收盘`
      : `数据为 ${tradeDate}`,
  };
}

// ── element size observer ──
function useElementSize() {
  const ref = React.useRef(null);
  const [size, setSize] = React.useState({ width: 0, height: 0 });
  React.useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setSize((s) => (s.width === Math.round(r.width) && s.height === Math.round(r.height)) ? s : { width: Math.round(r.width), height: Math.round(r.height) });
    };
    // immediate measure (ResizeObserver alone can be unreliable for the first paint)
    measure();
    const raf = requestAnimationFrame(measure);
    let ro;
    if (typeof ResizeObserver !== "undefined") { ro = new ResizeObserver(measure); ro.observe(el); }
    window.addEventListener("resize", measure);
    return () => { cancelAnimationFrame(raf); if (ro) ro.disconnect(); window.removeEventListener("resize", measure); };
  }, []);
  return [ref, size];
}

// ── watchlist (Rule 14.9 user-state) — server is source of truth, localStorage
// is an optimistic cache + offline fallback (Rule 11.10: add/remove issue real
// POSTs to the whitelisted /api/watchlist/{add,remove}, not a local-only stub).
// MODULE-LEVEL shared store so every useWatchlist instance (nav chip + leader ☆)
// stays in sync — a mutation from one re-renders all (Codex-2026-06-06 #5). ──
const HTR_WL = (() => {
  const KEY = "htr_watchlist_v2";
  const read = () => { try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { return []; } };
  let entries = read();
  const listeners = new Set();
  return {
    get: () => entries,
    setLocal: (next) => { entries = next; try { localStorage.setItem(KEY, JSON.stringify(next)); } catch (e) {} listeners.forEach((l) => l()); },
    subscribe: (l) => { listeners.add(l); return () => listeners.delete(l); },
  };
})();
let _wlHydrated = false;
function useWatchlist() {
  const [, force] = React.useReducer((x) => x + 1, 0);
  React.useEffect(() => HTR_WL.subscribe(force), []);
  const entries = HTR_WL.get();
  const fromPayload = (d) => (d && Array.isArray(d.entries)) ? d.entries.map((e) => ({ symbol: e.symbol, added_ts: e.added_ts })) : null;
  const post = (action, symbol) =>
    fetch(`/api/watchlist/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symbol }) })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { const next = fromPayload(d); if (next) HTR_WL.setLocal(next); })   // reconcile to server (notifies all instances)
      .catch(() => {});                                                                  // offline → optimistic local already applied
  // hydrate from the server ONCE across all instances (source of truth); keep localStorage on miss
  React.useEffect(() => {
    if (_wlHydrated) return; _wlHydrated = true;
    fetch("/api/watchlist", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { const next = fromPayload(d); if (next) HTR_WL.setLocal(next); })
      .catch(() => { _wlHydrated = false; });
  }, []);
  const add = (symbol) => { if (entries.some((e) => e.symbol === symbol)) return; HTR_WL.setLocal([...entries, { symbol, added_ts: new Date().toISOString() }]); post("add", symbol); };
  const remove = (symbol) => { HTR_WL.setLocal(entries.filter((e) => e.symbol !== symbol)); post("remove", symbol); };
  const has = (symbol) => entries.some((e) => e.symbol === symbol);
  return { entries, size: entries.length, add, remove, has, updated_ts: "" };
}

// ── Rule 11.9 — honest async state for per-symbol overlay bodies ──
// Driven by the useEnrichedCandidate `_status` map (pending/ok/failed) so a
// failed fetch never leaves the offline mock masquerading as real. `pending`
// renders a loading skeleton; `failed` renders a visible 示例占位 banner above
// the (crash-safe mock) body; `ok`/undefined renders nothing (real content shows).
function AsyncBodyState({ status }) {
  if (status === "pending") {
    return <div style={{ fontSize: 12, color: "var(--htr-ink-3)", padding: "8px 0", display: "flex", alignItems: "center", gap: 7 }}><span>⏳</span> 生成中…</div>;
  }
  if (status === "failed") {
    return <div style={{ fontSize: 11, color: "var(--htr-warn)", background: "var(--htr-warn-bg)", border: "1px solid var(--htr-warn)", borderRadius: 5, padding: "6px 9px", marginBottom: 8 }}>⚠ 真实数据未就绪（后端未返回）· 以下为示例占位 (Rule 11.9.4)</div>;
  }
  return null;
}

// ── glossary (Rule 9.4 — definitions never equate score to win rate) ──
const GLOSSARY = {
  "七档阶梯": "7 个建议价位 = 3 个买入档（激进/均衡/保守）+ 止损 + 3 个卖出档（卖1/卖2/延伸）。从 ATR 或日内波动算出，仅作研究参考，非订单。",
  "未校准研究分": "0-100 的研究分数。还没经过 ≥100 次「预测 → 实际结果」配对校准，所以不能当胜率读。",
  "市场温度": "把多个市场（日股/美股/SOX/USDJPY 等）的近期表现合成 0-100 温度计，>70 过热 / 50-69 升温 / 30-49 平温 / <30 冷。",
  "主题热力": "把市场分为若干主题，按强度排序。热度越高代表该主题当下越被资金推动。",
  "新闻催化": "最近 N 小时新闻摘要，按重要程度（urgency + sentiment）标 high/medium/low。",
  "决策日志": "§8.6 强制：每次扫描的候选、价位、模型版本都按日落盘到 reports/predictions/。",
  "策略卡片": "把 ladder + 风险 + 执行建议合成为一张研究参考卡。永远 advice-only，由用户在外部券商手动执行。",
  "K-fold 降级": "研究分经过样本外（OOS）K-fold 验证，若 OOS Brier ≥ 随机基线 0.25 则自动降级，不暴露校准概率。",
  "Daily Cockpit": "Rule 12.0 stage 0 pull-only 每日驾驶舱：仅在你主动打开时展示持仓 / 关注 / 延迟报价 / 公告，不推送、不下单。",
  "因子构成": "研究分背后的多个 alpha 因子（动量 / Sharpe / 量能 / 流动性等），均为排序信号，不是概率。",
  "AI 综合": "本地 LLM 把新闻 + 因子 + 技术面合成中文研究叙事。Rule 8.3.1 守门：叙事中不得出现胜率 / 概率 / 数字%。",
  "§8.6": "治理规则 8.6 — 每条预测都必须落盘到 decision log，含 prediction_id / model_version / decision_cutoff 等可重现字段。",
  "§10": "治理规则 10 — 自动化分 8 阶门槛，按顺序解锁：候选 → 阶梯 → 决策日志 → 反馈 → 校准 → 提醒 → 纸面 → 实盘。",
};
function Term({ children, k }) {
  const key = k || (typeof children === "string" ? children : "");
  const def = GLOSSARY[key];
  const ref = React.useRef(null);
  const tipId = React.useId();
  const [pos, setPos] = React.useState(null);
  if (!def) return children;
  const W = 280;
  const show = () => {
    const el = ref.current; if (!el) return;
    const r = el.getBoundingClientRect();
    let left = Math.min(r.left, window.innerWidth - 8 - W);
    if (left < 8) left = 8;
    const placeAbove = r.bottom > window.innerHeight - 130;
    setPos({ left, top: placeAbove ? null : Math.round(r.bottom + 6), bottom: placeAbove ? Math.round(window.innerHeight - r.top + 6) : null });
  };
  const hide = () => setPos(null);
  return (
    <span ref={ref} className="htr-term" tabIndex={0} aria-describedby={pos ? tipId : undefined} onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide} onKeyDown={(e) => { if (e.key === "Escape") hide(); }}>
      {children}
      {pos && ReactDOM.createPortal(
        <span id={tipId} role="tooltip" style={{
          position: "fixed", left: pos.left, top: pos.top ?? undefined, bottom: pos.bottom ?? undefined,
          width: W, padding: "9px 12px 10px", background: "var(--htr-ink)", color: "var(--htr-bg)",
          borderRadius: 6, boxShadow: "0 8px 30px rgba(0,0,0,.32)", fontSize: 11.5, lineHeight: 1.5,
          zIndex: 2147483600, pointerEvents: "none", fontFamily: "var(--htr-font-body)",
        }}>
          <span style={{ display: "block", fontFamily: "var(--htr-font-mono)", fontSize: 10.5, fontWeight: 700, color: "var(--htr-warn)", marginBottom: 3 }}>{key}</span>
          <span style={{ display: "block" }}>{def}</span>
        </span>, document.body)}
    </span>
  );
}

// ── CatalystBadges (ADR-0009 P15 / Rule 11.12) ──
// Surfaces the news-catalyst ordering + theme-leader designation the hybrid rerank
// (api/serializers) attaches to each candidate. Honest by construction: these
// explain the ORDER (today's news theme-heat), not a win-rate; the displayed score
// stays the raw uncalibrated research score (排序≠分数). Derived ONLY from the
// served candidate fields (PIT) — renders nothing when the candidate is not
// news-catalyzed, and never asserts a probability/win-rate.
const CATALYST_BADGE_NOTE = "🔥催化 / 👑龙头反映今日新闻主题热度对候选「排序」的影响，是排序信号、非胜率/概率；显示分仍为未校准研究分（排序≠分数）。";
function CatalystBadges({ c, size }) {
  // P20 (Rule 11.15): also render when the external SKHY ADR catalyst is active,
  // even if the name is not news-catalyzed.
  if (!c || (!c.newsCatalyzed && !c.skhyCatalystActive)) return null;
  const fs = size === "sm" ? 9.5 : undefined;
  return (
    <>
      {c.isThemeLeader && (
        <span className="htr-chip" title={CATALYST_BADGE_NOTE}
              style={{ fontSize: fs, cursor: "help", color: "var(--htr-warn)", borderColor: "color-mix(in oklab,var(--htr-warn) 34%,var(--htr-line-2))", background: "var(--htr-warn-bg)" }}>
          👑 龙头
        </span>
      )}
      {c.newsCatalyzed && (
        <span className="htr-chip" title={CATALYST_BADGE_NOTE}
              style={{ fontSize: fs, cursor: "help", color: "var(--htr-bear)", borderColor: "color-mix(in oklab,var(--htr-bear) 28%,var(--htr-line-2))" }}>
          🔥 新闻催化
        </span>
      )}
      {c.skhyCatalystActive && (
        // P20 / Rule 11.15 — read-only EXTERNAL SK hynix ADR (SKHY) catalyst indicator.
        // Neutral research wording only: explains why a JP semi name MAY react; NOT buy
        // advice; no probability / win-rate / expected-return / edge. Renders nothing
        // when SKHY is off / stale / pending / unavailable. Does NOT reorder candidates.
        <span className="htr-chip"
              title={`SK hynix ADR (SKHY) 外部催化观察 · 状态 ${c.skhyCatalystStatus || "off"} · 依据 ${(c.semiSympathyReasons || []).length} 项 · 相对强弱 ${c.relativeStrengthVsSkhy ?? "—"} · 隔夜 ${c.skhyOvernightMove ?? "—"}。解释该日本半导体名为何可能联动外部走势,属研究用排序提示,非买入建议,不预测收益。SKHY 待上市/过期/不可用时不显示。`}
              style={{ fontSize: fs, cursor: "help", color: "var(--htr-accent)", borderColor: "color-mix(in oklab,var(--htr-accent) 30%,var(--htr-line-2))" }}>
          🌐 SKHY联动·研究
        </span>
      )}
    </>
  );
}

// ── ActionBoardCard (P21-05 / Rule 11.16) ──
// Read-only Daily Action Board (交易计划板): TRADE PLANS the backend assembles
// from already-served fields (blended ordering / 7-tier ladder / chase-risk /
// Rule 11.8 sizing arithmetic). Plans are structure references, NOT
// predictions — the standing no-edge disclosure is non-collapsible
// (Rule 11.16.1). Fails open to nothing when the backend omits the board
// (offline mock has none). Rows inherit the served order (Rule 11.16.2).
const ACTION_BOARD_STATUS = {
  plan_ready: { label: "结构就绪", color: "var(--htr-accent)" },
  watch_only: { label: "仅观察 · 不追", color: "var(--htr-warn)" },
  s_kabu_only: { label: "仅 S株 可行", color: "var(--htr-ink-2)" },
  not_tradable: { label: "超出账户承受", color: "var(--htr-bear)" },
  insufficient_data: { label: "数据不足", color: "var(--htr-ink-3)" },
};

function ActionBoardCard() {
  const board = (window.HTR_DATA && window.HTR_DATA.actionBoard) || null;
  if (!board || !Array.isArray(board.rows) || !board.rows.length) return null;
  const yen = (x) => (x == null ? "—" : "¥" + Number(x).toLocaleString("ja-JP"));
  const tierTxt = (t) => (t ? `${yen(t.price)} (${t.pct > 0 ? "+" : ""}${t.pct}%)` : "—");
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <div style={{ padding: "11px 14px 0" }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: "var(--htr-ink-1)" }}>
          交易计划板 <span style={{ fontSize: 10, fontWeight: 400, color: "var(--htr-ink-3)" }}>Rule 11.16 · 只读 · 计划≠预测</span>
        </div>
        {/* Rule 11.16.1 — standing disclosure, non-collapsible */}
        <div style={{ fontSize: 10.5, lineHeight: 1.5, color: "var(--htr-ink-3)", margin: "4px 0 9px" }}>
          {board.disclosure}
        </div>
      </div>
      <div style={{ padding: "0 14px 10px", display: "flex", flexDirection: "column", gap: 8 }}>
        {board.rows.map((r) => {
          const st = ACTION_BOARD_STATUS[r.planStatus] || ACTION_BOARD_STATUS.insufficient_data;
          const entry = r.entryPlan || {};
          return (
            <div key={r.symbol} style={{ border: "1px solid var(--htr-line)", borderRadius: 7, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 5 }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
                <span style={{ fontSize: 12.5, fontWeight: 700, color: "var(--htr-ink-1)" }}>{r.symbol}</span>
                <span style={{ fontSize: 11, color: "var(--htr-ink-2)" }}>{r.nameJa}</span>
                <span className="htr-chip" style={{ fontSize: 10, color: st.color, borderColor: "var(--htr-line-2)" }}>{st.label}</span>
                <span style={{ fontSize: 10, color: "var(--htr-ink-3)", marginLeft: "auto" }}>条件 {r.conditionsMet}</span>
              </div>
              {r.entryPlan && (
                <div style={{ fontSize: 10.5, color: "var(--htr-ink-2)", display: "flex", gap: 12, flexWrap: "wrap" }}>
                  <span>建议价位·均衡 <b>{tierTxt(entry.balanced)}</b></span>
                  <span>激进 {tierTxt(entry.aggressive)} / 保守 {tierTxt(entry.conservative)}</span>
                  <span>止损参考 <b style={{ color: "var(--htr-bear)" }}>{tierTxt(r.stopRef)}</b></span>
                  <span>目标参考 {(r.exitRefs || []).map(tierTxt).join(" · ")}</span>
                </div>
              )}
              <div style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>
                {r.sizing
                  ? `仓位测算(Rule 11.8 算术,非指令) · 风险预算 ${(r.sizing.riskBudgetFrac * 100).toFixed(0)}% NAV=${yen(r.sizing.riskBudgetJpy)} · 整手 ${r.sizing.wholeLotShares}株${r.sizing.wholeLotShares ? ` (${yen(r.sizing.notionalWholeLotJpy)})` : ""} · S株 ${r.sizing.sKabuShares}株`
                  : "仓位测算不可用（无持仓 NAV，不虚构）"}
              </div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {(r.timingChecklist || []).map((c) => (
                  <span key={c.key} className="htr-chip" style={{ fontSize: 9.5, color: c.ok ? "var(--htr-ink-2)" : "var(--htr-warn)", borderColor: "var(--htr-line-2)" }}>
                    {c.ok ? "✓" : "✗"} {c.label}: {String(c.value)}
                  </span>
                ))}
              </div>
              {r.why && <div style={{ fontSize: 10, color: "var(--htr-ink-3)" }}>{r.why}</div>}
            </div>
          );
        })}
      </div>
      <div style={{ padding: "0 14px 11px", fontSize: 9.5, color: "var(--htr-ink-3)" }}>
        参数：风险预算 {(board.params.riskBudgetFrac * 100).toFixed(0)}% NAV · 集中度上限 {(board.params.concentrationCapFrac * 100).toFixed(0)}% NAV · 整手 {board.params.lotSize}株 · 市场时段 {board.marketSession || "UNKNOWN"} · 分数状态 {board.scoreStatus}
      </div>
    </div>
  );
}

// ── ExitBoardCard (P22-01 / Rule 11.17) ──
// Position Exit Discipline Board (持仓纪律板): pure arithmetic between the
// journal-derived holdings and the OPERATOR'S OWN declared take-profit/stop
// parameters (00_DESIGN §1 "2-5% 止盈换仓"). Predicts nothing; never renders a
// sell directive; a breached stop reference is surfaced prominently
// (Rule 11.17.2). Fails open to nothing when the backend omits the board.
const EXIT_BOARD_STATUS = {
  within_plan: { label: "计划内持有", color: "var(--htr-ink-2)" },
  past_first_take_profit: { label: "已越首止盈参考", color: "var(--htr-bull)" },
  stop_reference_breached: { label: "止损参考已破", color: "var(--htr-bear)" },
  insufficient_data: { label: "数据不足", color: "var(--htr-ink-3)" },
};

function ExitBoardCard() {
  const board = (window.HTR_DATA && window.HTR_DATA.exitBoard) || null;
  if (!board || !Array.isArray(board.rows) || !board.rows.length) return null;
  const yen = (x) => (x == null ? "—" : "¥" + Number(x).toLocaleString("ja-JP"));
  const refTxt = (t) => (t ? `${yen(t.price)} (距 ${t.distancePct > 0 ? "+" : ""}${t.distancePct}%)` : "—");
  return (
    <div className="htr-card" style={{ overflow: "hidden" }}>
      <div style={{ padding: "11px 14px 0" }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: "var(--htr-ink-1)" }}>
          持仓纪律板 <span style={{ fontSize: 10, fontWeight: 400, color: "var(--htr-ink-3)" }}>Rule 11.17 · 只读 · 你的参数,不是预测</span>
        </div>
        {/* Rule 11.17 — standing disclosure, non-collapsible */}
        <div style={{ fontSize: 10.5, lineHeight: 1.5, color: "var(--htr-ink-3)", margin: "4px 0 9px" }}>
          {board.disclosure}
        </div>
      </div>
      <div style={{ padding: "0 14px 10px", display: "flex", flexDirection: "column", gap: 8 }}>
        {board.rows.map((r) => {
          const st = EXIT_BOARD_STATUS[r.exitStatus] || EXIT_BOARD_STATUS.insufficient_data;
          const pnlColor = r.unrealizedReturnPct > 0 ? "var(--htr-bull)" : r.unrealizedReturnPct < 0 ? "var(--htr-bear)" : "var(--htr-ink-2)";
          return (
            <div key={r.symbol} style={{ border: "1px solid var(--htr-line)", borderRadius: 7, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 5 }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
                <span style={{ fontSize: 12.5, fontWeight: 700, color: "var(--htr-ink-1)" }}>{r.symbol}</span>
                <span style={{ fontSize: 10.5, color: "var(--htr-ink-3)" }}>{r.qty}株 · 成本 {yen(r.avgCost)}</span>
                <span style={{ fontSize: 11, color: pnlColor }}>
                  现价 {yen(r.marketPrice)}{r.unrealizedReturnPct != null ? ` (${r.unrealizedReturnPct > 0 ? "+" : ""}${r.unrealizedReturnPct}%)` : ""}
                </span>
                <span className="htr-chip" style={{ fontSize: 10, color: st.color, borderColor: "var(--htr-line-2)", marginLeft: "auto" }}>{st.label}</span>
              </div>
              {r.stopRef && (
                <div style={{ fontSize: 10.5, color: "var(--htr-ink-2)", display: "flex", gap: 12, flexWrap: "wrap" }}>
                  <span>止损参考 <b style={{ color: "var(--htr-bear)" }}>{refTxt(r.stopRef)}</b></span>
                  <span>止盈参考 {(r.takeProfitRefs || []).map(refTxt).join(" · ")}</span>
                </div>
              )}
              {r.statusNote && <div style={{ fontSize: 10, color: st.color }}>{r.statusNote}</div>}
            </div>
          );
        })}
      </div>
      <div style={{ padding: "0 14px 11px", fontSize: 9.5, color: "var(--htr-ink-3)" }}>
        纪律参数（你声明的,Rule 4 显式配置）：止盈参考 {(board.params.takeProfitFracs || []).map((f) => `+${(f * 100).toFixed(0)}%`).join("/")} · 止损参考 {(board.params.stopFrac * 100).toFixed(0)}% · 基准 成本价
        {board.actionBoardPlanReady != null && ` · 今日计划板 结构就绪+仅S株可行 共 ${board.actionBoardPlanReady} 个（事实计数,非换仓指令）`}
      </div>
    </div>
  );
}

Object.assign(window, {
  AnimatedPrice, Sparkline, KLineChart, NewsTimeline, GateFlow, ThemeHeatBars, DecisionLog, CatalystBadges,
  ActionBoardCard, ExitBoardCard,
  antiCollideLabels, ladderColor, useTickingPrice, marketFreshness, useElementSize, useWatchlist, Term, GLOSSARY,
  HTR: { fmtPrice, fmtPct, changeClass, arrow, heatColor, heatBg, LABELS: HTR_LABELS },
});
