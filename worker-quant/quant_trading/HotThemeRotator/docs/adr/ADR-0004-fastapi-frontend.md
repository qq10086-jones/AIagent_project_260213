# ADR-0004: FastAPI + Vite/React Frontend Adoption

## Status

Accepted 2026-05-24. **Amended 2026-05-24 (Phase 1 = zero-build)** — see Decision §2.

## Context

The user supplied a high-fidelity UI prototype (`quant.zip`, 4 layout variants V1-V4, shared design tokens, mock data fixture) labeling V3 "市场温度仪表盘" as the preferred direction. The prototype's `Rationale` section, authored by the same user, makes the architectural recommendation explicit:

> "建议把当前 Python 数据层保留, 上面替换成 FastAPI + 静态前端 (Vite + React)。Streamlit 适合 demo 不适合产品。"

The current Streamlit implementation (P8-01 through P8-08) reaches the limits of `st.components.html` for the visual surface V3 wants:

- 6-market temperature mosaic with per-cell sparklines, state badges, color-coded thresholds
- Theme heat treemap (variable-size bubble layout) with leader chips
- SVG vertical price ladder with current-price horizontal highlight
- News timeline with weight-coded dots + linked-symbol chips
- Bottom §10 gate flow with directional arrows
- Live ticking price (mock today; polling or websocket later)
- Multi-font stack (IBM Plex Sans/Mono/Serif + Noto Sans JP) with tabular numerics
- Light/dark theme via CSS variable swap

Reproducing each of these as Streamlit `markdown(unsafe_allow_html=True)` blocks would either degrade fidelity (no useTickingPrice equivalent, no SVG chart libraries, no smooth dark-mode toggle) or accumulate brittle HTML/CSS string concatenation that fails the §3 readability bar of `_inject_style`.

## Decision

Adopt a two-process architecture for the user-facing surface, keeping the Python data layer authoritative:

1. **`api/`** — new FastAPI application exposing **read-only** JSON endpoints over the existing `decision_log/` + `calibration/` + opportunity scanner output. The single primary endpoint `/api/dashboard` returns the full V3 data shape; further endpoints (`/api/predictions/{date}`, `/api/calibration?source=...`) follow as needs surface.
2. **`frontend/`** — new React 18 single-page app.
   - **Phase 1 (this iteration, ADOPTED 2026-05-24)**: zero-build. Source files (`shared.jsx`, `v3.jsx`, `data.js`, `index.html`) copied verbatim from `quant.zip`. React + Babel-standalone load via CDN inside `index.html`; JSX is compiled in-browser. The user-authored Rationale recommended Vite, but for a single-user local research tool the zero-build path renders V3 faithfully without npm dependencies. FastAPI mounts `frontend/` directly as static at `/`. First boot (~1.5 MB Babel-standalone CDN load) is acceptable since the dashboard is opened occasionally, not by anonymous web traffic.
   - **Phase 2 (deferred, future P8-10 if needed)**: migrate to Vite + ES modules when (a) page load latency matters, or (b) we want HMR for fast iteration, or (c) we add additional dashboards that share components. Migration cost is one-time refactor of `shared.jsx` + `v3.jsx` to named exports.

   Data flow: `data.js` provides mock baseline (kept as fallback for sections the Python layer hasn't surfaced yet); `index.html` boot script `fetch("/api/dashboard")`s, merges API data over mock, then mounts `<V3MarketDashboard />`.

The existing Streamlit app at `tools/streamlit_opportunity_app.py` is **kept as fallback** during the transition. It continues to serve the simple sample/yfinance flow. The new FastAPI+React surface becomes the recommended way to run the dashboard.

The Python data layer (`src/hot_theme_rotator/**`) is the **single source of truth** for: §10 gate state (from `_GATE_DEFINITIONS`), calibration status (from `build_calibration_report`), predictions (from `read_predictions`), and decision log (from `decision_log/jsonl_writer`). The serializer in `api/serializers.py` translates Python objects into the V3 JSON shape but does NOT compute scores, ground truth, or calibrations — those stay in the existing modules.

## Consequences

Positive:

- Visual fidelity to V3 design (user's preferred direction) without compromise.
- Frontend testability via vitest (when needed); backend via FastAPI TestClient.
- API layer becomes the reusable contract — future native iOS / Android / second dashboard can consume the same JSON.
- Streamlit fallback remains for users who want the simple flow.
- React + Vite is the standard modern frontend stack; team / future contributors will find it familiar.

Negative:

- Two new runtime dependencies (Node 22 + npm) and one new Python dependency family (FastAPI + uvicorn).
- Two processes to manage in dev (`uvicorn` on 8000 + `vite` on 5173) until prod build is wired.
- CORS configuration needed in dev (single-origin in prod via FastAPI static mount).
- `frontend/node_modules/` will be large (50-100 MB); must be `.gitignore`d.

Risks and mitigations:

- **Risk**: frontend label drift — a hand-edit in `v3.jsx` could relabel "未校准研究分" as "胜率", violating §9.4. **Mitigation**: the calibration text is rendered from the JSON `meta.calibration.text` field whose value comes from `build_calibration_badge` (already enforces §9.4); v3.jsx renders without rewording.
- **Risk**: an execute-button affordance accidentally lands in v3.jsx during future iteration, violating Rule 3. **Mitigation**: a startup smoke test in `tests/integration/test_frontend_advice_only.py` greps the built `dist/assets/*.js` for forbidden tokens (`onSubmit`, `place_order`, `execute`).
- **Risk**: V3 mock data shape includes fields current Python layer cannot produce (markets[] 6 entries, themes[] with heat, kline 40 bars, candidate.nameJa/nameCn, real `sample/brier`). **Mitigation**: serializer fills missing-data placeholders with explicit `"insufficient_data"` markers; frontend renders missing cells with visible "数据未就绪" tags rather than fabricated values.
- **Risk**: §10 gate labels in V3 mock don't match our `_GATE_DEFINITIONS`. **Mitigation**: serializer ALWAYS sources gate labels from Python; the mock `gates[]` array is discarded.

## Alternatives Considered

- **Streamlit `components.html` iframe of V3 build** — Less fidelity loss than rebuilding, but bidirectional state (sidebar mode toggle) becomes awkward; rejected.
- **Hand-port V3 to Streamlit `_render_*` functions** — Documented as "保守增量" option C in the P8-09 selection; rejected by user.
- **Keep Streamlit-only and not adopt V3** — Rejected: user explicitly chose to merge.

## Out of Scope

- Authentication / login — local single-user.
- WebSocket / SSE — polling is sufficient at current data frequencies.
- Real-time tick price source — V3's `useTickingPrice` stays as visual jitter only; not a real market data feed.
- K-line OHLC real-data integration — V3 `generateKline()` mock kept until `LegacyDailyPriceFetcher` (P9-02 follow-up) lands.
- Frontend test framework setup beyond a smoke build — vitest can be added later.
- Sunsetting Streamlit — kept as fallback.

## References

- `docs/02_GOVERNANCE.md` Rule 3 / Rule 4 / §3 / §9.4 / §10
- `docs/adr/ADR-0003-decision-log.md` (Python data layer staying authoritative)
- `quant.zip` (user-supplied V3 prototype, extracted to `.runtime/ui_inspection/`)
- `src/hot_theme_rotator/ui/opportunity_dashboard.py` `_GATE_DEFINITIONS` (gate label truth)
- `src/hot_theme_rotator/calibration/` (P9-03 calibration source for the badge)
