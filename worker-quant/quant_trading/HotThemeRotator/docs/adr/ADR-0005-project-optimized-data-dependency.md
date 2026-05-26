# ADR-0005: Project_optimized as Upstream Live-Data Producer

## Status

Accepted 2026-05-24. **Pending supersession by ADR-0008** on cutover day T (T to be confirmed by user; see ADR-0008).

After cutover, the read-only consumption pattern documented here becomes historical: this ADR describes the data flow in force from 2026-05-24 to T. Post-T, HotThemeRotator owns its portfolio journal directly (Section 14 of `02_GOVERNANCE.md`) and stops reading `Project_optimized/japan_market.db` at runtime.

## Context

User explicit ask: "我最重要看到的系统都是真实数据以及匹配我的持仓数据的系统，重要的是，我要能用这个系统".

The HotThemeRotator dashboard (V1-V4 from P8-09) currently uses `data.js` mock for 4 sections (markets / themes / newsTimeline / kline) and `build_sample_panel` for candidates, with no positions surface at all. To make the system usable for daily operations, every data section must be backed by real data, and current portfolio holdings must be visible alongside candidates.

The sibling project `worker-quant/quant_trading/Project_optimized/` is the user's live-trading system. It has:

- 6+ months of curated `japan_market.db` (SQLite): 30+ tables including `daily_prices`, `intraday_quotes`, `news_feed` / `news_items` / `news_sentiment`, `cross_asset_snapshots` (multi-market temperature inputs), `factor_signals` (theme heat inputs), `positions` / `fills` / `orders`, `decision_journal`, `intel_briefing`.
- Daily-produced JSON reports under `reports/`: `paper_trading_account.json` (live NAV + open positions + history), `briefing_latest.json`, `selected_tickers.json` (daily top-N screener output), `target_weights.csv`, `execution_quality.json`, etc.
- A canonical `universe.json` of 951 verified Japan tickers.

HotThemeRotator already had a narrow read-only bridge at `src/hot_theme_rotator/data/legacy_project_adapter.py` (P1-02) for prices and news. That bridge predates the active P8-09 dashboard direction and only covers two table families.

## Decision

HotThemeRotator's `data/` module is the **read-only consumer** of Project_optimized's data lifecycle. HotThemeRotator never writes to `japan_market.db` and never modifies Project_optimized's reports. Specifically:

1. **Positions**: `data/position_adapter.py` reads `japan_market.db.positions` + `japan_market.db.account_snapshots` tables filtered by `strategy_id` (default `etf_buyhold` = user's Path A live with 1306.T 900 股 @ ¥403). Returns `PortfolioState` (cash, NAV, positions list with avg cost / market price / unrealized P&L). **NOTE**: `Project_optimized/reports/paper_trading_account.json` is NOT used — that JSON snapshots the decommissioned `sprint` strategy (3041.T) and would mislead the user about current live holdings.
2. **K-line / OHLC**: `data/kline_adapter.py` reads the `daily_prices` table for any symbol + window. Doubles as P9-02's `LegacyDailyPriceFetcher` (satisfies the `PriceFetcher` Protocol).
3. **Multi-market temperature**: `data/market_temp_adapter.py` reads `cross_asset_snapshots` table and computes the 6-market mosaic (日经/TOPIX/SOX/S&P/USDJPY/上证 — with sparkline tail) — falls back to in-source numbers when table is sparse.
4. **Theme heat**: `data/theme_heat_adapter.py` reads `factor_signals` + `signals` tables and ranks themes by heat score; matches the V1-V4 themes section shape.
5. **News timeline**: `data/news_adapter.py` reads `news_feed` + `news_items` tables for the last N hours; produces V1-V4 newsTimeline shape with weight + linkedSymbols.
6. **Universe / selected tickers**: `data/universe_adapter.py` reads `universe.json` + `selected_tickers.json` for the real candidate scanner backing P8-15.

The Python `api/serializers.py` calls all adapters, returns the V3 JSON shape filled with real data. Frontend variants render without knowing the data came from a sibling project.

Boundary contracts:

- HotThemeRotator does **not** publish positions back. Rule 3 holds at API layer: only GET on /api/positions, no POST.
- HotThemeRotator does **not** trigger Project_optimized's daily runs. If Project_optimized hasn't refreshed today's data, HotThemeRotator shows yesterday's data with a "stale" badge (rather than fabricating).
- Each adapter docstring snapshots the table schema columns it depends on, so a Project_optimized schema migration breaks loudly here.

## Consequences

Positive:

- The dashboard becomes USABLE — user sees current 3041.T holding, today's real market temperature, today's news, today's K-line. Decision-making surface, not a research demo.
- Zero data duplication. The 6+ months of curated Project_optimized data is leveraged in place.
- P9-02's `PriceFetcher` Protocol gets its real-data wrapper (`LegacyDailyPriceFetcher`) as a side effect of P8-14.
- Cross-project decision log: positions read from Project_optimized + predictions written to HotThemeRotator's `reports/predictions/` creates a complete observe→predict→outcome loop.

Negative:

- HotThemeRotator becomes tightly coupled to Project_optimized's directory layout and DB schema. A move/rename in Project_optimized breaks us.
- Two projects on disk to maintain awareness of. Easier to confuse "edit which one?".
- Project_optimized's data freshness is now visible to the dashboard — if its daily runs lapse, HotThemeRotator inherits the staleness.

Risks and mitigations:

- **Risk**: Project_optimized DB schema migration (rare but possible). **Mitigation**: each adapter pins the column list it uses in its docstring; integration tests guard against missing columns; `data/legacy_project_adapter.py` already proves this pattern works.
- **Risk**: Project_optimized's `paper_trading_account.json` could record real-money positions in the future. **Mitigation**: HotThemeRotator never trades; per Rule 3, positions are display-only; no broker auth code reads this file.
- **Risk**: User edits Project_optimized data and HotThemeRotator silently consumes the edit. **Mitigation**: adapters surface `asof` timestamps prominently; UI shows when underlying data was last refreshed.

## Alternatives Considered

- **Rebuild a separate data pipeline in HotThemeRotator** — Rejected: duplicates 6+ months of work, fragments the user's single source of truth.
- **Move HotThemeRotator into Project_optimized** — Rejected: the projects have different goals (Project_optimized is live execution; HotThemeRotator is research dashboard); their dependency hierarchy already correctly places HotThemeRotator downstream.
- **Periodic data export from Project_optimized to HotThemeRotator-local copy** — Rejected: introduces sync lag without removing schema coupling. Direct read is simpler.

## Out of Scope

- Writing back to Project_optimized (forbidden by Rule 3 + this ADR).
- Cross-project shared modules — each project keeps its own `src/`.
- Migration of Project_optimized's `intraday_decision.py` etc. — covered in `docs/04_DATA_AND_OPEN_SOURCE.md` migration rule (wrap, don't copy).
- Real-money broker integration — `paper_trading_account.json` is paper-trading; live broker still requires §10 gate 8 explicit approval.

## References

- `docs/02_GOVERNANCE.md` Rule 3 / Rule 4 / §3 / §10 gate 8.
- `docs/04_DATA_AND_OPEN_SOURCE.md` (Project_optimized migration rule).
- `docs/adr/ADR-0003-decision-log.md` (predictions logged in HotThemeRotator).
- `docs/adr/ADR-0004-fastapi-frontend.md` (the dashboard that consumes these adapters).
- `src/hot_theme_rotator/data/legacy_project_adapter.py` (existing narrow bridge, P1-02 baseline pattern).
- `Project_optimized/japan_market.db` (30+ tables).
- `Project_optimized/reports/paper_trading_account.json` (live position snapshots).
