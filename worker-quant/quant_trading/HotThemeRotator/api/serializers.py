"""Translate Python data-layer objects into the V3 dashboard JSON shape.

The serializer does NOT compute scores, derive ground truth, or run calibrations
— those stay in their respective modules. It only translates.

Markets / themes / newsTimeline / kline currently return empty lists because
the Python layer does not yet produce a multi-market temperature surface, a
theme-heat ranker, a news timeline view, or OHLC bars for the dashboard hero.
Frontend renders these as "数据未就绪" placeholders until upstream lands.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hot_theme_rotator.calibration.reporter import DEFAULT_MIN_SAMPLES
from hot_theme_rotator.common.schema import PriceBar
from hot_theme_rotator.data.position_adapter import (
    DEFAULT_STRATEGY_ID,
    PositionAdapterError,
    default_db_path,
    load_portfolio_state,
)
from hot_theme_rotator.data.kline_adapter import (
    KlineAdapterError,
    fetch_kline,
    fetch_latest_close,
)
from hot_theme_rotator.data.market_temp_adapter import (
    MarketTempAdapterError,
    load_market_mosaic,
)
from hot_theme_rotator.data.news_adapter import (
    NewsAdapterError,
    load_news_timeline,
)
from hot_theme_rotator.data.theme_heat_adapter import (
    ThemeHeatAdapterError,
    load_theme_heat,
)
from hot_theme_rotator.data.universe_adapter import (
    ScreenerSnapshot,
    UniverseAdapterError,
    default_selected_tickers_path,
    load_screener_snapshot,
)
from hot_theme_rotator.data.external.realtime_price.health import read_price_health_report
from hot_theme_rotator.data.external.tdnet_storage import read_disclosures
from hot_theme_rotator.watchlist_intelligence.silent_queue import read_silent_events
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder
from hot_theme_rotator.decision_log.jsonl_writer import read_predictions
from hot_theme_rotator.opportunity.opportunity_scanner import OpportunityCandidate
from hot_theme_rotator.opportunity.price_ladder import PriceLadder
from hot_theme_rotator.reporting.daily_advisory_cockpit import (
    build_daily_advisory_cockpit,
)
from hot_theme_rotator.reporting.realtime_opportunity_panel import OpportunityPanelRow
from hot_theme_rotator.ui.opportunity_dashboard import (
    DashboardPanel,
    build_calibration_badge,
    build_gate_progress_rows,
    build_recent_predictions_view,
    build_sample_panel,
)


def build_dashboard_payload(
    *,
    base_dir: Path,
    top_n: int = 10,
    cockpit_trade_date: str | None = None,
) -> dict[str, Any]:
    """Build the /api/dashboard response payload from the Python data layer.

    Candidate path priority (P8-15):
    1. Real screener output (`Project_optimized/selected_tickers.json`) — preferred.
    2. Sample fixture (`build_sample_panel`) — fallback when screener unavailable.
    Frontend sees the same shape; only the underlying source differs.
    """
    candidates, top_symbol, trade_date, candidates_source = _real_or_sample_candidates(top_n=top_n)
    observation_date = cockpit_trade_date or date.today().isoformat()
    markets = _serialize_markets()
    themes = _serialize_themes()
    news = _serialize_news()
    return {
        "meta": _build_meta(
            trade_date=trade_date,
            base_dir=base_dir,
            candidates_source=candidates_source,
            real_markets=bool(markets),
            real_themes=bool(themes),
            real_news=bool(news),
        ),
        "gates": _serialize_gates(),
        "markets": markets,            # P8-11
        "themes": themes,              # P8-12
        "candidates": candidates,
        "newsTimeline": news,          # P8-13
        "decisionLog": _serialize_decision_log(base_dir=base_dir, trade_date=trade_date),
        "kline": _serialize_kline(symbol=top_symbol, sessions=252),  # P8-14 + P8-16 C2: 1y window for MA60 / 52w lines
        "positions": _serialize_positions(),  # P8-10
        "dailyCockpit": _serialize_daily_cockpit(
            base_dir=base_dir,
            trade_date=observation_date,
            watchlist=_watchlist_symbols(candidates, top_symbol),
        ),
    }


def _serialize_markets() -> list[dict[str, Any]]:
    """P8-11 — 6-market mosaic; soft-fail to [] so frontend mock fallback engages."""
    try:
        tiles = load_market_mosaic(default_db_path(), sessions=16)
    except MarketTempAdapterError:
        return []
    return [
        {
            "id": t.id,
            "label": t.label,
            "sub": t.sub,
            "region": t.region,
            "state": t.state,
            "price": t.price,
            "chg": t.chg,
            "temp": t.temp,
            "spark": list(t.spark),
            "asof": t.asof,
        }
        for t in tiles
    ]


def _serialize_themes() -> list[dict[str, Any]]:
    """P8-12 — top alpha factors as themes; soft-fail to [] for mock fallback."""
    try:
        rows = load_theme_heat(default_db_path(), top_n=6, leaders_per_theme=3)
    except ThemeHeatAdapterError:
        return []
    return [
        {
            "id": r.id,
            "label": r.label,
            "leaders": list(r.leaders),
            "heat": r.heat,
            "mom": r.momentum,  # V3 reads `.mom`
            "asof": r.asof,
        }
        for r in rows
    ]


def _serialize_news() -> list[dict[str, Any]]:
    """P8-13 — recent news timeline; soft-fail to [] for mock fallback."""
    try:
        rows = load_news_timeline(default_db_path(), hours=12, limit=20)
    except NewsAdapterError:
        return []
    return [
        {
            "ts": _format_ts_jst(r.ts),
            "src": r.src,
            "weight": r.weight,
            "text": r.text,
            "linkedSymbols": list(r.linked_symbols),
        }
        for r in rows
    ]


def _serialize_daily_cockpit(
    *,
    base_dir: Path,
    trade_date: str,
    watchlist: tuple[str, ...],
) -> dict[str, Any]:
    try:
        price_health = read_price_health_report(trade_date, base_dir=base_dir)
    except Exception:
        price_health = ()
    try:
        disclosures = tuple(
            item.to_dict() for item in read_disclosures(trade_date=trade_date, base_dir=base_dir)
        )
    except Exception:
        disclosures = ()
    try:
        silent_events = tuple(item.to_dict() for item in read_silent_events(trade_date, base_dir=base_dir))
    except Exception:
        silent_events = ()
    return build_daily_advisory_cockpit(
        trade_date=trade_date,
        watchlist=watchlist,
        price_health_rows=price_health,
        tdnet_disclosures=disclosures,
        silent_queue_rows=silent_events,
    )


def _watchlist_symbols(candidates: list[dict[str, Any]], top_symbol: str) -> tuple[str, ...]:
    symbols: list[str] = []
    if top_symbol:
        symbols.append(top_symbol)
    for candidate in candidates[:10]:
        symbol = str(candidate.get("symbol", ""))
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return tuple(symbols)


def _format_ts_jst(iso_ts: str) -> str:
    """`2026-05-21T20:00:00` → `05-21 20:00 JST`."""
    raw = str(iso_ts)
    if "T" in raw:
        date_part, time_part = raw.split("T", 1)
        # Strip seconds if present, drop any timezone suffix
        time_clean = time_part.split("+")[0].split("Z")[0].split(".")[0]
        if time_clean.count(":") >= 2:
            time_clean = ":".join(time_clean.split(":")[:2])
        md = date_part[5:] if len(date_part) >= 10 else date_part
        return f"{md} {time_clean} JST"
    return raw


def _real_or_sample_candidates(top_n: int) -> tuple[list[dict[str, Any]], str, str, str]:
    """Try real screener output first; fall back to sample fixture on failure.

    Returns (candidates_json, top_symbol, trade_date, source_label).
    """
    try:
        snapshot = load_screener_snapshot(default_selected_tickers_path())
        if snapshot.tickers:
            real = _serialize_real_candidates(snapshot, top_n=top_n)
            return (real, snapshot.tickers[0].symbol, snapshot.asof, "screener_v2")
    except UniverseAdapterError:
        pass
    panel = build_sample_panel(top_n=top_n)
    top_symbol = panel.rows[0].candidate.symbol if panel.rows else ""
    return (_serialize_candidates(panel.rows), top_symbol, panel.trade_date, "sample")


def _serialize_real_candidates(
    snapshot: ScreenerSnapshot,
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Build V1-V4 candidate dicts from real screener rows + real OHLC ref prices."""
    out: list[dict[str, Any]] = []
    for rank, ticker in enumerate(snapshot.tickers[:top_n], start=1):
        # Real reference price: latest close from kline_adapter; fall back to
        # the screener's recorded close if the DB lookup fails.
        ref_bar = None
        try:
            ref_bar = fetch_latest_close(default_db_path(), symbol=ticker.symbol)
        except KlineAdapterError:
            ref_bar = None
        if ref_bar is None:
            ref_bar = PriceBar.from_dict({
                "symbol": ticker.symbol,
                "asof": snapshot.asof,
                "open": ticker.close * 0.99,
                "high": ticker.close * 1.01,
                "low": ticker.close * 0.98,
                "close": ticker.close,
                "volume": max(ticker.adv / max(ticker.close, 1.0), 1.0),
                "turnover_jpy": ticker.adv,
            })
        ladder = build_price_ladder(ref_bar)
        score_0_100 = round(float(ticker.score) * 100, 2)
        out.append(
            {
                "rank": rank,
                "symbol": ticker.symbol,
                "nameJa": "",
                "nameCn": "",
                "theme": "screener_v2",
                "themeId": "screener",
                "priority": _priority_label(score_0_100),
                "score": score_0_100,
                "scoreStatus": "warning",
                "price": round(float(ref_bar.close), 2),
                "chg": 0.0,
                "one_liner": (
                    f"alpha {ticker.score:.3f} · ADV ¥{ticker.adv/1e6:.0f}M · "
                    f"mom20 {ticker.mom_20:+.1%} · vol {ticker.vol:.3f}"
                ),
                "reason": (
                    f"mom20 {ticker.mom_20:+.2%} · mom60 {ticker.mom_60:+.2%} · "
                    f"sharpe20 {ticker.sharpe_20:.2f} · adv_rank {ticker.adv_rank:.2f}"
                ) if ticker.mom_20 else (ticker.reason or "screener entry"),
                "risk": "未校准研究分。P9-03 校准需 ≥100 配对样本后才能视为胜率。",
                "dataQuality": (
                    f"{snapshot.version} asof {snapshot.asof}"
                    + (" · ref_price from real OHLC" if ref_bar else " · ref_price from screener close")
                ),
                "ladder": _serialize_ladder(ladder),
                "decisionCutoff": snapshot.asof,
            }
        )
    return out


def _serialize_kline(*, symbol: str, sessions: int = 40) -> list[dict[str, Any]]:
    """Last `sessions` OHLC bars for `symbol` from japan_market.db.

    Returns [] silently on missing DB / unknown symbol / column drift so the
    frontend falls back to mock K-line without breaking the response shape.
    Errors are logged via the adapter's exception (caught here).
    """
    if not symbol:
        return []
    try:
        bars = fetch_kline(default_db_path(), symbol=symbol, sessions=sessions)
    except KlineAdapterError:
        return []
    return [
        {
            "open": round(b.open, 4),
            "high": round(b.high, 4),
            "low": round(b.low, 4),
            "close": round(b.close, 4),
            "vol": round(b.volume),
            "date": b.asof,
        }
        for b in bars
    ]


def _serialize_positions(strategy_id: str = DEFAULT_STRATEGY_ID) -> dict[str, Any]:
    """Return current portfolio state from `japan_market.db` (ADR-0005).

    Reads positions + account_snapshots tables filtered by `strategy_id`
    (default `etf_buyhold` = user's Path A live). Fail-soft: any read error
    surfaces as `available=False` so frontend renders '持仓数据未就绪'
    rather than silently fabricating zeros.
    """
    try:
        state = load_portfolio_state(default_db_path(), strategy_id=strategy_id)
    except PositionAdapterError as exc:
        return {
            "available": False,
            "error": str(exc),
            "strategy_id": strategy_id,
            "asof": None,
            "positions_asof": None,
            "cash": None,
            "nav": None,
            "positions_value": None,
            "holdings": [],
        }
    return {
        "available": True,
        "error": None,
        "strategy_id": state.strategy_id,
        "asof": state.asof,
        "positions_asof": state.positions_asof,
        "cash": round(state.cash, 2),
        "nav": round(state.nav, 2),
        "positions_value": round(state.positions_value, 2),
        "holdings": [
            {
                "symbol": h.symbol,
                "asof": h.asof,
                "qty": h.qty,
                "avg_cost": round(h.avg_cost, 2),
                "market_price": round(h.market_price, 2),
                "market_value": round(h.market_value, 2),
                "unrealized_pnl": round(h.unrealized_pnl, 2),
                "unrealized_return_pct": round(h.unrealized_return_pct, 2),
            }
            for h in state.holdings
        ],
    }


def _build_meta(
    *,
    trade_date: str,
    base_dir: Path,
    candidates_source: str = "sample",
    real_markets: bool = False,
    real_themes: bool = False,
    real_news: bool = False,
) -> dict[str, Any]:
    now_jst = datetime.now(timezone(timedelta(hours=9)))
    try:
        predictions = read_predictions(trade_date=trade_date, base_dir=base_dir)
    except Exception:
        predictions = ()
    sample = len(predictions)
    # §9.4 — without P9-02 outcomes accumulating to ≥ DEFAULT_MIN_SAMPLES paired
    # records, calibration must stay `insufficient_calibration`. We never publish
    # a numeric brier from this endpoint.
    badge = build_calibration_badge("insufficient_calibration")
    return {
        "asof": now_jst.isoformat(timespec="seconds"),
        "tradeDate": trade_date,
        "refreshLabel": "120s",
        "eventTrigger": False,
        "candidatesSource": candidates_source,  # "screener_v2" | "sample"
        "calibration": {
            "level": badge["level"],
            "text": badge["text"],
            "sample": sample,
            "minSamples": DEFAULT_MIN_SAMPLES,
            "brier": None,
        },
        "dataAvailability": {
            "markets": real_markets,        # P8-11 — cross_asset_snapshots
            "themes": real_themes,          # P8-12 — factor_signals
            "newsTimeline": real_news,      # P8-13 — news_feed + news_items
            "kline": True,                  # P8-14 — daily_prices
            "positions": True,              # P8-10 — positions + account_snapshots
            "candidates": candidates_source != "sample",  # P8-15
        },
    }


def _serialize_gates() -> list[dict[str, str]]:
    """Source of truth: `_GATE_DEFINITIONS` via build_gate_progress_rows()."""
    return build_gate_progress_rows()


def _serialize_candidates(rows: tuple[OpportunityPanelRow, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        candidate = row.candidate
        ladder = row.ladder
        out.append(
            {
                "rank": candidate.rank,
                "symbol": candidate.symbol,
                "nameJa": "",   # Python layer doesn't carry instrument names
                "nameCn": "",
                "theme": candidate.trigger_theme,
                "themeId": candidate.trigger_theme,
                "priority": _priority_label(candidate.opportunity_score),
                "score": float(candidate.opportunity_score),
                "scoreStatus": "warning",
                "price": float(candidate.reference_price),
                "chg": 0.0,
                "one_liner": _summary_text(candidate),
                "reason": ", ".join(candidate.reason_codes) if candidate.reason_codes else "—",
                "risk": "未校准研究分。P9-03 校准需 ≥ 100 配对样本后才可视为胜率。",
                "dataQuality": (
                    "数据缺口：" + ("、".join(candidate.data_gaps) if candidate.data_gaps else "无")
                ),
                "ladder": _serialize_ladder(ladder),
                "decisionCutoff": candidate.decision_cutoff or "—",
            }
        )
    return out


def _serialize_ladder(ladder: PriceLadder) -> list[dict[str, Any]]:
    ref = float(ladder.reference_price)
    if ref <= 0:
        ref = 1.0
    tiers = (
        ("exit_stretch", "延伸卖出", ladder.stretch_exit),
        ("exit_2", "卖出 2", ladder.second_exit),
        ("exit_1", "卖出 1", ladder.first_exit),
        ("entry_aggressive", "买入 · 激进", ladder.aggressive_entry),
        ("entry_balanced", "买入 · 均衡", ladder.balanced_entry),
        ("entry_conservative", "买入 · 保守", ladder.conservative_entry),
        ("stop", "止损", ladder.stop_price),
    )
    return [
        {
            "kind": kind,
            "label": label,
            "price": float(price),
            "pct": round((float(price) - ref) / ref * 100, 2),
        }
        for kind, label, price in tiers
    ]


def _serialize_decision_log(
    *,
    base_dir: Path,
    trade_date: str,
) -> list[dict[str, Any]]:
    try:
        rows = build_recent_predictions_view(base_dir=base_dir, trade_date=trade_date)
    except Exception:
        rows = []
    out: list[dict[str, Any]] = []
    for row in rows[:8]:
        score = row.get("机会分")
        out.append(
            {
                "ts": row.get("决策时刻", "—"),
                "symbol": row.get("符号", "—"),
                "score": score if isinstance(score, (int, float)) else None,
                "action": "candidate_persisted",
                "note": f"模型 {row.get('模型', '—')}",
            }
        )
    return out


def _priority_label(score: float) -> str:
    if score >= 75:
        return "重点关注"
    if score >= 60:
        return "可观察"
    return "观察"


def _summary_text(candidate: OpportunityCandidate) -> str:
    if candidate.reason_codes:
        return "触发：" + "、".join(candidate.reason_codes)
    return "—"
