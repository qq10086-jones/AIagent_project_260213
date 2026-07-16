"""Translate Python data-layer objects into the V3 dashboard JSON shape.

The serializer does NOT compute scores, derive ground truth, or run calibrations
— those stay in their respective modules. It only translates.

Markets / themes / newsTimeline / kline currently return empty lists because
the Python layer does not yet produce a multi-market temperature surface, a
theme-heat ranker, a news timeline view, or OHLC bars for the dashboard hero.
Frontend renders these as "数据未就绪" placeholders until upstream lands.
"""
from __future__ import annotations

import json
import sqlite3
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
from hot_theme_rotator.data.stock_news_fetcher import load_latest_news_timeline
from hot_theme_rotator.candidate_engine.catalyst import build_catalyst_map
from hot_theme_rotator.candidate_engine.hybrid_rerank import rerank
from hot_theme_rotator.candidate_engine.theme_rotation import (
    CORE_MEMORY_SEMI_SYMBOLS,
    annotate_rotation,
    assess_core_theme_coverage,
)
from hot_theme_rotator.candidate_engine.theme_leaders import annotate_theme_leaders
from hot_theme_rotator.data.ticker_metadata import load_metadata, themes_for_symbol

_SERIALIZER_ROOT = Path(__file__).resolve().parents[1]
from hot_theme_rotator.data.theme_heat_adapter import (
    ThemeHeatAdapterError,
    load_theme_heat,
)
from hot_theme_rotator.data.universe_adapter import (
    ScreenerSnapshot,
    UniverseAdapterError,
    default_selected_tickers_path,
    freshest_selected_tickers_path,
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
    news = _serialize_news(base_dir)
    adr_meta, skhy_desk, skhy_event = _serialize_adr_watch(  # P20-05 / Rule 11.15
        base_dir, asof_limit=observation_date  # PIT: never select a future-dated ADR snapshot
    )
    try:  # P20 fix#1 — read-only SKHY candidate annotations (capped; never reorders, never advice)
        from hot_theme_rotator.candidate_engine.skhy_overlay import annotate_japan_semi

        candidates = annotate_japan_semi(candidates, skhy_event)
    except Exception:
        pass
    meta = _build_meta(
        trade_date=trade_date,
        base_dir=base_dir,
        candidates_source=candidates_source,
        real_markets=bool(markets),
        real_themes=bool(themes),
        real_news=bool(news),
    )
    if isinstance(meta, dict):
        meta.setdefault("dataQuality", {})["adrWatch"] = adr_meta
    positions = _serialize_positions()
    try:  # P21-04 / Rule 11.16 — read-only trade-plan board; fail-open to None
        from hot_theme_rotator.reporting.action_board import build_action_board

        jp_session = next((m.get("state") for m in markets if m.get("region") == "JP"), None)
        action_board = build_action_board(
            candidates,
            nav_jpy=positions.get("nav"),
            cash_jpy=positions.get("cash"),
            market_session=jp_session,
        )
    except Exception:
        action_board = None
    try:  # P22-01 / Rule 11.17 — exit-discipline arithmetic on current holdings
        from hot_theme_rotator.reporting.exit_board import build_exit_board

        plan_ready = None
        if action_board:
            plan_ready = sum(1 for r in action_board.get("rows", [])
                             if r.get("planStatus") in ("plan_ready", "s_kabu_only"))
        exit_board = build_exit_board(positions, action_board_plan_ready=plan_ready)
    except Exception:
        exit_board = None
    try:  # P25-04 / Section 17 — owner risk-mandate sleeve panel; fail-open to None
        from hot_theme_rotator.risk.sleeve_engine import build_risk_mandate_panel

        risk_mandate = build_risk_mandate_panel(positions, base_dir=base_dir)
    except Exception:
        risk_mandate = None
    return {
        "meta": meta,
        "gates": _serialize_gates(),
        "markets": markets,            # P8-11
        "themes": themes,              # P8-12
        "candidates": candidates,
        "newsTimeline": news,          # P8-13
        "decisionLog": _serialize_decision_log(base_dir=base_dir, trade_date=trade_date),
        "kline": _serialize_kline(symbol=top_symbol, sessions=252),  # P8-14 + P8-16 C2: 1y window for MA60 / 52w lines
        "actionBoard": action_board,   # P21-04 / Rule 11.16 — plans, not predictions
        "exitBoard": exit_board,       # P22-01 / Rule 11.17 — operator's own exit discipline
        "riskMandate": risk_mandate,   # P25-04 / Section 17 — owner risk mandate sleeves (read-only)
        "positions": positions,        # P8-10
        "dailyCockpit": _serialize_daily_cockpit(
            base_dir=base_dir,
            trade_date=observation_date,
            watchlist=_watchlist_symbols(candidates, top_symbol),
        ),
        "macroNews": _load_macro_overlay(base_dir),  # P10-26 slice 3 — macro/policy news heat
        "sKabuOverlay": _serialize_s_kabu_overlay(base_dir),  # task#5/#7 — S株-unlocked held/watchlist names
        "eventDesk": {"skhy": skhy_desk},  # P20-05 / Rule 11.15 — external ADR catalyst (read-only)
    }


def _serialize_adr_watch(
    base_dir: Path | str | None,
    *,
    asof_limit: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """P20-05 / Rule 11.15: read-only external ADR (SKHY) watch state for the
    dashboard. Returns ``(adr_meta, skhy_desk, event)``. Fail-open — no snapshot,
    stale, future-dated, or any error yields ``status='unavailable'`` /
    ``eventState='off'`` and leaves JP ranking unchanged.

    PIT (P20-10 fix): only snapshots whose date is <= ``asof_limit`` (default
    today JST) are considered, and the latest valid one is chosen — so a
    future-dated file (e.g. ``adr_watch_2099-01-01.json``) NEVER activates SKHY.
    Never produces probability / win-rate / expected-return / edge."""
    disclosure = "External ADR catalyst; not a JP buy signal; no probability."
    off_event = {
        "skhyCatalystStatus": "unavailable", "skhyCatalystActive": False, "eventState": "off",
        "skhyOvernightMove": None, "skhyFresh": False, "confirmations": (),
        "abnormalMove": False, "freshness": 0.0, "disclosure": disclosure,
    }
    unavailable = (
        {"asof": None, "status": "unavailable", "stale": True},
        {"status": "off", "disclosure": disclosure + " ADR watch unavailable."},
        off_event,
    )
    try:
        limit = None
        if asof_limit:
            try:
                limit = date.fromisoformat(str(asof_limit)[:10])
            except ValueError:
                limit = None
        if limit is None:
            limit = date.today()
        root = Path(base_dir) if base_dir else Path(".")
        # PIT: keep only snapshots whose filename date is NOT in the future relative
        # to the observation date, then pick the latest valid one. A future-dated
        # file must never be selected/activated (Rule 11.15 fail-closed/PIT).
        dated: list[tuple[date, Path]] = []
        for p in (root / "reports" / "adr").glob("adr_watch_*.json"):
            try:
                snap_date = date.fromisoformat(p.stem[len("adr_watch_"):])
            except ValueError:
                continue
            if snap_date <= limit:
                dated.append((snap_date, p))
        if not dated:
            return unavailable
        chosen = max(dated)[1]
        payload = json.loads(chosen.read_text(encoding="utf-8"))
        # Defense in depth: reject if the snapshot's internal asof is future too.
        try:
            if date.fromisoformat(str(payload.get("asof", ""))[:10]) > limit:
                return unavailable
        except ValueError:
            pass
        instruments = payload.get("instruments", {}) or {}
        skhy = instruments.get("SKHY", {}) or {}
        from hot_theme_rotator.candidate_engine.skhy_overlay import compute_skhy_event

        event = compute_skhy_event(instruments, asof=str(payload.get("asof", "")))
        adr_meta = {
            "asof": payload.get("asof"),
            "status": skhy.get("status", "unavailable"),
            "stale": bool(skhy.get("stale", True)),
        }
        skhy_desk = {"status": event["eventState"], "disclosure": event["disclosure"]}
        return adr_meta, skhy_desk, event
    except Exception:
        return unavailable


def _serialize_s_kabu_overlay(base_dir: Path | str | None) -> list[dict[str, Any]]:
    """S株 overlay candidates (Rule 5.2 / task#5): held+watchlist names the
    100-share-lot gate excludes but S株 unlocks. Additive, never edge — fail-soft
    to [] so a price-DB miss never breaks the dashboard."""
    try:
        from hot_theme_rotator.candidate_engine.s_kabu_universe import build_s_kabu_overlay

        ov = build_s_kabu_overlay(base_dir)
        return list(ov.get("candidates", []))
    except Exception:
        return []


def _load_macro_overlay(base_dir: Path | str | None) -> dict[str, Any] | None:
    """P10-26 slice 3 — attach the HTR-native macro-news overlay (slice 2) if built,
    so the dashboard can surface macro / policy / cross-market news heat. Returns
    None when not yet built; the frontend renders nothing in that case."""
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    d = root / "reports" / "news_macro"
    if not d.exists():
        return None
    files = sorted(d.glob("*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_news_catalyst_quality(base_dir: Path, trade_date: str) -> dict[str, Any]:
    """Latest daily-routine status for the ADR-0009 catalyst inputs.

    News / metadata refresh failures are non-fatal in ``daily_routine`` so the
    price screener can still run. The dashboard still needs to surface that the
    news-catalyst layer degraded; otherwise a badge can look authoritative while
    it is really a stale or pure-screener fallback.
    """
    status = {
        "degraded": False,
        "newsRefreshOk": None,
        "metadataRefreshOk": None,
        "asof": None,
        "reasons": [],
    }
    log_path = base_dir / "reports" / "observability" / "daily_routine_log.jsonl"
    if not log_path.exists():
        return status
    latest: dict[str, Any] | None = None
    try:
        for raw in log_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("mode") != "afterclose":
                continue
            if trade_date and row.get("asof") != trade_date:
                continue
            if isinstance(row.get("candidates"), dict):
                latest = row
    except (OSError, json.JSONDecodeError):
        return {
            **status,
            "degraded": True,
            "reasons": ["daily_routine_log_unreadable"],
        }
    if latest is None:
        return status

    candidates = latest.get("candidates") or {}
    news_ok = candidates.get("news_refresh_rc") == 0
    meta_ok = candidates.get("meta_refresh_rc") == 0
    reasons: list[str] = []
    if not news_ok:
        reasons.append("news_refresh_failed")
    if not meta_ok:
        reasons.append("metadata_refresh_failed")
    return {
        "degraded": bool(reasons),
        "newsRefreshOk": news_ok,
        "metadataRefreshOk": meta_ok,
        "asof": latest.get("asof"),
        "reasons": reasons,
    }


def _core_theme_coverage(trade_date: str) -> dict[str, Any]:
    path = default_db_path()
    price_dates: dict[str, str] = {}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for symbol in CORE_MEMORY_SEMI_SYMBOLS:
                row = conn.execute(
                    "SELECT MAX(date) FROM daily_prices WHERE symbol = ?",
                    (symbol,),
                ).fetchone()
                if row and row[0]:
                    price_dates[symbol] = str(row[0])
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - coverage is diagnostic; dashboard must still load
        price_dates = {}
    return assess_core_theme_coverage(price_dates, latest_trade_date=trade_date)


def _to_eastern(now_jst: datetime) -> datetime:
    """Convert a JST datetime to US/Eastern (DST-aware via zoneinfo; EDT fallback)."""
    try:
        from zoneinfo import ZoneInfo
        return now_jst.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        # No tz database available — assume EDT (UTC-4); off by 1h in US winter only.
        return now_jst.astimezone(timezone(timedelta(hours=-4)))


def _in_window(dt: datetime, start_min: int, end_min: int) -> bool:
    mins = dt.hour * 60 + dt.minute
    return start_min <= mins < end_min


def _market_session_state(region: str, raw_state: str, now_jst: datetime) -> str:
    """Rule 11.9.3 — derive OPEN/CLOSED/LIVE from a real clock + region trading
    hours, NOT the adapter's hard-coded label.

    The market-temp adapter hard-codes per-market `state` literals (JP "OPEN",
    US "CLOSED", FX "LIVE"), calendar/clock-blind — so it surfaced JP "OPEN" at
    23:39 JST (closed since 15:30) and US "CLOSED" while the US session was open.
    Exchange holidays are not yet modelled (P12-03 follow-up — a JP holiday weekday
    would still read OPEN); the weekday + session-hours derivation is the reported
    defect's fix and the frontend freshness label still covers holidays via the
    asof↔tradeDate gap.
    """
    if raw_state == "UNKNOWN":
        return "UNKNOWN"
    region = (region or "").upper()
    if region == "JP":
        if now_jst.weekday() >= 5:
            return "CLOSED"
        # TSE cash equities: 09:00-11:30 and 12:30-15:30 JST (lunch break = closed).
        return "OPEN" if (_in_window(now_jst, 540, 690) or _in_window(now_jst, 750, 930)) else "CLOSED"
    if region == "US":
        et = _to_eastern(now_jst)
        if et.weekday() >= 5:
            return "CLOSED"
        return "OPEN" if _in_window(et, 570, 960) else "CLOSED"   # 09:30-16:00 ET
    if region == "FX":
        # Spot FX runs ~24h on weekdays; closed over the weekend.
        return "CLOSED" if now_jst.weekday() >= 5 else "LIVE"
    return "CLOSED"


def _serialize_markets() -> list[dict[str, Any]]:
    """P8-11 — 6-market mosaic; soft-fail to [] so frontend mock fallback engages."""
    try:
        tiles = load_market_mosaic(default_db_path(), sessions=16)
    except MarketTempAdapterError:
        return []
    now_jst = datetime.now(timezone(timedelta(hours=9)))
    return [
        {
            "id": t.id,
            "label": t.label,
            "sub": t.sub,
            "region": t.region,
            "state": _market_session_state(t.region, t.state, now_jst),
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


def _serialize_news(base_dir: Path | None = None) -> list[dict[str, Any]]:
    """P8-13 — recent news timeline. Prefer the FRESH HTR-native store
    (`reports/news/{date}.json`, kept current by the daily Google-News fetch,
    2026-06-14); fall back to the (frozen) sibling DB news tables, then [] for
    mock fallback. The HTR rows already carry JST-formatted ts + theme tags."""
    if base_dir is not None:
        htr = load_latest_news_timeline(base_dir, limit=20)
        if htr:
            return htr
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


def _htr_screener_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "reports" / "screener"


def _latest_theme_counts(base_dir: Path) -> dict[str, int]:
    news_dir = base_dir / "reports" / "news"
    if not news_dir.exists():
        return {}
    files = sorted(news_dir.glob("*.json"))
    if not files:
        return {}
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    counts = payload.get("theme_counts") or {}
    return {str(k): int(v or 0) for k, v in counts.items()}


def _metadata_name(symbol: str, metadata: dict[str, dict[str, Any]]) -> str:
    meta = metadata.get(symbol) or {}
    return str(meta.get("name") or meta.get("longName") or meta.get("shortName") or "")


def _freshest_selected_tickers_path() -> Path:
    """Prefer the freshest HTR-native screener snapshot over the sibling file.

    P12-03 (Rule 11.9 freshness): the dashboard candidate panel used to read
    ``Project_optimized/selected_tickers.json``, which goes stale because the sibling
    refreshes ``daily_prices`` but NOT its screener output (it sat at 2026-05-27 on the
    06-01 launch). ``daily_routine`` afterclose writes HTR-native
    ``reports/screener/selected_tickers_{date}.json`` daily; prefer the newest of those
    when it is at least as fresh (by trade_date) as the sibling, so the dashboard shows
    today's candidates and stops depending on the un-refreshed sibling file.
    """
    return freshest_selected_tickers_path(
        base_dir=_SERIALIZER_ROOT,
        sibling_path=default_selected_tickers_path(),
        htr_screener_dir=_htr_screener_dir(),
    )


def _real_or_sample_candidates(top_n: int) -> tuple[list[dict[str, Any]], str, str, str]:
    """Try real screener output first; fall back to sample fixture on failure.

    Returns (candidates_json, top_symbol, trade_date, source_label).
    """
    try:
        snapshot = load_screener_snapshot(_freshest_selected_tickers_path())
        if snapshot.tickers:
            real = _serialize_real_candidates(snapshot, top_n=top_n)
            # Leader = the reranked #1 (news-catalyzed), not the raw screener #1.
            top_symbol = real[0]["symbol"] if real else snapshot.tickers[0].symbol
            return (real, top_symbol, snapshot.asof, "screener_v2")
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
    """Build V1-V4 candidate dicts from real screener rows + real OHLC ref prices,
    reranked by the news catalyst (ADR-0009 P15-02d).

    The price screener gives the strong/liquid universe; the hybrid rerank blends in
    today's news theme-heat so news-catalyzed hot-theme leaders surface. Honest:
    `score` stays the raw (absolute) uncalibrated research score; the ORDER reflects
    the news-catalyzed blend; each candidate carries its theme/catalyst annotation so
    the ordering is explained, not hidden. Fail-open — when news/metadata are absent
    the catalyst is 0 and the order degrades to the pure screener (Rule 11.9)."""
    metadata = load_metadata()
    lite: list[dict[str, Any]] = []
    price_features: dict[str, dict[str, Any]] = {}
    for t in snapshot.tickers:
        raw = t.raw or {}
        stale_price = _recent_price_stale(t.symbol)
        themes = themes_for_symbol(t.symbol, metadata)
        lite.append(
            {
                "symbol": t.symbol,
                "score": float(t.score),
                "fundamental_score": float(t.fundamental_score or 1.0),
                "mom_20": float(t.mom_20 or 0.0),
                "stale_price": stale_price,
                "themes": themes,
                "_t": t,
            }
        )
        price_features[t.symbol] = {
            "ret5": float(raw.get("ret5", raw.get("mom_5", 0.0)) or 0.0),
            "ret20": float(t.mom_20 or 0.0),
            "ret60": float(t.mom_60 or 0.0),
            "dist_high20": float(raw.get("dist_high20", raw.get("distFromHigh20d", -1.0)) or -1.0),
            "dist_high52": float(raw.get("dist_high52", raw.get("distFromHigh52w", -1.0)) or -1.0),
            "volume_z": float(raw.get("vol_z", 0.0) or 0.0),
            "relative5_vs_leader": float(raw.get("relative5_vs_leader", 0.0) or 0.0),
            "relative5_vs_benchmark": float(raw.get("relative5_vs_benchmark", 0.0) or 0.0),
            "fundamental_score": float(t.fundamental_score or 0.0),
            "fresh": not stale_price,
        }
    try:
        catmap = build_catalyst_map([t.symbol for t in snapshot.tickers], base_dir=_SERIALIZER_ROOT)
    except Exception:  # noqa: BLE001 — catalyst is best-effort; never break the panel
        catmap = {}
    for item in lite:
        company_catalyzed = catmap.get(item["symbol"], {}).get("evidence_level") == "company"
        item["company_catalyzed"] = company_catalyzed
        price_features[item["symbol"]]["company_catalyzed"] = company_catalyzed
    lite = annotate_rotation(
        lite,
        theme_counts=_latest_theme_counts(_SERIALIZER_ROOT),
        price_features=price_features,
    )
    reranked = rerank(lite, catmap)

    out: list[dict[str, Any]] = []
    for rank, row in enumerate(reranked[:top_n], start=1):
        ticker = row["_t"]
        # Real reference price: latest close from kline_adapter; fall back to
        # the screener's recorded close if the DB lookup fails.
        # Real reference price + real daily change from the last two (split-
        # adjusted) bars, instead of the placeholder chg=0.0 every candidate used.
        ref_bar = None
        chg = 0.0
        try:
            recent = fetch_kline(default_db_path(), symbol=ticker.symbol, sessions=2)
            if recent:
                ref_bar = recent[-1]
                if len(recent) >= 2 and recent[-2].close:
                    chg = round((recent[-1].close / recent[-2].close - 1.0) * 100.0, 2)
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
        top_theme = row.get("top_theme")
        catalyzed = bool(row.get("news_catalyzed"))
        sector_catalyzed = bool(row.get("sector_catalyzed"))
        themed = catalyzed or sector_catalyzed
        catalyst_score = float(row.get("catalyst_score", 0.0))
        out.append(
            {
                "rank": rank,
                "symbol": ticker.symbol,
                "nameJa": _metadata_name(ticker.symbol, metadata),
                "nameCn": "",
                "theme": top_theme if themed else "screener_v2",
                "themeId": top_theme if themed else "screener",
                # ADR-0009 P15-02d — news-catalyst annotations (order reflects these).
                "newsCatalyzed": catalyzed,
                "companyCatalyzed": bool(row.get("company_catalyzed")),
                "sectorCatalyzed": sector_catalyzed,
                "catalystEvidenceLevel": row.get("catalyst_evidence_level", "none"),
                "catalystEvidence": row.get("catalyst_evidence", {}),
                "topTheme": top_theme,
                "hotThemes": list(row.get("hot_themes", []) or []),
                "catalystScore": round(catalyst_score, 4),
                "blendedScore": round(float(row.get("blended_score", 0.0)) * 100, 2),
                "qualityPenalty": round(float(row.get("quality_penalty", 0.0)) * 100, 2),
                "themeRegime": row.get("theme_regime"),
                "leaderSymbol": row.get("leader_symbol"),
                "leaderExtended": bool(row.get("leader_extended")),
                "chaseRisk": row.get("chase_risk", "none"),
                "secondLineCandidate": bool(row.get("second_line_candidate")),
                "rotationScore": round(float(row.get("rotation_score", 0.0)) * 100, 2),
                "rotationReasons": list(row.get("rotation_reasons", []) or []),
                # P20 fix#1 — real theme membership + recent return for the read-only
                # SKHY sympathy annotation (applied later in build_dashboard_payload).
                "themes": list(themes_for_symbol(ticker.symbol, metadata) or []),
                "recentReturn": (ticker.raw or {}).get("ret5"),
                "priority": _priority_label(score_0_100),
                "score": score_0_100,
                "scoreStatus": "warning",
                "price": round(float(ref_bar.close), 2),
                "chg": chg,
                "one_liner": (
                    (f"🔥 {top_theme} 催化 · " if catalyzed else "")
                    + f"alpha {ticker.score:.3f} · ADV ¥{ticker.adv/1e6:.0f}M · "
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
    # ADR-0009 P15-02b — mark the top candidate in each hot theme as its leader.
    return annotate_theme_leaders(out)


def _recent_price_stale(symbol: str, *, sessions: int = 6) -> bool:
    try:
        bars = fetch_kline(default_db_path(), symbol=symbol, sessions=sessions)
    except KlineAdapterError:
        return False
    if len(bars) < 5:
        return False
    closes = [round(float(b.close), 6) for b in bars[-5:]]
    return len(set(closes)) == 1


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
        "dataQuality": {
            "newsCatalyst": _latest_news_catalyst_quality(base_dir, trade_date),
            "coreThemeCoverage": _core_theme_coverage(trade_date),
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
