"""Event Desk — event/theme → exposed names → priced-in read (ADR-0009 P16, Rule 11.13).

The owner trades on events (停戦 / AI-semi / storage surge / yen-160). This module is
the honest analysis-support core: given an event or theme it returns (a) which liquid JP
names are EXPOSED and (b) a PRICED-IN read for each — how much it has ALREADY moved
(1d/5d/20d), versus the passive benchmark, and a purely descriptive freshness label
(fresh / extended / rolling_over / falling).

It does NOT predict the event's outcome and emits no probability/win-rate (Rule 11.13 /
8.3 / 9.4). Every field is verifiable price data. The price source is injected (Rule 2.1
provenance) so the logic is deterministic under test; the default is a best-effort
yfinance reader that degrades to empty offline.
"""
from __future__ import annotations

from typing import Any, Callable

from hot_theme_rotator.candidate_engine.tradability import tradability as _tradability

# Representative LIQUID JP names per theme. A seed map (expandable) — not a recommendation
# list; it answers "who is exposed", which the owner's judgment then acts on.
EXPOSURE_MAP: dict[str, tuple[tuple[str, str], ...]] = {
    "semi": (("8035.T", "東京エレクトロン"), ("6857.T", "アドバンテスト"), ("6146.T", "ディスコ"), ("6920.T", "レーザーテック")),
    "ai": (("9984.T", "ソフトバンクG"), ("6098.T", "リクルート"), ("6758.T", "ソニーG")),
    "auto": (("7203.T", "トヨタ"), ("7267.T", "ホンダ"), ("7201.T", "日産"), ("7269.T", "スズキ")),
    "bank": (("8306.T", "三菱UFJ"), ("8316.T", "三井住友FG"), ("8411.T", "みずほ"), ("8604.T", "野村")),
    "defense": (("7011.T", "三菱重工"), ("7013.T", "IHI"), ("7012.T", "川崎重工")),
    "energy": (("1605.T", "INPEX"), ("5020.T", "ENEOS"), ("5021.T", "コスモ"), ("1662.T", "石油資源開発")),
    "optical": (("5803.T", "フジクラ"), ("5801.T", "古河電工"), ("5802.T", "住友電工")),
    "memory": (("285A.T", "キオクシア"), ("6857.T", "アドバンテスト"), ("3436.T", "SUMCO")),
}

# Event keyword → themes it moves. The owner's named events map to the war-premium /
# exporter baskets. (Direction is the owner's call — this only surfaces WHO is exposed.)
EVENT_THEMES: dict[str, tuple[str, ...]] = {
    "geopolitics": ("defense", "energy"),
    "ceasefire": ("defense", "energy"),
    "停戦": ("defense", "energy"),
    "fx": ("auto", "semi"),
    "円安": ("auto", "semi"),
    "yen_weak": ("auto", "semi"),
}

BENCHMARK_TICKER = "1306.T"  # TOPIX ETF — the passive baseline the owner already holds.

PriceFetcher = Callable[[list[str]], dict[str, list[float]]]


def _ret(closes: list[float], n: int) -> float | None:
    """Simple return over the last n sessions (close[-1]/close[-1-n] - 1)."""
    if not closes or len(closes) <= n:
        return None
    prev = closes[-1 - n]
    if not prev:
        return None
    return closes[-1] / prev - 1.0


def _freshness(ret20: float | None, ret5: float | None) -> str:
    """Descriptive label for how far a name has ALREADY moved — NOT a buy/sell call."""
    if ret20 is None:
        return "unknown"
    if ret20 >= 0.15 and (ret5 is not None and ret5 <= -0.05):
        return "rolling_over"   # 涨过一大段、近5日在回吐
    if ret20 >= 0.25:
        return "extended"       # 已大涨、过热
    if ret20 <= -0.10:
        return "falling"        # 在下行
    return "fresh"              # 未过热


def themes_for_event(event: str) -> list[str]:
    """Resolve an event/theme string to the theme baskets it touches."""
    key = (event or "").strip().lower()
    if key in EXPOSURE_MAP:
        return [key]
    for alias, themes in EVENT_THEMES.items():
        if alias.lower() in key:
            return list(themes)
    return []


def priced_in_read(
    tickers: list[str],
    *,
    price_fetcher: PriceFetcher,
    benchmark_ticker: str = BENCHMARK_TICKER,
    account_jpy: float = 400_000.0,
) -> list[dict[str, Any]]:
    """For each ticker: last close, 1d/5d/20d return, excess vs benchmark (5d),
    distance from the 20-session high, and a descriptive freshness label.

    Fail-open: a ticker with no/short price history yields nulls + label 'unknown',
    never an exception (Rule 11.9). All values are verifiable price facts."""
    want = list(dict.fromkeys([*tickers, benchmark_ticker]))
    try:
        closes = price_fetcher(want) or {}
    except Exception:  # noqa: BLE001 — price source is best-effort; never break the desk
        closes = {}

    bench = closes.get(benchmark_ticker) or []
    bench_5d = _ret(bench, 5)

    out: list[dict[str, Any]] = []
    for t in tickers:
        s = closes.get(t) or []
        ret5 = _ret(s, 5)
        ret20 = _ret(s, 20)
        window = s[-20:] if len(s) >= 1 else []
        hi = max(window) if window else None
        out.append({
            "symbol": t,
            "last": round(s[-1], 2) if s else None,
            "ret1d": round(_ret(s, 1), 4) if _ret(s, 1) is not None else None,
            "ret5d": round(ret5, 4) if ret5 is not None else None,
            "ret20d": round(ret20, 4) if ret20 is not None else None,
            "excess5dVsBenchmark": round(ret5 - bench_5d, 4) if (ret5 is not None and bench_5d is not None) else None,
            "distFromHigh20d": round(s[-1] / hi - 1.0, 4) if (s and hi) else None,
            "freshness": _freshness(ret20, ret5),
            # ADR-0010 / Rule 5.1 — execution gate: is this even tradable at the
            # account size, net of JPX tick cost? (deterministic, no prediction)
            "tradability": _tradability(s[-1], account_jpy, require_adv=False) if s else None,
        })
    return out


def build_event_desk(
    event: str,
    *,
    price_fetcher: PriceFetcher,
    benchmark_ticker: str = BENCHMARK_TICKER,
    account_jpy: float = 400_000.0,
) -> dict[str, Any]:
    """Top-level: event/theme string → exposed names with their priced-in read.

    Honest by construction (Rule 11.13): no event-outcome probability, exposure is a
    data join (not a recommendation), every number is verifiable price action."""
    themes = themes_for_event(event)
    names: list[tuple[str, str]] = []
    seen: set[str] = set()
    for th in themes:
        for sym, name in EXPOSURE_MAP.get(th, ()):  # preserve order, dedupe
            if sym not in seen:
                seen.add(sym)
                names.append((sym, name))
    read = priced_in_read([s for s, _ in names], price_fetcher=price_fetcher, benchmark_ticker=benchmark_ticker, account_jpy=account_jpy)
    name_by_sym = dict(names)
    for row in read:
        row["name"] = name_by_sym.get(row["symbol"], "")
    return {
        "event": event,
        "themes": themes,
        "benchmark": benchmark_ticker,
        "exposure": read,
        "disclosure": "事件作战台:只显示受影响标的与其已发生的价格变动(可复核),不预测事件结果、不给上涨概率/胜率。方向判断是你的(Rule 11.13)。",
    }


def yfinance_price_fetcher(tickers: list[str], *, sessions: int = 25) -> dict[str, list[float]]:
    """Best-effort live close history via yfinance (auto_adjust=False, Rule 11.9.6
    same basis). Returns {} on any failure so the desk degrades gracefully offline."""
    try:
        import yfinance as yf  # local import — optional dependency at call time
        df = yf.download(tickers, period=f"{max(sessions, 30)}d", auto_adjust=False, progress=False)["Close"]
        out: dict[str, list[float]] = {}
        for t in tickers:
            try:
                col = df[t] if len(tickers) > 1 else df
                vals = [float(x) for x in col.dropna().tolist()]
                if vals:
                    out[t] = vals[-sessions:]
            except Exception:  # noqa: BLE001
                continue
        return out
    except Exception:  # noqa: BLE001
        return {}
