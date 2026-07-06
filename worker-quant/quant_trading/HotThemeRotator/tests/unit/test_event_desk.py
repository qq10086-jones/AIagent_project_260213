"""Event Desk core — exposure map + priced-in read (P16-E2/E3, Rule 11.13)."""
from __future__ import annotations

from hot_theme_rotator.candidate_engine.event_desk import (
    EXPOSURE_MAP,
    build_event_desk,
    priced_in_read,
    themes_for_event,
)


def _series(start: float, steps: list[float]) -> list[float]:
    """Build a close series from per-step pct moves (e.g. [0.0,0.0,...])."""
    s = [start]
    for r in steps:
        s.append(s[-1] * (1 + r))
    return s


def _flat(symbols, n=25, level=100.0):
    return {s: [level] * n for s in symbols}


def test_themes_for_event_resolves_themes_and_event_aliases():
    assert themes_for_event("semi") == ["semi"]
    # a ceasefire maps to the war-premium baskets (defense + energy)
    assert themes_for_event("イランと米国が停戦") == ["defense", "energy"]
    assert themes_for_event("ceasefire in the middle east") == ["defense", "energy"]
    # weak-yen maps to exporters
    assert themes_for_event("円安が160円突破") == ["auto", "semi"]
    # unknown → empty (no fabrication)
    assert themes_for_event("全然関係ない話題") == []


def test_priced_in_read_computes_returns_and_excess():
    # SEMI ran +25% over 20 sessions; benchmark flat → big positive excess.
    up = _series(100.0, [0.0113] * 20)            # ~+25% over 20 steps
    fetch = lambda ts: {"8035.T": up, "1306.T": [100.0] * 21}
    rows = priced_in_read(["8035.T"], price_fetcher=fetch, benchmark_ticker="1306.T")
    r = rows[0]
    assert r["symbol"] == "8035.T"
    assert r["ret20d"] is not None and r["ret20d"] > 0.20
    assert r["excess5dVsBenchmark"] is not None and r["excess5dVsBenchmark"] > 0
    assert r["distFromHigh20d"] is not None and r["distFromHigh20d"] <= 0  # at/below its high


def test_freshness_labels_are_descriptive():
    # extended: +30% over 20d, still rising
    ext = _series(100.0, [0.013] * 20)
    # rolling_over: ran +20% then last 5 sessions down hard
    roll = _series(100.0, [0.02] * 15 + [-0.02] * 5)
    # falling: down 15%
    fall = _series(100.0, [-0.008] * 20)
    # fresh: roughly flat
    fresh = _series(100.0, [0.001] * 20)
    fetch = lambda ts: {"EXT": ext, "ROLL": roll, "FALL": fall, "FRESH": fresh, "1306.T": [100.0] * 21}
    by = {r["symbol"]: r["freshness"] for r in priced_in_read(["EXT", "ROLL", "FALL", "FRESH"], price_fetcher=fetch)}
    assert by["EXT"] == "extended"
    assert by["ROLL"] == "rolling_over"
    assert by["FALL"] == "falling"
    assert by["FRESH"] == "fresh"


def test_priced_in_read_fails_open_on_missing_or_short_data():
    # No data for the ticker, fetcher returns empty → nulls + 'unknown', no exception.
    rows = priced_in_read(["NODATA.T"], price_fetcher=lambda ts: {})
    assert rows[0]["last"] is None and rows[0]["freshness"] == "unknown"

    # A fetcher that raises must not break the desk.
    def boom(ts):
        raise RuntimeError("network down")

    rows = priced_in_read(["X.T"], price_fetcher=boom)
    assert rows[0]["freshness"] == "unknown"


def test_build_event_desk_joins_exposure_and_is_honest():
    fetch = _flat([s for th in ("defense", "energy") for s, _ in EXPOSURE_MAP[th]] + ["1306.T"])
    desk = build_event_desk("イラン停戦合意", price_fetcher=fetch)
    assert desk["themes"] == ["defense", "energy"]
    syms = {r["symbol"] for r in desk["exposure"]}
    assert "7011.T" in syms and "1605.T" in syms        # 三菱重工 + INPEX exposed
    assert all(r.get("name") for r in desk["exposure"])  # names joined
    # Rule 11.13 honesty: no probability/win-rate, standing disclosure present.
    blob = str(desk).lower()
    assert "probability" not in blob and "win" not in blob
    assert "胜率" in desk["disclosure"] and "不预测" in desk["disclosure"]


def test_build_event_desk_unknown_event_yields_no_exposure():
    desk = build_event_desk("无关事件", price_fetcher=lambda ts: {})
    assert desk["themes"] == [] and desk["exposure"] == []
