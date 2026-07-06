"""Unit tests for the forward-test harness loader (ADR-0011 / §16, P19-01 inc-2)."""
from statistics import median

import pytest

from hot_theme_rotator.backtesting.forward_eval_data import (
    estimate_round_trip_cost_frac,
    group_live_daily,
    load_live_panels,
)
from hot_theme_rotator.candidate_engine.tradability import round_trip_cost_bps


class _Pred:
    def __init__(self, pid, buy, date, live=True, price=None):
        self.prediction_id = pid
        self.buy = buy
        self.trade_date = date
        self.extra = {"reference_price": price} if price is not None else {}
        self._live = live


class _Out:
    def __init__(self, pid, realized_returns):
        self.prediction_id = pid
        self.realized_returns = realized_returns


def _is_live(p):
    return p._live


# ── group_live_daily ─────────────────────────────────────────────────────────


def test_group_live_daily_basic_panel():
    preds = [_Pred(f"d1-{i}", buy=i / 10, date="2026-06-01", price=1000 + i) for i in range(5)]
    outs = {f"d1-{i}": _Out(f"d1-{i}", {"3D": i / 100}) for i in range(5)}
    panels = group_live_daily(preds, outs, horizon=3, is_live=_is_live, min_names=5)
    assert len(panels) == 1
    p = panels[0]
    assert p.date == "2026-06-01"
    assert p.scores == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert p.returns == [0.0, 0.01, 0.02, 0.03, 0.04]
    assert p.prices == [1000, 1001, 1002, 1003, 1004]


def test_group_live_daily_filters_nonlive_unjoined_and_missing_horizon():
    preds = [_Pred(f"x{i}", buy=0.5, date="2026-06-02", price=1000) for i in range(5)]
    preds.append(_Pred("nonlive", 0.9, "2026-06-02", live=False, price=1000))  # skipped: not live
    preds.append(_Pred("noout", 0.9, "2026-06-02", price=1000))               # skipped: no outcome
    preds.append(_Pred("no3d", 0.9, "2026-06-02", price=1000))                # skipped: missing 3D
    outs = {f"x{i}": _Out(f"x{i}", {"3D": 0.01}) for i in range(5)}
    outs["no3d"] = _Out("no3d", {"1D": 0.01})  # has 1D but not 3D
    panels = group_live_daily(preds, outs, horizon=3, is_live=_is_live, min_names=5)
    assert len(panels) == 1
    assert len(panels[0].scores) == 5  # only the 5 valid x-names


def test_group_live_daily_skips_days_below_min_names():
    preds = [_Pred(f"s{i}", 0.5, "2026-06-03", price=1000) for i in range(3)]
    outs = {f"s{i}": _Out(f"s{i}", {"3D": 0.01}) for i in range(3)}
    panels = group_live_daily(preds, outs, horizon=3, is_live=_is_live, min_names=5)
    assert panels == []


# ── estimate_round_trip_cost_frac ────────────────────────────────────────────


def test_estimate_round_trip_cost_frac_matches_tradability_median():
    prices = [1000.0, 2000.0]
    expected = median(
        [round_trip_cost_bps(1000.0), round_trip_cost_bps(2000.0)]
    ) / 1e4
    assert estimate_round_trip_cost_frac(prices) == pytest.approx(expected)


def test_estimate_round_trip_cost_frac_ignores_none_and_nonpositive():
    prices = [None, 0.0, -5.0, 1500.0]
    assert estimate_round_trip_cost_frac(prices) == pytest.approx(
        round_trip_cost_bps(1500.0) / 1e4
    )


def test_estimate_round_trip_cost_frac_empty_is_none():
    assert estimate_round_trip_cost_frac([None, 0.0]) is None


# ── load_live_panels IO guard ────────────────────────────────────────────────


def test_load_live_panels_missing_journals_returns_empty(tmp_path):
    assert load_live_panels(tmp_path, horizon=3) == []
