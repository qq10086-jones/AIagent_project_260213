"""Execution / tradability gate (ADR-0010, Rule 5.1)."""
from __future__ import annotations

import math

import pytest

from hot_theme_rotator.candidate_engine.tradability import (
    jpx_tick,
    lots_affordable,
    net_expected_after_cost,
    round_trip_cost_bps,
    tradability,
)


def test_jpx_tick_ladder():
    assert jpx_tick(300) == 1.0
    assert jpx_tick(3_000) == 1.0
    assert jpx_tick(3_001) == 5.0
    assert jpx_tick(4_999) == 5.0
    assert jpx_tick(20_000) == 10.0
    assert jpx_tick(400_000) == 1_000.0


def test_round_trip_cost_rises_at_cheap_prices():
    # The paradox: the cheapest names carry the worst tick cost.
    cheap = round_trip_cost_bps(300, spread_ticks=2)   # 2 * (1/300) * 1e4 = 66.7
    mid = round_trip_cost_bps(1_000, spread_ticks=2)   # 20.0
    assert math.isclose(cheap, 66.67, abs_tol=0.5)
    assert math.isclose(mid, 20.0, abs_tol=0.1)
    assert cheap > mid
    # slippage adds on top
    assert round_trip_cost_bps(1_000, spread_ticks=2, slippage_bps=10) == 30.0


def test_lots_affordable_and_diversification_cap():
    acct = 400_000
    # ¥300 name: lot ¥30k = 7.5% → diversifiable, several lots within 34% cap
    cheap = lots_affordable(300, acct)
    assert cheap["diversifiable"] and cheap["lots"] >= 1
    # ¥2000 name: lot ¥200k = 50% > 34% cap → not diversifiable
    rich = lots_affordable(2_000, acct)
    assert not rich["diversifiable"]


def test_net_expected_after_cost():
    # 3% gross on a ¥1000 name, 2-tick spread + 10bps slip = 30bps cost → ~2.7% net
    net = net_expected_after_cost(0.03, 1_000, spread_ticks=2, slippage_bps=10)
    assert math.isclose(net, 0.027, abs_tol=0.0005)


def test_tradability_sweet_spot_passes():
    # ~¥1000, decent ADV, 3% expected gross → tradable in the sweet spot.
    v = tradability(1_000, 400_000, adv_jpy=200_000_000, expected_gross=0.03)
    assert v["tradable"] is True and v["reasons"] == []
    assert v["survives2xCost"] is True
    assert v["netAfterCost"] is not None and v["netAfterCost"] > 0


def test_tradability_cheap_name_killed_by_cost():
    # ¥300 name: round-trip ~67bps > 60bps cap → not tradable (cost gate).
    v = tradability(300, 400_000, expected_gross=0.03)
    assert v["tradable"] is False
    assert any("round-trip" in r for r in v["reasons"])


def test_tradability_expensive_name_killed_by_lot_size():
    # ¥3000 name: one lot ¥300k = 75% of a ¥400k account → can't diversify.
    v = tradability(3_000, 400_000, expected_gross=0.03)
    assert v["tradable"] is False
    assert any("account" in r or "cap" in r for r in v["reasons"])


def test_tradability_fails_2x_cost_stress():
    # Thin 0.4% gross on a ¥1000 name can't survive 2x of ~20-30bps cost.
    v = tradability(1_000, 400_000, expected_gross=0.004)
    assert v["survives2xCost"] is False
    assert v["tradable"] is False


def test_tradability_adv_floor():
    v = tradability(1_000, 400_000, adv_jpy=5_000_000, expected_gross=0.03)  # ¥5M ADV < ¥50M
    assert v["advOk"] is False and v["tradable"] is False
    assert any("ADV" in r for r in v["reasons"])


# ── Codex-driven fixes (2026-06-17) ──

def test_adv_unverified_fails_closed_by_default():
    # No ADV + require_adv default True → liquidity gate fails closed (no silent pass).
    v = tradability(1_000, 400_000, expected_gross=0.03)
    assert v["advVerified"] is False and v["tradable"] is False
    assert any("ADV unverified" in r for r in v["reasons"])


def test_require_adv_false_gives_structural_verdict():
    # Callers without volume data (e.g. Event Desk) opt out → cost+lot verdict, ADV flagged.
    v = tradability(1_000, 400_000, expected_gross=0.03, require_adv=False)
    assert v["advVerified"] is False and v["tradable"] is True  # passes cost+lot
    assert not any("ADV" in r for r in v["reasons"])


def test_invalid_price_is_rejected():
    v = tradability(0, 400_000)
    assert v["tradable"] is False and any("invalid price" in r for r in v["reasons"])


def test_negative_cost_params_raise():
    with pytest.raises(ValueError):
        round_trip_cost_bps(1_000, spread_ticks=-1)
    with pytest.raises(ValueError):
        round_trip_cost_bps(1_000, slippage_bps=-5)
