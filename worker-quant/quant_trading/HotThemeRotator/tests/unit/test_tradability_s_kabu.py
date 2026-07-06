"""Tests for S株 (fractional) execution mode in the tradability gate (Rule 5.2)."""
import pytest

from hot_theme_rotator.candidate_engine.tradability import (
    s_kabu_affordable,
    s_kabu_round_trip_cost_bps,
    s_kabu_tradability,
    tradability,
)


# ── cost model ───────────────────────────────────────────────────────────────


def test_s_kabu_cost_is_timing_buffer_only_default_5bps():
    assert s_kabu_round_trip_cost_bps() == 5.0


def test_s_kabu_cost_is_price_independent():
    assert s_kabu_round_trip_cost_bps(timing_buffer_bps=3.0) == 3.0


def test_s_kabu_cost_rejects_negative():
    with pytest.raises(ValueError):
        s_kabu_round_trip_cost_bps(timing_buffer_bps=-1.0)


# ── affordability (1-share unit) ─────────────────────────────────────────────


def test_s_kabu_affordable_expensive_name_fits_one_share():
    # 8035.T @ ¥77,600 on ¥400k: 1 share = 19.4% < 34% cap → affordable
    a = s_kabu_affordable(77_600.0, 400_000.0)
    assert a["affordable"] is True
    assert a["sharesAffordable"] == 1  # floor(136000 / 77600)
    assert a["positionFrac"] == pytest.approx(0.194, abs=1e-3)
    assert a["concentrationWarn"] is False  # 19.4% < 20% band


def test_s_kabu_affordable_flags_concentration_over_20pct():
    # ¥90,000 on ¥400k: 1 share = 22.5% > 20% warn band, still < 34% cap
    a = s_kabu_affordable(90_000.0, 400_000.0)
    assert a["affordable"] is True
    assert a["concentrationWarn"] is True


def test_s_kabu_affordable_rejects_when_one_share_exceeds_cap():
    # ¥150,000 on ¥400k: 1 share = 37.5% > 34% cap → not affordable
    a = s_kabu_affordable(150_000.0, 400_000.0)
    assert a["affordable"] is False


# ── the headline: lot-untradable but S株-tradable ────────────────────────────


def test_expensive_name_lot_untradable_but_s_kabu_tradable():
    px = 77_600.0  # 8035.T Tokyo Electron
    lot = tradability(px, 400_000.0, require_adv=False)
    sk = s_kabu_tradability(px, 400_000.0, require_adv=False)
    assert lot["tradable"] is False          # 100-share lot = ¥7.76M, can't afford
    assert any("lot" in r.lower() for r in lot["reasons"])
    assert sk["tradable"] is True            # 1 share fits → S株 sees it
    assert sk["sharesAffordable"] == 1


def test_s_kabu_tradability_adv_fails_closed_when_unverified():
    sk = s_kabu_tradability(77_600.0, 400_000.0, require_adv=True)  # no adv_jpy
    assert sk["tradable"] is False
    assert any("ADV unverified" in r for r in sk["reasons"])


def test_s_kabu_tradability_2x_cost_stress_with_tiny_cost():
    # cost ~5bps; a 0.5% gross easily survives 2x cost (20bps)
    sk = s_kabu_tradability(77_600.0, 400_000.0, expected_gross=0.005, require_adv=False)
    assert sk["survives2xCost"] is True
    assert sk["netAfterCost"] == pytest.approx(0.005 - 0.0005, abs=1e-6)


def test_s_kabu_tradability_invalid_price():
    sk = s_kabu_tradability(0.0)
    assert sk["tradable"] is False
    assert sk["execMode"] == "s_kabu"
