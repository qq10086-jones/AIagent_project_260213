"""Tests for the S株 universe overlay (Rule 5.2 / task #5)."""
from hot_theme_rotator.candidate_engine.s_kabu_universe import s_kabu_overlay_rows


def _lookup(prices):
    return lambda sym: prices.get(sym)


def test_overlay_includes_lot_unaffordable_expensive_name():
    # 8035.T @ ¥77,600: lot-untradable (¥7.76M/lot) but S株-tradable (1 share)
    rows = s_kabu_overlay_rows(["8035.T"], account_jpy=400_000.0, price_lookup=_lookup({"8035.T": 77_600.0}))
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "8035.T"
    assert r["execMode"] == "s_kabu"
    assert r["source"] == "s_kabu_overlay"
    assert r["sharesAffordable"] == 1
    assert r["lotTradable"] is False


def test_overlay_skips_lot_affordable_cheap_name_by_default():
    # ¥1,000 name: a 100-lot (¥100k) fits a ¥400k account → lot handles it → no overlay
    rows = s_kabu_overlay_rows(["1234.T"], account_jpy=400_000.0, price_lookup=_lookup({"1234.T": 1_000.0}))
    assert rows == []


def test_overlay_can_include_all_when_flag_off():
    rows = s_kabu_overlay_rows(
        ["1234.T"], account_jpy=400_000.0,
        price_lookup=_lookup({"1234.T": 1_000.0}), only_lot_unaffordable=False,
    )
    assert len(rows) == 1
    assert rows[0]["execMode"] == "s_kabu"


def test_overlay_skips_unpriceable_name():
    rows = s_kabu_overlay_rows(["9999.T"], account_jpy=400_000.0, price_lookup=_lookup({}))
    assert rows == []


def test_overlay_skips_when_one_share_over_position_cap():
    # ¥150,000 on ¥400k: 1 share = 37.5% > 34% cap → S株 can't help → skipped
    rows = s_kabu_overlay_rows(["8888.T"], account_jpy=400_000.0, price_lookup=_lookup({"8888.T": 150_000.0}))
    assert rows == []


def test_overlay_flags_concentration_warning():
    # ¥90,000 on ¥400k: 1 share = 22.5% (>20% warn, <34% cap) → included with warn
    rows = s_kabu_overlay_rows(["7777.T"], account_jpy=400_000.0, price_lookup=_lookup({"7777.T": 90_000.0}))
    assert len(rows) == 1
    assert rows[0]["concentrationWarn"] is True
