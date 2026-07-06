"""Disclosure-surprise + novelty signal (ADR-0010 P17-2)."""
from __future__ import annotations

from hot_theme_rotator.candidate_engine.disclosure_surprise import (
    rank_disclosures,
    surprise_signal,
)


def test_upward_guidance_is_positive_material_novel():
    s = surprise_signal("2026年3月期 通期業績予想の上方修正に関するお知らせ")
    assert s["category"] == "earnings" and s["material"] is True
    assert s["direction"] == 1 and s["novel"] is True
    assert s["surpriseScore"] == 1.0


def test_downward_guidance_is_negative():
    s = surprise_signal("通期業績予想の下方修正に関するお知らせ")
    assert s["direction"] == -1 and s["material"] is True
    assert s["surpriseScore"] == -1.0


def test_dividend_increase_positive():
    s = surprise_signal("剰余金の配当（増配）に関するお知らせ")
    assert s["category"] == "dividend" and s["direction"] == 1 and s["surpriseScore"] == 1.0


def test_correction_is_stale_and_dampened():
    # 訂正 = amendment → not novel → score heavily dampened even if direction is up.
    s = surprise_signal("（訂正）通期業績予想の上方修正に関するお知らせの一部訂正")
    assert s["novel"] is False
    assert s["direction"] == 1
    assert abs(s["surpriseScore"]) < 0.5   # 1 * 1.0(material) * 0.2(stale) = 0.2


def test_immaterial_disclosure_low_score():
    # A non-earnings/dividend disclosure with no surprise word → 0.
    s = surprise_signal("本社移転に関するお知らせ")
    assert s["direction"] == 0 and s["surpriseScore"] == 0.0


def test_empty_title_is_safe():
    s = surprise_signal("")
    assert s["surpriseScore"] == 0.0 and s["direction"] == 0


# ── Codex-driven fixes (2026-06-17) ──

def test_mixed_signal_is_ambiguous_not_up():
    # Both up- and down-words present → must NOT default to +1; direction 0 + flagged.
    s = surprise_signal("売上高は上方修正だが営業利益は下方修正")
    assert s["mixed"] is True and s["direction"] == 0 and s["surpriseScore"] == 0.0


def test_other_category_does_not_block_reclassification():
    # A caller-supplied "other" must not suppress a clear earnings-title match.
    s = surprise_signal("通期業績予想の上方修正に関するお知らせ", category="other")
    assert s["category"] == "earnings" and s["material"] is True and s["direction"] == 1


def test_rank_flags_pit_timestamp_presence():
    rows = [
        {"ticker": "1.T", "title": "通期業績予想の上方修正", "published_ts": "2026-06-17T15:00:00+09:00"},
        {"ticker": "2.T", "title": "通期業績予想の上方修正"},  # no PIT timestamp
    ]
    ranked = rank_disclosures(rows)
    by = {r["ticker"]: r["pitOk"] for r in ranked}
    assert by["1.T"] is True and by["2.T"] is False


def test_rank_tolerates_non_dict_entries():
    ranked = rank_disclosures([{"ticker": "1.T", "title": "増配"}, "garbage", None])
    assert len(ranked) == 1 and ranked[0]["ticker"] == "1.T"


def test_rank_orders_by_absolute_surprise():
    rows = [
        {"ticker": "1.T", "title": "本社移転に関するお知らせ"},                       # 0
        {"ticker": "2.T", "title": "通期業績予想の上方修正に関するお知らせ"},          # +1
        {"ticker": "3.T", "title": "（訂正）業績予想の下方修正の一部訂正"},            # -0.2 stale
    ]
    ranked = rank_disclosures(rows)
    assert ranked[0]["ticker"] == "2.T"          # strongest |score| first
    assert ranked[-1]["ticker"] == "1.T"         # non-event last
    assert len(ranked) == 3                       # nothing dropped
