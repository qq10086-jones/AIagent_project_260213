"""Tests for the TDnet revision-document parser (P23-A, Lane A).

Contracts: parse 前回発表予想(A)/今回修正予想(B) rows into per-metric
{before, after, pct}; handle 未定 (undetermined), △/▲ negatives, comma
grouping; compute pct only when both sides are numeric and A != 0; never
fabricate a magnitude.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.tdnet_revision_docs import (  # noqa: E402
    parse_revision_text,
)

SAMPLE = """
2027年3月期の連結業績予想の修正
 売上高 営業利益 経常利益 親会社株主に帰属する当期純利益 1株当たり当期純利益
前回発表予想（Ａ） 150,000 16,000 16,500 10,000 201円50銭
今回修正予想（Ｂ） 160,000 17,500 18,000 11,000 221円71銭
増減額（Ｂ－Ａ） 10,000 1,500 1,500 1,000
増減率（％） 6.7 9.4 9.1 10.0
"""

SAMPLE_MITEI = """
 売上高 営業利益 経常利益 当期純利益
前回発表予想（Ａ） 未定 未定 未定 未定
今回修正予想（Ｂ） 160,000 17,500 18,000 11,000
"""

SAMPLE_NEGATIVE = """
 売上高 営業利益 経常利益 当期純利益
前回発表予想（Ａ） 50,000 1,200 1,100 800
今回修正予想（Ｂ） 48,000 △300 △400 △1,000
"""


def test_parse_standard_revision_table():
    out = parse_revision_text(SAMPLE)
    assert out["parsed"] is True
    rev = out["metrics"]["revenue"]
    assert rev["before"] == 150000 and rev["after"] == 160000
    assert abs(rev["pct"] - (160000 / 150000 - 1.0)) < 1e-9
    op = out["metrics"]["operating_income"]
    assert op["before"] == 16000 and op["after"] == 17500
    ni = out["metrics"]["net_income"]
    assert ni["after"] == 11000


def test_mitei_before_yields_no_pct():
    out = parse_revision_text(SAMPLE_MITEI)
    assert out["parsed"] is True
    rev = out["metrics"]["revenue"]
    assert rev["before"] is None
    assert rev["after"] == 160000
    assert rev["pct"] is None  # 未定 → no magnitude, never fabricated


def test_triangle_negatives():
    out = parse_revision_text(SAMPLE_NEGATIVE)
    op = out["metrics"]["operating_income"]
    assert op["after"] == -300
    assert abs(op["pct"] - (-300 - 1200) / 1200) < 1e-9  # -1.25
    ni = out["metrics"]["net_income"]
    assert ni["after"] == -1000


SAMPLE_NEGATIVE_BASE = """
 売上高 営業利益 経常利益 当期純利益
前回発表予想（Ａ） 50,000 △294 △300 △200
今回修正予想（Ｂ） 50,000 △495 △500 △400
"""


def test_widening_loss_reads_as_negative_surprise():
    out = parse_revision_text(SAMPLE_NEGATIVE_BASE)
    op = out["metrics"]["operating_income"]
    # (−495 − (−294)) / |−294| = −0.6837… — NOT +68% (the b/a−1 sign trap)
    assert op["pct"] < 0
    assert abs(op["pct"] - (-495 + 294) / 294) < 1e-9


# --- correctness review 2026-07-06: parser robustness (findings 2/3/7) -------

SAMPLE_BANK = """
 経常収益 経常利益 親会社株主に帰属する当期純利益
前回発表予想（Ａ） 100,000 20,000 12,000
今回修正予想（Ｂ） 90,000 18,000 10,800
"""

SAMPLE_PROSE_HEADER = """
売上高、営業利益、経常利益及び当期純利益の予想を修正いたします。
 営業利益 経常利益
前回発表予想（Ａ） 3,100 3,100
今回修正予想（Ｂ） 2,700 2,700
"""

SAMPLE_SANKO_ACTUALS = """
 売上高 営業利益 経常利益 当期純利益
前回発表予想（Ａ）
（参考）前期実績 20,000 800 750 500
今回修正予想（Ｂ） 35,000 1,900 1,700 1,200
"""


def test_bank_leading_column_maps_metrics_correctly():
    out = parse_revision_text(SAMPLE_BANK)
    # 経常収益 is revenue (not mis-mapped to ordinary_income)
    assert out["metrics"]["revenue"]["before"] == 100000
    assert out["metrics"]["ordinary_income"]["before"] == 20000
    assert out["metrics"]["net_income"]["before"] == 12000
    # no fabricated surprise on a shifted column
    assert abs(out["metrics"]["revenue"]["pct"] - (90000 / 100000 - 1)) < 1e-9


def test_prose_line_does_not_hijack_header():
    out = parse_revision_text(SAMPLE_PROSE_HEADER)
    # only 営業利益/経常利益 were revised — revenue/net must be ABSENT, not −10%
    assert set(out["metrics"]) == {"operating_income", "ordinary_income"}
    assert out["metrics"]["operating_income"]["after"] == 2700


def test_sanko_actuals_not_adopted_as_prior_forecast():
    out = parse_revision_text(SAMPLE_SANKO_ACTUALS)
    # A-row is genuinely blank (values wrapped onto a 前期実績 reference line
    # that MUST be ignored) → A stays None, no fabricated revision
    rev = out["metrics"]["revenue"]
    assert rev["before"] is None
    assert rev["after"] == 35000
    assert rev["pct"] is None


def test_unparseable_text_is_honest():
    out = parse_revision_text("本日、代表取締役の異動についてお知らせします。")
    assert out["parsed"] is False
    assert out["metrics"] == {}


SAMPLE_SPACED_HEADER = """
 売 上 高 営 業 利 益 経 常 利 益 当 期 純 利 益
 百万円 百万円 百万円 百万円
前回発表予想（Ａ） 1,850 3 3 △58
今回修正予想（Ｂ） 1,850 3 3 441
"""

SAMPLE_WRAPPED_VALUES = """
 売 上 高 営 業 利 益 経 常 利 益 に帰属する
 当期純利益
 百万円 百万円 百万円 百万円
前回発表予想（Ａ）
 34,000 1,350 1,300 870 168.91
今回修正予想（Ｂ） 35,363 1,878 1,694 1,176 228.33
"""

SAMPLE_HAPPYO_B = """
 売 上 高 営 業 利 益 経 常 利 益 に帰属する
前回発表予想 （Ａ） 31,000 3,100 3,100 2,000 287円37銭
今回発表予想 （Ｂ） 30,000 2,700 2,700 3,000 431円05銭
"""


def test_letter_spaced_header_parses():
    out = parse_revision_text(SAMPLE_SPACED_HEADER)
    assert out["parsed"] is True
    assert out["metrics"]["net_income"]["before"] == -58
    assert out["metrics"]["net_income"]["after"] == 441


def test_values_wrapped_to_next_line():
    out = parse_revision_text(SAMPLE_WRAPPED_VALUES)
    assert out["parsed"] is True
    op = out["metrics"]["operating_income"]
    assert op["before"] == 1350 and op["after"] == 1878
    assert abs(op["pct"] - (1878 / 1350 - 1.0)) < 1e-9


def test_konkai_happyo_b_row_variant():
    out = parse_revision_text(SAMPLE_HAPPYO_B)
    assert out["parsed"] is True
    rev = out["metrics"]["revenue"]
    assert rev["before"] == 31000 and rev["after"] == 30000
    assert abs(rev["pct"] - (30000 / 31000 - 1.0)) < 1e-9
