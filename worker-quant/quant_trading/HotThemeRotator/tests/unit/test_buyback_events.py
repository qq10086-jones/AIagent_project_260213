"""P34-01a tests — buyback subtype classification and structured extraction.

Titles below are verbatim or near-verbatim from the stored TDnet corpus
(reports/tdnet/*.jsonl, 2026-06-30..2026-08-07).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.buyback_events import (  # noqa: E402
    BuybackParseError,
    classify_buyback_subtype,
    compute_event_id,
    corpus_summary,
    is_buyback_related,
    link_execution_reports,
    parse_buyback_event,
)
from hot_theme_rotator.data.external.tdnet_parser import classify_category  # noqa: E402


def _disc(title, ticker="5133.T", published="2026-07-15T15:00:00+09:00"):
    return {
        "disclosure_id": "a" * 32,
        "ticker": ticker,
        "published_ts": published,
        "collected_ts": "2026-07-15T15:05:00+09:00",
        "title": title,
        "category": "buyback",
        "url": "https://example.invalid/x.pdf",
    }


# --- the defect being fixed -------------------------------------------------

def test_resolution_no_longer_classifies_as_order():
    """The `order` rule matches 株式の取得, which 自己株式の取得 contains."""
    title = "自己株式の取得に係る事項の決定に関するお知らせ"
    assert classify_category(title) == "buyback"


def test_execution_report_no_longer_classifies_as_order():
    assert classify_category("自己株式の取得状況に関するお知らせ") == "buyback"


def test_combined_buyback_and_dividend_prefers_buyback():
    title = "自己株式取得及び剰余金の配当に関するお知らせ"
    assert classify_category(title) == "buyback"


def test_non_treasury_titles_are_unaffected():
    assert classify_category("公開買付けの開始に関するお知らせ") == "tob"
    assert classify_category("2027年3月期第1四半期決算短信") == "earnings"
    assert classify_category("剰余金の配当に関するお知らせ") == "dividend"
    assert classify_category("株式分割に関するお知らせ") == "split"


# --- subtype taxonomy -------------------------------------------------------

@pytest.mark.parametrize(
    "title,expected",
    [
        ("自己株式取得に係る事項の決定に関するお知らせ", "resolution"),
        ("自己株式の取得に係る事項の決定に関するお知らせ", "resolution"),
        ("自己株式の取得状況に関するお知らせ", "execution_report"),
        ("自己株式の消却に関するお知らせ", "cancellation"),
        ("自己株式の処分に関するお知らせ", "disposal"),
        ("自己株式の取得の終了に関するお知らせ", "completion"),
        ("（開示事項の変更）自己株式の取得に係る事項の決定に関するお知らせ（取得枠の拡大）",
         "modification"),
        ("自己株式に関するその他のお知らせ", "other_treasury"),
    ],
)
def test_subtype_classification(title, expected):
    assert classify_buyback_subtype(title) == expected


def test_disposal_is_not_a_buyback_event():
    """処分 is the LARGEST treasury subtype (293/547) and has opposite sign."""
    ev = parse_buyback_event(_disc("自己株式の処分に関するお知らせ"))
    assert ev.subtype == "disposal"
    assert ev.is_t1_event is False


def test_cancellation_is_not_a_t1_event():
    ev = parse_buyback_event(_disc("自己株式の消却に関するお知らせ"))
    assert ev.subtype == "cancellation"
    assert ev.is_t1_event is False


def test_non_treasury_returns_none():
    assert classify_buyback_subtype("2027年3月期決算短信") is None
    assert parse_buyback_event(_disc("2027年3月期決算短信")) is None


def test_is_buyback_related_handles_fullwidth():
    assert is_buyback_related("自己株式の取得")
    assert not is_buyback_related("業務提携のお知らせ")


# --- contamination flags ----------------------------------------------------

def test_earnings_contamination_excludes_from_t1():
    ev = parse_buyback_event(_disc("2027年3月期業績予想の修正及び自己株式取得に係る事項の決定"))
    assert ev.subtype == "resolution"
    assert "earnings" in ev.contamination
    assert ev.is_t1_event is False


def test_dividend_contamination_excludes_from_t1():
    ev = parse_buyback_event(_disc("自己株式取得及び剰余金の配当（増配）に関するお知らせ"))
    assert "dividend" in ev.contamination
    assert ev.is_t1_event is False


def test_clean_resolution_is_a_t1_event():
    ev = parse_buyback_event(_disc("自己株式取得に係る事項の決定に関するお知らせ"))
    assert ev.is_t1_event is True
    assert ev.contamination == ()


def test_same_release_cancellation_is_flagged():
    ev = parse_buyback_event(_disc("自己株式取得に係る事項の決定及び自己株式の消却に関するお知らせ"))
    assert ev.subtype == "resolution"
    assert "cancellation_same_release" in ev.contamination


# --- corrections ------------------------------------------------------------

def test_correction_is_flagged_and_names_its_target():
    ev = parse_buyback_event(
        _disc("（訂正）「自己株式取得に係る事項の決定に関するお知らせ」の一部訂正について"))
    assert ev.is_correction is True
    assert ev.supersedes_hint == "自己株式取得に係る事項の決定に関するお知らせ"


def test_event_id_is_stable_across_title_correction():
    """A correction must not mint a new event id for the same underlying event."""
    a = compute_event_id("5133.T", "2026-07-15T15:00:00+09:00", "resolution")
    b = compute_event_id("5133.T", "2026-07-15T15:00:00+09:00", "resolution")
    assert a == b and len(a) == 32


def test_event_id_differs_by_subtype():
    a = compute_event_id("5133.T", "2026-07-15T15:00:00+09:00", "resolution")
    b = compute_event_id("5133.T", "2026-07-15T15:00:00+09:00", "execution_report")
    assert a != b


# --- extraction is honest about what it does not know -----------------------

def test_amount_extracted_from_title_when_present():
    ev = parse_buyback_event(_disc("自己株式取得に係る事項の決定（上限100億円）に関するお知らせ"))
    assert ev.amount_cap_jpy == 100 * 10**8
    assert ev.field_status["amount_cap_jpy"] == "title"


def test_absent_fields_are_none_with_a_reason_not_guessed():
    ev = parse_buyback_event(_disc("自己株式取得に係る事項の決定に関するお知らせ"))
    assert ev.amount_cap_jpy is None
    assert ev.share_cap is None
    assert ev.window_start is None
    assert ev.field_status["share_cap"] == "absent_in_title_requires_document"
    assert ev.parser_confidence == "low"


def test_percent_extraction_and_confidence_promotion():
    ev = parse_buyback_event(
        _disc("自己株式取得に係る事項の決定（上限50億円、発行済株式総数の2.5%）に関するお知らせ"))
    assert ev.amount_cap_jpy == 50 * 10**8
    assert ev.percent_of_shares == 2.5
    assert ev.parser_confidence == "high"


def test_acquisition_method_strata_are_extracted():
    tostnet = parse_buyback_event(_disc("自己株式の取得（ToSTNeT-3による買付け）の決定に関するお知らせ"))
    assert tostnet.acquisition_method == "tostnet"
    auction = parse_buyback_event(_disc("自己株式取得に係る事項の決定（市場買付）に関するお知らせ"))
    assert auction.acquisition_method == "auction"
    unknown = parse_buyback_event(_disc("自己株式取得に係る事項の決定に関するお知らせ"))
    assert unknown.acquisition_method is None
    assert unknown.field_status["acquisition_method"] == "absent_in_title_requires_document"


# --- fail-closed ------------------------------------------------------------

def test_missing_provenance_field_fails_closed():
    bad = _disc("自己株式取得に係る事項の決定に関するお知らせ")
    del bad["url"]
    with pytest.raises(BuybackParseError, match="provenance"):
        parse_buyback_event(bad)


def test_malformed_timestamp_fails_closed():
    bad = _disc("自己株式取得に係る事項の決定に関するお知らせ")
    bad["published_ts"] = "yesterday"
    with pytest.raises(BuybackParseError, match="ISO 8601"):
        parse_buyback_event(bad)


def test_empty_title_fails_closed():
    with pytest.raises(BuybackParseError):
        parse_buyback_event(_disc("   "))


def test_non_mapping_fails_closed():
    with pytest.raises(BuybackParseError):
        parse_buyback_event("自己株式")


# --- resolution -> execution report linkage ---------------------------------

def test_execution_reports_link_to_prior_resolution():
    events = [
        parse_buyback_event(_disc("自己株式取得に係る事項の決定に関するお知らせ",
                                  published="2026-05-01T15:00:00+09:00")),
        parse_buyback_event(_disc("自己株式の取得状況に関するお知らせ",
                                  published="2026-06-01T15:00:00+09:00")),
        parse_buyback_event(_disc("自己株式の取得状況に関するお知らせ",
                                  published="2026-07-01T15:00:00+09:00")),
    ]
    links = link_execution_reports(events)
    assert len(links[events[0].event_id]) == 2


def test_reports_before_any_resolution_stay_unattached():
    """No backward linking — that would manufacture a forward-looking join."""
    events = [
        parse_buyback_event(_disc("自己株式の取得状況に関するお知らせ",
                                  published="2026-04-01T15:00:00+09:00")),
        parse_buyback_event(_disc("自己株式取得に係る事項の決定に関するお知らせ",
                                  published="2026-05-01T15:00:00+09:00")),
    ]
    links = link_execution_reports(events)
    assert links[events[1].event_id] == []


def test_links_do_not_cross_tickers():
    events = [
        parse_buyback_event(_disc("自己株式取得に係る事項の決定に関するお知らせ",
                                  ticker="1111.T", published="2026-05-01T15:00:00+09:00")),
        parse_buyback_event(_disc("自己株式の取得状況に関するお知らせ",
                                  ticker="2222.T", published="2026-06-01T15:00:00+09:00")),
    ]
    links = link_execution_reports(events)
    assert links[events[0].event_id] == []


def test_corpus_summary_separates_t1_from_treasury():
    events = [
        parse_buyback_event(_disc("自己株式取得に係る事項の決定に関するお知らせ")),
        parse_buyback_event(_disc("自己株式の処分に関するお知らせ")),
        parse_buyback_event(_disc("自己株式の取得状況に関するお知らせ")),
    ]
    summary = corpus_summary(events)
    assert summary["total_treasury_events"] == 3
    assert summary["t1_primary_events"] == 1
    assert summary["by_subtype"]["disposal"] == 1
