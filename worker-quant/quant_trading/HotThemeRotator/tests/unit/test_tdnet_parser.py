"""Tests for TDnet parsers (P10-14 Cycle 1) — fixture-based, no network."""
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.tdnet_parser import (  # noqa: E402
    TdnetParseError,
    classify_category,
    normalize_ticker,
    parse_tdnet_html,
    parse_yanoshin_json,
)


FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "tdnet"


# ---------- normalize_ticker ----------


def test_normalize_ticker_4digit():
    assert normalize_ticker("6779") == "6779.T"


def test_normalize_ticker_5digit_tdnet_form():
    assert normalize_ticker("67790") == "6779.T"


def test_normalize_ticker_strips_whitespace():
    assert normalize_ticker("  6779  ") == "6779.T"


def test_normalize_ticker_rejects_non_string():
    with pytest.raises(TdnetParseError):
        normalize_ticker(6779)


def test_normalize_ticker_rejects_letters():
    with pytest.raises(TdnetParseError):
        normalize_ticker("AAPL")


def test_normalize_ticker_rejects_wrong_length():
    with pytest.raises(TdnetParseError):
        normalize_ticker("12")
    with pytest.raises(TdnetParseError):
        normalize_ticker("123456")


def test_normalize_ticker_5digit_must_end_with_zero():
    with pytest.raises(TdnetParseError):
        normalize_ticker("67791")


# ---------- classify_category ----------


def test_classify_earnings():
    assert classify_category("業績予想の修正に関するお知らせ") == "earnings"
    assert classify_category("2026年3月期決算短信") == "earnings"


def test_classify_tob():
    assert classify_category("公開買付けに関する意見表明のお知らせ") == "tob"
    assert classify_category("TOB に関するお知らせ") == "tob"


def test_classify_dividend():
    assert classify_category("配当金のお知らせ") == "dividend"
    assert classify_category("無配のお知らせ") == "dividend"


def test_classify_split():
    assert classify_category("株式分割のお知らせ") == "split"
    assert classify_category("株式併合に関するお知らせ") == "split"


def test_classify_suspension():
    assert classify_category("売買停止のお知らせ") == "suspension"
    assert classify_category("上場廃止に関するお知らせ") == "suspension"


def test_classify_order():
    assert classify_category("業務提携に関するお知らせ") == "order"
    assert classify_category("大型受注のお知らせ") == "order"


def test_classify_governance():
    assert classify_category("代表取締役の異動に関するお知らせ") == "governance"


def test_classify_unmapped_falls_back_to_other():
    assert classify_category("特殊なお知らせ") == "other"


def test_classify_non_string_returns_other():
    assert classify_category(None) == "other"


def test_classify_tob_precedes_earnings():
    """A title mentioning both TOB and 業績 should classify as tob (order matters)."""
    title = "公開買付けに関する業績への影響について"
    assert classify_category(title) == "tob"


# ---------- parse_yanoshin_json ----------


def test_parse_yanoshin_json_from_fixture():
    fixture = FIXTURE_DIR / "yanoshin_sample.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    records = parse_yanoshin_json(
        payload, collected_ts="2026-05-25T16:00:00+09:00"
    )
    assert len(records) == 3
    by_ticker = {r.ticker: r for r in records}
    assert by_ticker["6779.T"].category == "earnings"
    assert by_ticker["1306.T"].category == "dividend"
    assert by_ticker["6768.T"].category == "tob"


def test_parse_yanoshin_json_preserves_company_name():
    fixture = FIXTURE_DIR / "yanoshin_sample.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    records = parse_yanoshin_json(
        payload, collected_ts="2026-05-25T16:00:00+09:00"
    )
    nihon_dempa = next(r for r in records if r.ticker == "6779.T")
    assert nihon_dempa.company_name == "日本電波工業株式会社"


def test_parse_yanoshin_json_attaches_raw_dict():
    fixture = FIXTURE_DIR / "yanoshin_sample.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    records = parse_yanoshin_json(
        payload, collected_ts="2026-05-25T16:00:00+09:00"
    )
    assert records[0].raw is not None
    assert "company_code" in records[0].raw
    assert "markets" in records[0].raw


def test_parse_yanoshin_json_rejects_non_mapping():
    with pytest.raises(TdnetParseError):
        parse_yanoshin_json(
            ["item"], collected_ts="2026-05-25T16:00:00+09:00"
        )


def test_parse_yanoshin_json_rejects_missing_items_key():
    with pytest.raises(TdnetParseError):
        parse_yanoshin_json(
            {"data": []}, collected_ts="2026-05-25T16:00:00+09:00"
        )


def test_parse_yanoshin_json_rejects_string_items():
    with pytest.raises(TdnetParseError):
        parse_yanoshin_json(
            {"items": "not a list"}, collected_ts="2026-05-25T16:00:00+09:00"
        )


def test_parse_yanoshin_json_skips_items_missing_required_fields():
    payload = {
        "items": [
            {  # valid
                "company_code": "67790",
                "pubdate": "2026-05-25T08:30:00+09:00",
                "title": "業績予想の修正",
                "url": "https://example.com/x.pdf",
            },
            {  # missing url
                "company_code": "67790",
                "pubdate": "2026-05-25T08:30:00+09:00",
                "title": "別件",
            },
            {  # bad ticker
                "company_code": "INVALID",
                "pubdate": "2026-05-25T08:30:00+09:00",
                "title": "業績修正",
                "url": "https://example.com/y.pdf",
            },
            {  # bad pubdate
                "company_code": "67790",
                "pubdate": "not iso",
                "title": "業績修正",
                "url": "https://example.com/z.pdf",
            },
        ]
    }
    records = parse_yanoshin_json(
        payload, collected_ts="2026-05-25T16:00:00+09:00"
    )
    assert len(records) == 1


def test_parse_yanoshin_json_skips_non_mapping_items():
    payload = {"items": ["string item", None, 42]}
    records = parse_yanoshin_json(
        payload, collected_ts="2026-05-25T16:00:00+09:00"
    )
    assert records == ()


# ---------- real Yanoshin shape (post live smoke test 2026-05-25) ----------


def test_parse_yanoshin_real_wrapped_items():
    """Real Yanoshin wraps each item in {'Tdnet': {...}}, uses 'document_url'
    instead of 'url', and 'pubdate' is space-separated 'YYYY-MM-DD HH:MM:SS'
    without timezone. Verified against live endpoint 2026-05-25.
    """
    fixture = FIXTURE_DIR / "yanoshin_real_sample.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    records = parse_yanoshin_json(
        payload, collected_ts="2026-05-25T20:00:00+09:00"
    )
    assert len(records) == 3
    by_ticker = {r.ticker: r for r in records}
    assert by_ticker["6779.T"].category == "earnings"
    assert by_ticker["6779.T"].published_ts == "2026-05-25T18:15:00+09:00"
    assert by_ticker["6779.T"].url.startswith("https://webapi.yanoshin.jp")
    assert by_ticker["1306.T"].category == "dividend"
    assert by_ticker["6768.T"].category == "tob"


def test_parse_yanoshin_real_preserves_company_name():
    fixture = FIXTURE_DIR / "yanoshin_real_sample.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    records = parse_yanoshin_json(
        payload, collected_ts="2026-05-25T20:00:00+09:00"
    )
    nihon_dempa = next(r for r in records if r.ticker == "6779.T")
    assert nihon_dempa.company_name == "日本電波工業"


def test_parse_yanoshin_handles_mixed_wrapped_and_flat():
    """Both shapes coexist (real wrapped + old flat); parser tolerates either."""
    payload = {
        "items": [
            {
                "Tdnet": {
                    "company_code": "67790",
                    "pubdate": "2026-05-25 18:15:00",
                    "title": "業績修正",
                    "document_url": "https://example.com/a.pdf",
                }
            },
            {
                "company_code": "13060",
                "pubdate": "2026-05-25T15:00:00+09:00",
                "title": "配当",
                "url": "https://example.com/b.pdf",
            },
        ]
    }
    records = parse_yanoshin_json(
        payload, collected_ts="2026-05-25T20:00:00+09:00"
    )
    assert len(records) == 2


def test_normalize_pubdate_adds_jst_to_space_format():
    """Internal helper: 'YYYY-MM-DD HH:MM:SS' -> ISO with +09:00."""
    from hot_theme_rotator.data.external.tdnet_parser import _normalize_pubdate
    assert _normalize_pubdate("2026-05-25 18:15:00") == "2026-05-25T18:15:00+09:00"


def test_normalize_pubdate_passes_through_iso_with_tz():
    from hot_theme_rotator.data.external.tdnet_parser import _normalize_pubdate
    assert (
        _normalize_pubdate("2026-05-25T18:15:00+09:00")
        == "2026-05-25T18:15:00+09:00"
    )


def test_normalize_pubdate_appends_jst_to_t_separated_naive():
    """Per Codex review 2026-05-25 (Rule 8.2 PIT): naive T-separated ISO without
    TZ must get JST appended, since Yanoshin pubdate is implicitly JST."""
    from hot_theme_rotator.data.external.tdnet_parser import _normalize_pubdate
    assert (
        _normalize_pubdate("2026-05-25T18:15:00")
        == "2026-05-25T18:15:00+09:00"
    )


def test_normalize_pubdate_preserves_z_suffix():
    from hot_theme_rotator.data.external.tdnet_parser import _normalize_pubdate
    assert (
        _normalize_pubdate("2026-05-25T18:15:00Z")
        == "2026-05-25T18:15:00Z"
    )


# ---------- parse_tdnet_html ----------


def test_parse_tdnet_html_from_fixture():
    fixture = FIXTURE_DIR / "tdnet_html_sample.html"
    html = fixture.read_text(encoding="utf-8")
    records = parse_tdnet_html(
        html,
        trade_date="2026-05-25",
        collected_ts="2026-05-25T16:00:00+09:00",
    )
    assert len(records) == 3
    by_ticker = {r.ticker: r for r in records}
    assert by_ticker["6779.T"].category == "earnings"
    assert by_ticker["1306.T"].category == "dividend"
    assert by_ticker["6768.T"].category == "tob"


def test_parse_tdnet_html_builds_iso_published_ts_with_jst():
    fixture = FIXTURE_DIR / "tdnet_html_sample.html"
    html = fixture.read_text(encoding="utf-8")
    records = parse_tdnet_html(
        html,
        trade_date="2026-05-25",
        collected_ts="2026-05-25T16:00:00+09:00",
    )
    nihon_dempa = next(r for r in records if r.ticker == "6779.T")
    assert nihon_dempa.published_ts == "2026-05-25T08:30:00+09:00"


def test_parse_tdnet_html_skips_header_row():
    """Header rows use <th>, parsed as 0 <td>, so skipped."""
    fixture = FIXTURE_DIR / "tdnet_html_sample.html"
    html = fixture.read_text(encoding="utf-8")
    records = parse_tdnet_html(
        html,
        trade_date="2026-05-25",
        collected_ts="2026-05-25T16:00:00+09:00",
    )
    assert len(records) == 3


def test_parse_tdnet_html_rejects_non_string():
    with pytest.raises(TdnetParseError):
        parse_tdnet_html(
            b"<html></html>",
            trade_date="2026-05-25",
            collected_ts="2026-05-25T16:00:00+09:00",
        )


def test_parse_tdnet_html_empty_returns_empty_tuple():
    records = parse_tdnet_html(
        "<html><body></body></html>",
        trade_date="2026-05-25",
        collected_ts="2026-05-25T16:00:00+09:00",
    )
    assert records == ()


def test_parse_tdnet_html_skips_row_with_bad_time():
    html = """
    <table>
      <tr><td>not-a-time</td><td>67790</td><td>name</td><td>業績修正</td><td><a href="x.pdf">x</a></td></tr>
      <tr><td>08:30</td><td>67790</td><td>name</td><td>業績修正</td><td><a href="y.pdf">y</a></td></tr>
    </table>
    """
    records = parse_tdnet_html(
        html,
        trade_date="2026-05-25",
        collected_ts="2026-05-25T16:00:00+09:00",
    )
    assert len(records) == 1


def test_parse_tdnet_html_skips_row_with_missing_url():
    html = """
    <table>
      <tr><td>08:30</td><td>67790</td><td>name</td><td>業績修正</td><td>no link here</td></tr>
      <tr><td>09:00</td><td>67790</td><td>name</td><td>業績下方修正</td><td><a href="y.pdf">y</a></td></tr>
    </table>
    """
    records = parse_tdnet_html(
        html,
        trade_date="2026-05-25",
        collected_ts="2026-05-25T16:00:00+09:00",
    )
    assert len(records) == 1
