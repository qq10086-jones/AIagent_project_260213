"""Tests for the EDINET fundamental panel ingestion (P23-B).

Parser contracts under test: element whitelist mapping, consolidated-over-
parent preference, relative-year extraction from context IDs, honest skipping
of empty values, PIT record assembly (published_ts = THIS document's submit
time; prior-year period ends flagged as estimated), and idempotent storage.
No network — the client takes an injected transport.
"""
import io
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.edinet_fundamentals import (  # noqa: E402
    EdinetFundamentalsClient,
    build_records,
    parse_summary_csv,
    upsert_records,
)

HEADER = '"要素ID"\t"項目名"\t"コンテキストID"\t"相対年度"\t"連結・個別"\t"期間・時点"\t"ユニットID"\t"単位"\t"値"'


def _row(eid, ctx, val):
    return f'"{eid}"\t"項目"\t"{ctx}"\t"当期"\t"連結"\t"期間"\t"JPY"\t"円"\t"{val}"'


def _csv_zip(lines, member="XBRL_TO_CSV/jpcrp030000-asr-001_E00000-000_2026-03-31_01_2026-06-26.csv"):
    text = "\n".join([HEADER] + lines)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, text.encode("utf-16"))
        zf.writestr("XBRL_TO_CSV/jpaud-aai-cn-001_x.csv", "x".encode("utf-16"))
    return buf.getvalue()


def test_parse_maps_elements_and_relative_years():
    blob = _csv_zip([
        _row("jpcrp_cor:NetSalesSummaryOfBusinessResults", "CurrentYearDuration", "1000"),
        _row("jpcrp_cor:NetSalesSummaryOfBusinessResults", "Prior1YearDuration", "900"),
        _row("jpcrp_cor:NetAssetsSummaryOfBusinessResults", "CurrentYearInstant", "5000"),
        _row("jpcrp_cor:RateOfReturnOnEquitySummaryOfBusinessResults", "CurrentYearDuration", "0.112"),
        _row("jpcrp_cor:NetAssetsPerShareSummaryOfBusinessResults", "Prior2YearInstant", "1451.27"),
    ])
    years = parse_summary_csv(blob)
    assert years[0]["revenue"] == 1000.0
    assert years[0]["net_assets"] == 5000.0
    assert years[0]["roe"] == 0.112
    assert years[1]["revenue"] == 900.0
    assert years[2]["bps"] == 1451.27


def test_consolidated_preferred_over_parent_only():
    blob = _csv_zip([
        _row("jpcrp_cor:NetSalesSummaryOfBusinessResults", "CurrentYearDuration_NonConsolidatedMember", "800"),
        _row("jpcrp_cor:NetSalesSummaryOfBusinessResults", "CurrentYearDuration", "1000"),
        # parent-only arriving AFTER consolidated must not overwrite it
        _row("jpcrp_cor:NetAssetsSummaryOfBusinessResults", "CurrentYearInstant", "5000"),
        _row("jpcrp_cor:NetAssetsSummaryOfBusinessResults", "CurrentYearInstant_NonConsolidatedMember", "4000"),
    ])
    years = parse_summary_csv(blob)
    assert years[0]["revenue"] == 1000.0
    assert years[0]["net_assets"] == 5000.0
    assert years[0]["consolidated"] is True


def test_parent_only_filer_still_parses():
    blob = _csv_zip([
        _row("jpcrp_cor:NetSalesSummaryOfBusinessResults", "CurrentYearDuration_NonConsolidatedMember", "800"),
    ])
    years = parse_summary_csv(blob)
    assert years[0]["revenue"] == 800.0
    assert years[0]["consolidated"] is False


def test_empty_and_dash_values_skipped_and_segment_contexts_ignored():
    blob = _csv_zip([
        _row("jpcrp_cor:NetSalesSummaryOfBusinessResults", "CurrentYearDuration", "－"),
        _row("jpcrp_cor:NetSalesSummaryOfBusinessResults", "CurrentYearDuration_SomeSegmentMember", "123"),
    ])
    assert parse_summary_csv(blob) == []


def test_build_records_pit_and_estimated_periods():
    years = [
        {"relative_year": 0, "consolidated": True, "revenue": 1000.0},
        {"relative_year": 2, "consolidated": True, "revenue": 800.0},
    ]
    recs = build_records(
        years,
        doc_id="S100TEST",
        symbol="6248.T",
        doc_type_code="120",
        period_end="2026-03-31",
        submitted_at=datetime(2026, 6, 26, 9, 2),
    )
    by_rel = {r["relative_year"]: r for r in recs}
    assert by_rel[0]["fiscal_period_end"] == "2026-03-31"
    assert by_rel[0]["period_basis"] == "reported"
    assert by_rel[2]["fiscal_period_end"] == "2024-03-31"
    assert by_rel[2]["period_basis"] == "estimated_shift"
    # PIT: every relative year carries THIS document's publish time
    assert all(r["published_ts"].startswith("2026-06-26") for r in recs)
    assert all(r["symbol"] == "6248.T" for r in recs)


def test_upsert_idempotent(tmp_path):
    db = tmp_path / "f.db"
    recs = build_records(
        [{"relative_year": 0, "consolidated": True, "revenue": 1000.0}],
        doc_id="S100TEST", symbol="6248.T", doc_type_code="120",
        period_end="2026-03-31", submitted_at=datetime(2026, 6, 26),
    )
    assert upsert_records(db, recs) == 1
    assert upsert_records(db, recs) == 0  # second pass inserts nothing
    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from fundamental_snapshots").fetchone()[0] == 1


def test_client_transport_injection_and_seccode_mapping():
    def transport(url, params, raw=False):
        assert "documents.json" in url
        return {"results": [
            {"docID": "S100A", "docTypeCode": "120", "secCode": "62480",
             "csvFlag": "1", "docDescription": "有価証券報告書", "periodEnd": "2026-03-31",
             "submitDateTime": "2026-06-26 09:02", "filerName": "テスト社"},
            {"docID": "S100B", "docTypeCode": "120", "secCode": None,
             "csvFlag": "1", "docDescription": "有価証券報告書（内国投資信託受益証券）"},
            {"docID": "S100C", "docTypeCode": "180", "secCode": "12340", "csvFlag": "1"},
        ]}

    client = EdinetFundamentalsClient(api_key="k", transport=transport)
    docs = client.list_fundamental_documents("2026-06-26")
    assert [d["doc_id"] for d in docs] == ["S100A"]  # fund + 臨時報告書 filtered out
    assert docs[0]["symbol"] == "6248.T"
