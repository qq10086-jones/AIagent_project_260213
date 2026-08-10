"""P36-01 tests — 所有者別状況 extraction, fraction/percent guard, PIT storage.

The CSV fixture mirrors the real EDINET type=5 layout (UTF-16 TSV inside a zip,
columns 要素ID / 項目名 / コンテキストID / ... / 値) and uses the element IDs and
VALUES confirmed live against doc S100YNWZ (4750.T).
"""
import io
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.edinet_ownership import (  # noqa: E402
    OwnershipParseError,
    build_ownership_record,
    parse_ownership_csv,
    stored_ownership_doc_ids,
    upsert_ownership,
    validate_ownership,
)

HEADER = ["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別",
          "期間・時点", "ユニットID", "単位", "値"]

# Real values from 4750.T: individual 68.83%, other corps 27.51%, financial
# 2.39%, securities 0.24%, foreign corp 0.87%, foreign individual 0.16%.
REAL_PCTS = {
    "jpcrp_cor:PercentageOfShareholdingsFinancialInstitutions": "0.0239",
    "jpcrp_cor:PercentageOfShareholdingsFinancialServiceProviders": "0.0024",
    "jpcrp_cor:PercentageOfShareholdingsOtherCorporations": "0.2751",
    "jpcrp_cor:PercentageOfShareholdingsForeignersOtherThanIndividuals": "0.0087",
    "jpcrp_cor:PercentageOfShareholdingsForeignIndividuals": "0.0016",
    "jpcrp_cor:PercentageOfShareholdingsIndividualsAndOthers": "0.6883",
    "jpcrp_cor:PercentageOfShareholdingsNationalAndLocalGovernments": "－",
}
REAL_COUNTS = {
    "jpcrp_cor:NumberOfShareholdersTotal": "2913",
    "jpcrp_cor:NumberOfShareholdersIndividualsAndOthers": "2824",
    "jpcrp_cor:NumberOfShareholdersForeignInvestorsOtherThanIndividuals": "13",
    "jpcrp_cor:NumberOfShareholdersForeignIndividualInvestors": "15",
}


def _zip(rows, ctx="CurrentYearInstant_OrdinaryShareMember", member="jpcrp_x.csv"):
    lines = ["\t".join(f'"{h}"' for h in HEADER)]
    for eid, val in rows:
        cells = [f'"{eid}"', '"label"', f'"{ctx}"', '""', '""', '""', '""', '""',
                 f'"{val}"']
        lines.append("\t".join(cells))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, "\n".join(lines).encode("utf-16"))
    return buf.getvalue()


def _real_zip():
    return _zip(list(REAL_PCTS.items()) + list(REAL_COUNTS.items()))


# --- parsing ----------------------------------------------------------------

def test_parses_real_ownership_block():
    got = parse_ownership_csv(_real_zip())
    assert got["pct_individual"] == pytest.approx(0.6883)
    assert got["pct_foreign_corporate"] == pytest.approx(0.0087)
    assert got["n_shareholders_total"] == 2913
    # 「－」 (no government holders) is an absence, not a parse failure
    assert "pct_government" not in got


def test_document_without_ownership_block_returns_empty():
    assert parse_ownership_csv(_zip([("jpcrp_cor:NetSalesSummaryOfBusinessResults",
                                      "123")])) == {}


def test_non_zip_input_raises():
    with pytest.raises(OwnershipParseError, match="zip"):
        parse_ownership_csv(b"not a zip at all")


def test_wrong_context_is_ignored():
    """Prior-year and non-ordinary-class contexts are not this year's register."""
    z = _zip(list(REAL_PCTS.items()), ctx="Prior1YearInstant_OrdinaryShareMember")
    assert parse_ownership_csv(z) == {}
    z2 = _zip(list(REAL_PCTS.items()), ctx="CurrentYearInstant_TreasuryShareMember")
    assert parse_ownership_csv(z2) == {}


def test_plain_current_year_instant_is_accepted():
    z = _zip(list(REAL_PCTS.items()), ctx="CurrentYearInstant")
    assert parse_ownership_csv(z)["pct_individual"] == pytest.approx(0.6883)


# --- the fraction/percent trap ----------------------------------------------

def test_real_fractions_validate():
    ok, reason = validate_ownership(parse_ownership_csv(_real_zip()))
    assert ok, reason


def test_percent_scaled_values_are_rejected_with_a_named_reason():
    """If a future filing or a code change delivered 68.83 instead of 0.6883,
    storing it would make the conditioning variable 100x wrong and still look
    plausible. It must fail loudly."""
    scaled = {k: (str(float(v) * 100) if v != "－" else v)
              for k, v in REAL_PCTS.items()}
    ok, reason = validate_ownership(parse_ownership_csv(_zip(list(scaled.items()))))
    assert not ok
    assert "PERCENTS" in reason and "FRACTIONS" in reason


def test_categories_that_do_not_partition_are_rejected():
    partial = {"jpcrp_cor:PercentageOfShareholdingsIndividualsAndOthers": "0.30"}
    ok, reason = validate_ownership(parse_ownership_csv(_zip(list(partial.items()))))
    assert not ok and "partition" in reason


def test_empty_record_is_rejected():
    ok, reason = validate_ownership({})
    assert not ok and "no ownership" in reason


def test_out_of_range_fraction_rejected():
    ok, reason = validate_ownership({
        "pct_individual": 1.5, "pct_other_corporations": -0.5})
    assert not ok


# --- record assembly --------------------------------------------------------

def _record(**kw):
    params = dict(doc_id="S100YNWZ", symbol="4750.T", period_end="2026-03-31",
                  submitted_at="2026-06-26T09:02:00", doc_type_code="120",
                  parsed=parse_ownership_csv(_real_zip()))
    params.update(kw)
    return build_ownership_record(**params)


def test_record_carries_pit_fields_and_derived_aggregates():
    r = _record()
    assert r["as_of"] == "2026-03-31"                 # instant at FY end
    assert r["published_ts"] == "2026-06-26T09:02:00"  # when it became public
    assert r["pct_foreign_total"] == pytest.approx(0.0087 + 0.0016)
    assert r["pct_individual_total"] == pytest.approx(0.6883)


def test_record_from_empty_parse_raises():
    with pytest.raises(OwnershipParseError, match="no ownership block"):
        _record(parsed={})


def test_record_from_invalid_parse_raises():
    with pytest.raises(OwnershipParseError):
        _record(parsed={"pct_individual": 0.3})


# --- storage ----------------------------------------------------------------

def test_upsert_is_idempotent_and_resumable(tmp_path):
    db = tmp_path / "own.db"
    r = _record()
    assert upsert_ownership(db, [r]) == 1
    assert upsert_ownership(db, [r]) == 1        # re-run overwrites, no duplicate
    import sqlite3
    conn = sqlite3.connect(str(db))
    n = conn.execute("select count(*) from ownership_snapshots").fetchone()[0]
    got = conn.execute("select pct_individual_total, published_ts from "
                       "ownership_snapshots").fetchone()
    conn.close()
    assert n == 1
    assert got[0] == pytest.approx(0.6883)
    assert stored_ownership_doc_ids(db) == {"S100YNWZ"}


def test_stored_ids_on_missing_db_is_empty(tmp_path):
    assert stored_ownership_doc_ids(tmp_path / "nope.db") == set()


def test_two_years_of_same_symbol_coexist(tmp_path):
    db = tmp_path / "own.db"
    upsert_ownership(db, [_record()])
    upsert_ownership(db, [_record(doc_id="S100OLDX", period_end="2025-03-31",
                                  submitted_at="2025-06-26T09:02:00")])
    assert len(stored_ownership_doc_ids(db)) == 2


# --- shares outstanding (size control) --------------------------------------

def _zip_with_shares(share_ctx="FilingDateInstant_OrdinaryShareMember"):
    lines = ["\t".join(f'"{h}"' for h in HEADER)]
    for eid, val in list(REAL_PCTS.items()) + list(REAL_COUNTS.items()):
        lines.append("\t".join([f'"{eid}"', '"l"',
                                '"CurrentYearInstant_OrdinaryShareMember"',
                                '""', '""', '""', '""', '""', f'"{val}"']))
    lines.append("\t".join(
        ['"jpcrp_cor:NumberOfIssuedSharesAsOfFiscalYearEndIssuedSharesTotalNumberOfSharesEtc"',
         '"l"', f'"{share_ctx}"', '""', '""', '""', '""', '""', '"7618000"']))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("jpcrp_x.csv", "\n".join(lines).encode("utf-16"))
    return buf.getvalue()


def test_shares_outstanding_is_extracted():
    got = parse_ownership_csv(_zip_with_shares())
    assert got["shares_outstanding"] == 7618000
    assert got["pct_individual"] == pytest.approx(0.6883)


def test_shares_outstanding_accepts_current_year_context():
    got = parse_ownership_csv(_zip_with_shares(share_ctx="CurrentYearInstant"))
    assert got["shares_outstanding"] == 7618000


def test_prior_year_share_count_is_rejected():
    """A 5-year block must not overwrite the current count with an old one."""
    got = parse_ownership_csv(_zip_with_shares(
        share_ctx="Prior4YearInstant_NonConsolidatedMember"))
    assert "shares_outstanding" not in got


def test_record_and_storage_carry_shares(tmp_path):
    r = build_ownership_record(
        doc_id="S1", symbol="4750.T", period_end="2026-03-31",
        submitted_at="2026-06-26T09:02:00", doc_type_code="120",
        parsed=parse_ownership_csv(_zip_with_shares()))
    assert r["shares_outstanding"] == 7618000
    db = tmp_path / "own.db"
    upsert_ownership(db, [r])
    import sqlite3
    conn = sqlite3.connect(str(db))
    got = conn.execute("select shares_outstanding from ownership_snapshots").fetchone()[0]
    conn.close()
    assert got == 7618000


def test_legacy_table_without_shares_column_is_migrated(tmp_path):
    """An existing panel predates the column; upsert must ALTER, not crash."""
    import sqlite3
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE ownership_snapshots (doc_id TEXT, symbol TEXT, "
                 "as_of TEXT, published_ts TEXT, doc_type_code TEXT, "
                 "pct_government REAL, pct_financial_institutions REAL, "
                 "pct_securities_firms REAL, pct_other_corporations REAL, "
                 "pct_foreign_corporate REAL, pct_foreign_individual REAL, "
                 "pct_individual REAL, pct_foreign_total REAL, "
                 "pct_individual_total REAL, n_shareholders_total REAL, "
                 "n_shareholders_individual REAL, "
                 "n_shareholders_foreign_corporate REAL, "
                 "n_shareholders_foreign_individual REAL, source TEXT, "
                 "PRIMARY KEY (doc_id, symbol))")
    conn.commit()
    conn.close()
    r = build_ownership_record(
        doc_id="S1", symbol="4750.T", period_end="2026-03-31",
        submitted_at="2026-06-26T09:02:00", doc_type_code="120",
        parsed=parse_ownership_csv(_zip_with_shares()))
    assert upsert_ownership(db, [r]) == 1
    conn = sqlite3.connect(str(db))
    cols = {d[1] for d in conn.execute("pragma table_info(ownership_snapshots)")}
    conn.close()
    assert "shares_outstanding" in cols
