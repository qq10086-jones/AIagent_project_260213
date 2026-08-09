"""P36-01 — 所有者別状況 (ownership structure) from EDINET 有価証券報告書.

Why this exists
---------------
T2 (ownership-conditioned PEAD) rests on Jinushi et al. (TJAR 13): Japanese
post-earnings drift decayed overall between 2002 and 2020, but did NOT decay in
firms with **low foreign ownership / high individual ownership**. That
conditioning variable was the one genuinely missing input in the T2 chain — the
P23-B panel already supplies the EPS history and PIT filing timestamps.

Every annual report carries a 所有者別状況 table: shareholder counts, units held,
and ownership share per investor category. The element IDs below were confirmed
against a live filing (doc S100YNWZ, 4750.T) rather than assumed from taxonomy
documentation.

Two traps this module handles explicitly
-----------------------------------------
1. **The percentages are FRACTIONS, not percents.** The Japanese label reads
   「所有株式数の割合（％）」 but the XBRL value for 68.83% is ``0.6883``.
   Reading them as percents would put foreign ownership at 0.87% instead of
   87%… or 0.0087% instead of 0.87%, depending on which way you err. The parser
   stores fractions and :func:`validate_ownership` checks the categories sum to
   ~1.0, which catches the confusion immediately.
2. **One observation per annual filing.** Ownership is an INSTANT
   (``CurrentYearInstant_OrdinaryShareMember``) as of the fiscal year end, made
   public at ``submitted_at``. It is therefore an annual PIT snapshot with a
   validity window running to the next filing — never a continuously-known
   series. Consumers must join on the publication timestamp, exactly like the
   restated-fundamentals rows.

Rule 3: data extraction only. No score, no signal, no recommendation.
"""
from __future__ import annotations

import re
import sqlite3
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "OWNERSHIP_SCHEMA",
    "PERCENT_ELEMENTS",
    "COUNT_ELEMENTS",
    "OwnershipParseError",
    "parse_ownership_csv",
    "build_ownership_record",
    "validate_ownership",
    "upsert_ownership",
    "stored_ownership_doc_ids",
]

# Confirmed live 2026-08-09 against doc S100YNWZ (4750.T), 有価証券報告書.
PERCENT_ELEMENTS = {
    "jpcrp_cor:PercentageOfShareholdingsNationalAndLocalGovernments": "pct_government",
    "jpcrp_cor:PercentageOfShareholdingsFinancialInstitutions": "pct_financial_institutions",
    "jpcrp_cor:PercentageOfShareholdingsFinancialServiceProviders": "pct_securities_firms",
    "jpcrp_cor:PercentageOfShareholdingsOtherCorporations": "pct_other_corporations",
    "jpcrp_cor:PercentageOfShareholdingsForeignersOtherThanIndividuals": "pct_foreign_corporate",
    "jpcrp_cor:PercentageOfShareholdingsForeignIndividuals": "pct_foreign_individual",
    "jpcrp_cor:PercentageOfShareholdingsIndividualsAndOthers": "pct_individual",
}

COUNT_ELEMENTS = {
    "jpcrp_cor:NumberOfShareholdersTotal": "n_shareholders_total",
    "jpcrp_cor:NumberOfShareholdersIndividualsAndOthers": "n_shareholders_individual",
    "jpcrp_cor:NumberOfShareholdersForeignInvestorsOtherThanIndividuals":
        "n_shareholders_foreign_corporate",
    "jpcrp_cor:NumberOfShareholdersForeignIndividualInvestors":
        "n_shareholders_foreign_individual",
}

_PCT_FIELDS = tuple(PERCENT_ELEMENTS.values())

# Ownership is an instant on the ordinary-share class. Treasury/other share
# classes carry their own members and are deliberately NOT summed in.
_CTX_RE = re.compile(r"^CurrentYearInstant(_OrdinaryShareMember)?$")

_EMPTY_VALUES = {"", "－", "―", "-", "N/A", "NaN", "null", "None"}

# The categories partition the register, so they must sum to ~1. Filings round
# each category independently, so an exact 1.0 is not expected.
_SUM_TOLERANCE = 0.02


class OwnershipParseError(ValueError):
    """Raised when an ownership block cannot be read safely."""


def _parse_value(raw: str) -> float | None:
    s = raw.strip().strip('"').replace(",", "")
    if s in _EMPTY_VALUES:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_ownership_csv(zip_bytes: bytes) -> dict[str, Any]:
    """Extract the 所有者別状況 block from an EDINET type=5 CSV zip.

    Returns a dict of the mapped fields (missing categories simply absent — a
    filing with no government holders reports 「－」, which is a real zero-ish
    absence, not a parse failure). Returns ``{}`` when the document carries no
    ownership block at all, so a caller can distinguish "not present" from
    "present but unreadable" (the latter raises).
    """
    try:
        zf = zipfile.ZipFile(BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise OwnershipParseError(f"not a readable zip: {exc}") from exc
    members = [n for n in zf.namelist() if "jpcrp" in n and n.endswith(".csv")]
    if not members:
        return {}
    try:
        text = zf.read(members[0]).decode("utf-16")
    except (UnicodeDecodeError, KeyError) as exc:
        raise OwnershipParseError(f"csv member unreadable: {exc}") from exc

    lines = text.splitlines()
    if not lines:
        return {}
    header = [c.strip('"') for c in lines[0].split("\t")]
    try:
        i_eid = header.index("要素ID")
        i_ctx = header.index("コンテキストID")
        i_val = header.index("値")
    except ValueError:
        return {}

    out: dict[str, Any] = {}
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(header):
            continue
        eid = cells[i_eid].strip('"')
        field = PERCENT_ELEMENTS.get(eid) or COUNT_ELEMENTS.get(eid)
        if field is None:
            continue
        if not _CTX_RE.match(cells[i_ctx].strip('"')):
            continue
        value = _parse_value(cells[i_val])
        if value is None:
            continue
        out[field] = value
    return out


def validate_ownership(record: Mapping[str, Any]) -> tuple[bool, str]:
    """Check the ownership fractions partition the register.

    Returns (ok, reason). The decisive guard against the fraction/percent
    confusion: if the values were percents (68.83 rather than 0.6883) the sum
    would land near 100, not near 1, and this fails loudly instead of storing a
    silently 100x-wrong conditioning variable.
    """
    present = [record.get(f) for f in _PCT_FIELDS if record.get(f) is not None]
    if not present:
        return False, "no ownership percentage categories present"
    total = sum(present)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        if 90.0 <= total <= 110.0:
            return False, (
                f"ownership categories sum to {total:.2f} — these look like "
                f"PERCENTS, but this schema stores FRACTIONS (68.83% -> 0.6883)")
        # Name the outlier: observed cause is a filer typing ONE field in a
        # different unit (e.g. 3925.T reports foreign-corporate as 51.45 while
        # every sibling field is a fraction). Recording which field is out of
        # family lets a human adjudicate; guessing a rescale would fabricate.
        outlier = max(
            ((f, record.get(f)) for f in _PCT_FIELDS if record.get(f) is not None),
            key=lambda kv: kv[1], default=(None, None))
        extra = (f"; largest category {outlier[0]}={outlier[1]} looks out of "
                 f"family" if outlier[0] and outlier[1] and outlier[1] > 1.0 else "")
        return False, (
            f"ownership categories sum to {total:.4f}, outside 1.0 "
            f"+/- {_SUM_TOLERANCE}; the register does not partition{extra}")
    for f in _PCT_FIELDS:
        v = record.get(f)
        if v is not None and not (0.0 <= v <= 1.0):
            return False, f"{f}={v} outside [0, 1]"
    return True, "ok"


def build_ownership_record(
    *,
    doc_id: str,
    symbol: str,
    period_end: str,
    submitted_at: str,
    doc_type_code: str,
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one PIT ownership snapshot. Raises if the block is unusable."""
    if not parsed:
        raise OwnershipParseError(f"{doc_id}: no ownership block in document")
    ok, reason = validate_ownership(parsed)
    if not ok:
        raise OwnershipParseError(f"{doc_id} ({symbol}): {reason}")

    record = {
        "doc_id": doc_id,
        "symbol": symbol,
        "as_of": period_end,          # ownership is an INSTANT at fiscal year end
        "published_ts": submitted_at,  # PIT: when it became public
        "doc_type_code": doc_type_code,
        "source": "edinet_shareholder_status",
    }
    for f in _PCT_FIELDS:
        record[f] = parsed.get(f)
    for f in COUNT_ELEMENTS.values():
        record[f] = parsed.get(f)

    # Derived aggregates — the two T2 actually conditions on.
    fc = parsed.get("pct_foreign_corporate") or 0.0
    fi = parsed.get("pct_foreign_individual") or 0.0
    record["pct_foreign_total"] = fc + fi
    record["pct_individual_total"] = parsed.get("pct_individual")
    return record


OWNERSHIP_SCHEMA = """
CREATE TABLE IF NOT EXISTS ownership_snapshots (
    doc_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    as_of TEXT NOT NULL,
    published_ts TEXT NOT NULL,
    doc_type_code TEXT,
    pct_government REAL,
    pct_financial_institutions REAL,
    pct_securities_firms REAL,
    pct_other_corporations REAL,
    pct_foreign_corporate REAL,
    pct_foreign_individual REAL,
    pct_individual REAL,
    pct_foreign_total REAL,
    pct_individual_total REAL,
    n_shareholders_total REAL,
    n_shareholders_individual REAL,
    n_shareholders_foreign_corporate REAL,
    n_shareholders_foreign_individual REAL,
    source TEXT,
    PRIMARY KEY (doc_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_ownership_symbol_pub
    ON ownership_snapshots (symbol, published_ts);
"""


def upsert_ownership(db_path: Path | str, records: list[dict[str, Any]]) -> int:
    """Idempotent upsert keyed on (doc_id, symbol)."""
    if not records:
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(OWNERSHIP_SCHEMA)
        cols = [
            "doc_id", "symbol", "as_of", "published_ts", "doc_type_code",
            *_PCT_FIELDS, "pct_foreign_total", "pct_individual_total",
            *COUNT_ELEMENTS.values(), "source",
        ]
        placeholders = ",".join("?" * len(cols))
        conn.executemany(
            f"INSERT OR REPLACE INTO ownership_snapshots ({','.join(cols)}) "
            f"VALUES ({placeholders})",
            [tuple(r.get(c) for c in cols) for r in records],
        )
        conn.commit()
        return len(records)
    finally:
        conn.close()


def stored_ownership_doc_ids(db_path: Path | str) -> set[str]:
    """doc_ids already stored — lets a backfill resume without refetching."""
    path = Path(db_path)
    if not path.exists():
        return set()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in conn.execute(
            "select name from sqlite_master where type='table'")}
        if "ownership_snapshots" not in tables:
            return set()
        return {r[0] for r in conn.execute(
            "select distinct doc_id from ownership_snapshots")}
    finally:
        conn.close()
