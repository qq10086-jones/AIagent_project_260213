"""EDINET fundamental panel ingestion (P23-B — official FSA source, free).

Fills the missing piece identified 2026-07-03: Project_v5 had a working EDINET
list client and a PIT-ready ``fundamental_snapshots`` schema but the values
parser was never built (and its backfill filtered docTypeCode 130 = 訂正 as
"annual"; the correct annual 有価証券報告書 code is 120 — verified live).

This module is HTR-side and self-contained (ADR-0005 — no cross-tree import).
It lists filings, downloads the type=5 CSV package, and extracts the
経営指標等 summary block — which carries FIVE fiscal years per annual report
(Current + Prior1..4): revenue / ordinary income / net income / net assets /
total assets / BPS / EPS / ROE / equity ratio — into
``data/raw/htr_fundamentals.db``.

PIT contract: every stored row's ``published_ts`` is THIS document's
submitDateTime (when the values became available via this filing). Prior-year
rows therefore carry a conservative (late) availability stamp; their
``fiscal_period_end`` is an estimated year-shift flagged ``estimated_shift``.
Research-data layer only: no scores, no probabilities, no advice.
"""
from __future__ import annotations

import json
import re
import sqlite3
import zipfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"
DEFAULT_TIMEOUT_SECONDS = 60

# 有価証券報告書 (annual) + 半期報告書 (semiannual). 120 verified live 2026-07-03.
FUNDAMENTAL_DOC_TYPES = {"120", "160"}

# 経営指標等 summary elements → snapshot fields.
ELEMENT_MAP = {
    "jpcrp_cor:NetSalesSummaryOfBusinessResults": "revenue",
    "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults": "revenue",
    "jpcrp_cor:OperatingRevenue1SummaryOfBusinessResults": "revenue",
    "jpcrp_cor:OperatingIncomeLossSummaryOfBusinessResults": "operating_income",
    "jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults": "ordinary_income",
    "jpcrp_cor:NetIncomeLossSummaryOfBusinessResults": "net_income",
    "jpcrp_cor:ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults": "net_income",
    "jpcrp_cor:NetAssetsSummaryOfBusinessResults": "net_assets",
    "jpcrp_cor:TotalAssetsSummaryOfBusinessResults": "total_assets",
    "jpcrp_cor:NetAssetsPerShareSummaryOfBusinessResults": "bps",
    "jpcrp_cor:BasicEarningsLossPerShareSummaryOfBusinessResults": "eps",
    "jpcrp_cor:RateOfReturnOnEquitySummaryOfBusinessResults": "roe",
    "jpcrp_cor:EquityToAssetRatioSummaryOfBusinessResults": "equity_ratio",
}

_VALUE_FIELDS = (
    "revenue", "operating_income", "ordinary_income", "net_income",
    "net_assets", "total_assets", "bps", "eps", "roe", "equity_ratio",
)

# Plain context = consolidated block; _NonConsolidatedMember = 提出会社 (parent).
# Segment/other members are NOT fundamental summary rows and are ignored.
_CTX_RE = re.compile(
    r"^(Current|Prior([1-9]))Year(Duration|Instant)(_NonConsolidatedMember)?$"
)

_EMPTY_VALUES = {"", "－", "―", "-", "N/A", "NaN"}


def _parse_value(raw: str) -> float | None:
    s = raw.strip().strip('"').replace(",", "")
    if s in _EMPTY_VALUES:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_summary_csv(zip_bytes: bytes) -> list[dict[str, Any]]:
    """Extract per-relative-year summary values from an EDINET type=5 CSV zip.

    Returns a list of dicts sorted by ``relative_year`` (0 = current), each
    holding the mapped fields plus ``consolidated``. Consolidated (plain
    context) values take precedence over parent-only values for the same
    (year, field); a parent-only-filer still parses (consolidated=False).
    Fail-closed: unknown members, empty values, malformed rows are skipped.
    """
    zf = zipfile.ZipFile(BytesIO(zip_bytes))
    members = [n for n in zf.namelist() if "jpcrp" in n and n.endswith(".csv")]
    if not members:
        return []
    text = zf.read(members[0]).decode("utf-16")
    lines = text.splitlines()
    if not lines:
        return []
    header = [c.strip('"') for c in lines[0].split("\t")]
    try:
        i_eid = header.index("要素ID")
        i_ctx = header.index("コンテキストID")
        i_val = header.index("値")
    except ValueError:
        return []

    # (relative_year, field) -> (value, consolidated)
    acc: dict[tuple[int, str], tuple[float, bool]] = {}
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(header):
            continue
        eid = cells[i_eid].strip('"')
        field = ELEMENT_MAP.get(eid)
        if field is None:
            continue
        m = _CTX_RE.match(cells[i_ctx].strip('"'))
        if m is None:
            continue
        rel = 0 if m.group(1) == "Current" else int(m.group(2))
        consolidated = m.group(4) is None
        value = _parse_value(cells[i_val])
        if value is None:
            continue
        key = (rel, field)
        if key in acc and acc[key][1] and not consolidated:
            continue  # never let parent-only overwrite consolidated
        acc[key] = (value, consolidated)

    years: dict[int, dict[str, Any]] = {}
    for (rel, field), (value, consolidated) in acc.items():
        y = years.setdefault(rel, {"relative_year": rel, "consolidated": False})
        y[field] = value
        y["consolidated"] = y["consolidated"] or consolidated
    return [years[k] for k in sorted(years)]


def _shift_period_end(period_end: str, years_back: int) -> str | None:
    try:
        d = date.fromisoformat(period_end)
    except (TypeError, ValueError):
        return None
    try:
        return d.replace(year=d.year - years_back).isoformat()
    except ValueError:  # Feb 29 → Feb 28
        return d.replace(year=d.year - years_back, day=28).isoformat()


def build_records(
    years: list[dict[str, Any]],
    *,
    doc_id: str,
    symbol: str,
    doc_type_code: str,
    period_end: str | None,
    submitted_at: datetime,
) -> list[dict[str, Any]]:
    """Attach document identity + PIT stamps to parsed per-year values."""
    records = []
    for y in years:
        rel = int(y["relative_year"])
        fiscal_end = period_end if rel == 0 else (
            _shift_period_end(period_end, rel) if period_end else None
        )
        rec = {
            "doc_id": doc_id,
            "relative_year": rel,
            "symbol": symbol,
            "fiscal_period_end": fiscal_end,
            "period_basis": "reported" if rel == 0 else "estimated_shift",
            "published_ts": submitted_at.isoformat(),
            "doc_type_code": doc_type_code,
            "consolidated": 1 if y.get("consolidated") else 0,
            "source": "edinet_csv",
        }
        for f in _VALUE_FIELDS:
            rec[f] = y.get(f)
        records.append(rec)
    return records


_SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamental_snapshots (
    doc_id TEXT NOT NULL,
    relative_year INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    fiscal_period_end TEXT,
    period_basis TEXT,
    published_ts TEXT NOT NULL,
    doc_type_code TEXT,
    consolidated INTEGER,
    revenue REAL, operating_income REAL, ordinary_income REAL, net_income REAL,
    net_assets REAL, total_assets REAL, bps REAL, eps REAL, roe REAL,
    equity_ratio REAL,
    source TEXT,
    PRIMARY KEY (doc_id, relative_year)
);
CREATE INDEX IF NOT EXISTS idx_fund_symbol_period
    ON fundamental_snapshots (symbol, fiscal_period_end);
"""


def upsert_records(db_path: Path | str, records: list[dict[str, Any]]) -> int:
    """Idempotent insert (PK doc_id+relative_year). Returns rows newly added."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        added = 0
        cols = ("doc_id", "relative_year", "symbol", "fiscal_period_end",
                "period_basis", "published_ts", "doc_type_code", "consolidated",
                *_VALUE_FIELDS, "source")
        sql = (f"INSERT OR IGNORE INTO fundamental_snapshots ({','.join(cols)}) "
               f"VALUES ({','.join('?' * len(cols))})")
        for r in records:
            cur = conn.execute(sql, tuple(r.get(c) for c in cols))
            added += cur.rowcount
        conn.commit()
        return added
    finally:
        conn.close()


def stored_doc_ids(db_path: Path | str) -> set[str]:
    p = Path(db_path)
    if not p.exists():
        return set()
    conn = sqlite3.connect(str(p))
    try:
        conn.executescript(_SCHEMA)
        return {r[0] for r in conn.execute(
            "select distinct doc_id from fundamental_snapshots")}
    finally:
        conn.close()


class EdinetFundamentalsClient:
    """Minimal EDINET v2 client (list + type=5 CSV fetch), transport-injectable."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        import os

        self._api_key = api_key or os.environ.get("EDINET_API_KEY", "")
        self._transport = transport

    def _get(self, url: str, params: dict[str, Any], *, raw: bool = False) -> Any:
        if self._transport is not None:
            return self._transport(url, params, raw=raw)
        q = dict(params)
        q["Subscription-Key"] = self._api_key
        req = Request(f"{url}?{urlencode(q)}", headers={"User-Agent": "HTR/0.1"})
        with urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as resp:
            data = resp.read()
        return data if raw else json.loads(data.decode("utf-8"))

    def list_fundamental_documents(self, target_date: str) -> list[dict[str, Any]]:
        """Filings on ``target_date`` that are fundamental docs for listed
        companies: docTypeCode ∈ {120, 160}, secCode present, CSV available,
        investment-trust filings excluded."""
        data = self._get(f"{EDINET_BASE}/documents.json",
                         {"date": target_date, "type": 2})
        out = []
        for r in data.get("results", []) or []:
            if str(r.get("docTypeCode") or "") not in FUNDAMENTAL_DOC_TYPES:
                continue
            sec = r.get("secCode")
            if not sec or str(r.get("csvFlag") or "0") != "1":
                continue
            if "投資信託" in str(r.get("docDescription") or ""):
                continue
            code = str(sec)
            if len(code) == 5 and code.endswith("0"):
                code = code[:4]
            submitted = str(r.get("submitDateTime") or f"{target_date} 00:00")
            out.append({
                "doc_id": str(r.get("docID")),
                "symbol": f"{code}.T",
                "doc_type_code": str(r.get("docTypeCode")),
                "period_end": r.get("periodEnd"),
                "submitted_at": submitted,
                "filer": str(r.get("filerName") or ""),
            })
        return out

    def fetch_csv_zip(self, doc_id: str) -> bytes:
        blob = self._get(f"{EDINET_BASE}/documents/{doc_id}", {"type": 5}, raw=True)
        if not isinstance(blob, (bytes, bytearray)) or not bytes(blob).startswith(b"PK"):
            raise RuntimeError(f"EDINET doc {doc_id}: response is not a zip")
        return bytes(blob)
