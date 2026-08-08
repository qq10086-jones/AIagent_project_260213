"""P34-01a — structured buyback (自己株式) events from TDnet disclosures.

The defect this fixes
---------------------
``tdnet_parser._CATEGORY_RULES`` has no buyback rule, and its ``order`` rule
matches ``株式の取得``. A buyback resolution is titled
「自己株式の取得に係る事項の決定に関するお知らせ」— which contains 株式の取得 —
so buyback resolutions were silently filed as ``order`` (business alliances and
large contracts). Measured on the stored corpus (2,344 disclosures,
2026-06-30..2026-08-07): 547 titles contain 自己株式, filed as governance 276 /
order 225 / other 30 / earnings 16. **Zero** were identifiable as buybacks.

Why a subtype taxonomy rather than one flag
-------------------------------------------
「自己株式」 covers events with opposite economics, and the largest group is not
a buyback at all:

  disposal (処分) 293 | execution_report (取得状況) 160 | cancellation (消却) 18
  resolution (取得決定) 17 | ...

**処分 is a treasury-share DISPOSAL** — shares going out, typically for stock
compensation. Treating it as a buyback would put the sign backwards on the
single largest subtype. So the parser assigns a subtype, and only ``resolution``
is the T1 event; everything else is context or exclusion.

Extraction honesty
------------------
The TDnet RSS feed carries titles, not documents. Some fields (amount cap, share
count, window, method) appear in titles only sometimes, and mostly live in the
PDF. Every field therefore carries an explicit extraction status, and anything
not present is ``None`` with ``field_status`` recording *why*. There is no
inference from a typical case: a buyback whose size we do not know is recorded
as unknown, because a fabricated size would flow straight into a
percent-of-market-cap sort.

Rule 3: this module classifies and records. It never sizes a position, never
scores a name, and emits no expected return.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BUYBACK_SUBTYPES",
    "T1_EVENT_SUBTYPE",
    "BuybackEvent",
    "BuybackParseError",
    "classify_buyback_subtype",
    "is_buyback_related",
    "parse_buyback_event",
    "link_execution_reports",
    "corpus_summary",
]


class BuybackParseError(ValueError):
    """Raised when a disclosure cannot be safely interpreted as a buyback event."""


# Only `resolution` is the T1 study event. The rest are context or exclusions.
BUYBACK_SUBTYPES = (
    "resolution",         # 取得に係る事項の決定 — the T1 event
    "modification",       # 開示事項の変更 / 取得枠の拡大 — changes a live programme
    "execution_report",   # 取得状況 — monthly progress
    "completion",         # 取得終了
    "cancellation",       # 消却 — retirement of held treasury shares
    "disposal",           # 処分 — shares going OUT. NOT a buyback.
    "other_treasury",     # mentions 自己株式, none of the above
)
T1_EVENT_SUBTYPE = "resolution"

_TREASURY = re.compile(r"自己株式|自社株")
_CORRECTION = re.compile(r"^[（(]訂正")
_MODIFICATION = re.compile(r"開示事項の変更|取得枠の拡大|取得枠の縮小")
_DISPOSAL = re.compile(r"自己株式の?処分|第三者割当")
_CANCELLATION = re.compile(r"自己株式の?消却|株式の消却")
_EXEC_REPORT = re.compile(r"取得状況")
_COMPLETION = re.compile(r"取得の?終了|取得完了")
_RESOLUTION = re.compile(r"自己株式の?取得.{0,20}(決定|決議)|取得に係る事項の決定")

# Acquisition method. The T1 pre-registration strata depend on this: ToSTNeT-3
# off-auction buybacks are executed differently from on-market auction purchases,
# and the literature does not treat them as one population.
_METHOD_TOSTNET = re.compile(r"ToSTNeT|東証立会外|立会外")
_METHOD_AUCTION = re.compile(r"市場買付|取引所市場における買付|立会内")
_METHOD_TENDER = re.compile(r"公開買付|TOB")

# Contamination: same-title co-announcements the event study must be able to exclude.
_EARNINGS = re.compile(r"決算|業績予想|業績の?修正|通期予想")
_DIVIDEND = re.compile(r"配当|増配|減配")
_SPLIT = re.compile(r"株式分割|株式併合")

# Numeric extraction from titles (rarely present; PDF is the real source).
_AMOUNT_OKU = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*億円")
_AMOUNT_MAN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*万円")
_SHARES = re.compile(r"(\d[\d,]*)\s*(?:万)?株")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")


def _normalize(title: str) -> str:
    """NFKC so full-width digits/parens compare like their ASCII forms."""
    return unicodedata.normalize("NFKC", title or "")


def is_buyback_related(title: str) -> bool:
    return bool(_TREASURY.search(_normalize(title)))


def classify_buyback_subtype(title: str) -> str | None:
    """Return the buyback subtype, or None if the title is not treasury-related.

    Order is load-bearing and tested. ``disposal`` and ``cancellation`` are
    checked BEFORE ``resolution`` because 「自己株式の消却」 and
    「自己株式の処分」 both contain 自己株式 and would otherwise be swept into a
    generic 取得 match; and a 処分 counted as a buyback puts the sign backwards
    on the largest subtype in the corpus.
    """
    t = _normalize(title)
    if not _TREASURY.search(t):
        return None
    if _EXEC_REPORT.search(t):
        return "execution_report"
    if _DISPOSAL.search(t):
        return "disposal"
    if _COMPLETION.search(t):
        return "completion"
    if _MODIFICATION.search(t):
        return "modification"
    if _CANCELLATION.search(t) and not _RESOLUTION.search(t):
        return "cancellation"
    if _RESOLUTION.search(t):
        return "resolution"
    return "other_treasury"


def _extract_method(title: str) -> tuple[str | None, str]:
    t = _normalize(title)
    if _METHOD_TOSTNET.search(t):
        return "tostnet", "title"
    if _METHOD_TENDER.search(t):
        return "tender_offer", "title"
    if _METHOD_AUCTION.search(t):
        return "auction", "title"
    return None, "absent_in_title_requires_document"


def _extract_amount_jpy(title: str) -> tuple[float | None, str]:
    t = _normalize(title).replace(",", "")
    m = _AMOUNT_OKU.search(t)
    if m:
        return float(m.group(1)) * 100_000_000, "title"
    m = _AMOUNT_MAN.search(t)
    if m:
        return float(m.group(1)) * 10_000, "title"
    return None, "absent_in_title_requires_document"


def _extract_percent(title: str) -> tuple[float | None, str]:
    m = _PERCENT.search(_normalize(title))
    if m:
        return float(m.group(1)), "title"
    return None, "absent_in_title_requires_document"


def compute_event_id(ticker: str, published_ts: str, subtype: str) -> str:
    """Idempotent event id.

    Keyed on (ticker, published_ts, subtype) rather than the title, so that a
    later corrective filing about the SAME event does not mint a new id purely
    because its title gained a 「（訂正）」 prefix.
    """
    payload = f"{ticker}|{published_ts}|{subtype}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


@dataclass(frozen=True)
class BuybackEvent:
    event_id: str
    ticker: str
    subtype: str
    published_ts: str          # PIT: when TDnet published
    collected_ts: str          # PIT: when we fetched it
    title: str
    url: str
    disclosure_id: str

    is_correction: bool = False
    supersedes_hint: str | None = None

    amount_cap_jpy: float | None = None
    share_cap: int | None = None
    percent_of_shares: float | None = None
    window_start: str | None = None
    window_end: str | None = None
    acquisition_method: str | None = None
    executed_amount_jpy: float | None = None
    completion_ratio: float | None = None

    contamination: tuple[str, ...] = ()
    field_status: dict[str, str] = field(default_factory=dict)
    parser_confidence: str = "low"      # low | medium | high
    parser_version: str = "p34-01a-v1"
    raw: dict[str, Any] | None = None

    @property
    def is_t1_event(self) -> bool:
        """Only an uncontaminated resolution enters the T1 primary sample."""
        return self.subtype == T1_EVENT_SUBTYPE and not self.contamination

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["contamination"] = list(self.contamination)
        d["is_t1_event"] = self.is_t1_event
        return d


def parse_buyback_event(disclosure: Mapping[str, Any]) -> BuybackEvent | None:
    """Parse one stored TdnetDisclosure dict into a BuybackEvent.

    Returns ``None`` when the disclosure is not treasury-related. Raises
    :class:`BuybackParseError` when it IS treasury-related but structurally
    unusable — malformed input fails closed rather than yielding an event with
    invented fields.
    """
    if not isinstance(disclosure, Mapping):
        raise BuybackParseError(f"disclosure must be a mapping, got {type(disclosure).__name__}")

    title = disclosure.get("title")
    if not isinstance(title, str) or not title.strip():
        raise BuybackParseError("disclosure has no usable title")

    subtype = classify_buyback_subtype(title)
    if subtype is None:
        return None

    for required in ("ticker", "published_ts", "collected_ts", "url", "disclosure_id"):
        value = disclosure.get(required)
        if not isinstance(value, str) or not value.strip():
            raise BuybackParseError(
                f"treasury disclosure missing required field {required!r}; "
                f"refusing to emit an event with a hole in its provenance"
            )
    for ts_field in ("published_ts", "collected_ts"):
        try:
            datetime.fromisoformat(disclosure[ts_field])
        except ValueError as exc:
            raise BuybackParseError(f"{ts_field} is not ISO 8601: {disclosure[ts_field]!r}") from exc

    t = _normalize(title)
    contamination: list[str] = []
    if _EARNINGS.search(t):
        contamination.append("earnings")
    if _DIVIDEND.search(t):
        contamination.append("dividend")
    if _SPLIT.search(t):
        contamination.append("split")
    if subtype == "resolution" and _CANCELLATION.search(t):
        contamination.append("cancellation_same_release")

    amount, amount_src = _extract_amount_jpy(title)
    percent, percent_src = _extract_percent(title)
    method, method_src = _extract_method(title)

    field_status = {
        "amount_cap_jpy": amount_src,
        "percent_of_shares": percent_src,
        "acquisition_method": method_src,
        "share_cap": "absent_in_title_requires_document",
        "window_start": "absent_in_title_requires_document",
        "window_end": "absent_in_title_requires_document",
        "executed_amount_jpy": "absent_in_title_requires_document",
        "completion_ratio": "absent_in_title_requires_document",
    }
    extracted = sum(1 for v in field_status.values() if v == "title")
    confidence = "high" if extracted >= 2 else ("medium" if extracted == 1 else "low")

    is_correction = bool(_CORRECTION.search(t))

    return BuybackEvent(
        event_id=compute_event_id(disclosure["ticker"], disclosure["published_ts"], subtype),
        ticker=disclosure["ticker"],
        subtype=subtype,
        published_ts=disclosure["published_ts"],
        collected_ts=disclosure["collected_ts"],
        title=title,
        url=disclosure["url"],
        disclosure_id=disclosure["disclosure_id"],
        is_correction=is_correction,
        supersedes_hint=_correction_target(title) if is_correction else None,
        amount_cap_jpy=amount,
        percent_of_shares=percent,
        acquisition_method=method,
        contamination=tuple(contamination),
        field_status=field_status,
        parser_confidence=confidence,
    )


def _correction_target(title: str) -> str | None:
    """The quoted original title inside a 「（訂正）「...」の一部訂正」 notice."""
    m = re.search(r"[「『](.+?)[」』]", _normalize(title))
    return m.group(1) if m else None


def link_execution_reports(
    events: Sequence[BuybackEvent],
    *,
    max_lag_days: int = 400,
) -> dict[str, list[str]]:
    """Map each resolution event_id to the execution reports that follow it.

    TDnet execution reports do not cite the resolution they belong to, so the
    link is inferred: a report attaches to the most recent PRIOR resolution for
    the same ticker, within ``max_lag_days``. This is an inference and is
    reported as one — a report with no prior resolution in the corpus is left
    unattached rather than being assigned to the nearest one in either
    direction, which would silently manufacture forward-looking links.
    """
    by_ticker: dict[str, list[BuybackEvent]] = {}
    for ev in events:
        by_ticker.setdefault(ev.ticker, []).append(ev)

    links: dict[str, list[str]] = {
        ev.event_id: [] for ev in events if ev.subtype == T1_EVENT_SUBTYPE
    }
    for ticker, rows in by_ticker.items():
        rows = sorted(rows, key=lambda e: e.published_ts)
        resolutions = [e for e in rows if e.subtype == T1_EVENT_SUBTYPE]
        if not resolutions:
            continue
        for ev in rows:
            if ev.subtype not in ("execution_report", "completion"):
                continue
            prior = [r for r in resolutions if r.published_ts <= ev.published_ts]
            if not prior:
                continue  # unattached on purpose: no backward links
            parent = prior[-1]
            try:
                lag = (datetime.fromisoformat(ev.published_ts)
                       - datetime.fromisoformat(parent.published_ts)).days
            except ValueError:
                continue
            if 0 <= lag <= max_lag_days:
                links[parent.event_id].append(ev.event_id)
    return links


def corpus_summary(events: Iterable[BuybackEvent]) -> dict[str, Any]:
    """Counts by subtype, contamination, and confidence — for the smoke artifact."""
    events = list(events)
    by_subtype: dict[str, int] = {}
    by_confidence: dict[str, int] = {}
    contaminated = 0
    corrections = 0
    for ev in events:
        by_subtype[ev.subtype] = by_subtype.get(ev.subtype, 0) + 1
        by_confidence[ev.parser_confidence] = by_confidence.get(ev.parser_confidence, 0) + 1
        if ev.contamination:
            contaminated += 1
        if ev.is_correction:
            corrections += 1
    t1 = [e for e in events if e.is_t1_event]
    return {
        "total_treasury_events": len(events),
        "by_subtype": dict(sorted(by_subtype.items())),
        "by_parser_confidence": dict(sorted(by_confidence.items())),
        "corrections": corrections,
        "contaminated": contaminated,
        "t1_primary_events": len(t1),
        "note": (
            "t1_primary_events counts UNCONTAMINATED resolutions only. "
            "disposal (処分) is a treasury-share disposal, not a buyback, and is "
            "never part of the T1 sample."
        ),
    }
