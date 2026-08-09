"""P36-02 — 決算短信 earnings events with point-in-time event dating.

Why not EDINET
--------------
EDINET's ``submitDateTime`` is a genuine PIT stamp, but its median lag of 87
days identifies the 有価証券報告書 — the statutory three-month annual report.
Jinushi's event is the **決算短信**, the earnings flash the TSE asks for within
about 45 days. Studying drift from the annual report measures a later, largely
priced-in disclosure. This module therefore dates events from TDnet.

The after-hours rule is the whole point of "event dating"
----------------------------------------------------------
Most Japanese earnings land AFTER the 15:30 close. A disclosure published at
16:00 cannot be traded that day, so its event date is the NEXT trading day —
Jinushi shifts these explicitly, and skipping the shift would credit the study
with a day of return nobody could have captured. :func:`event_date_for` applies
the shift and reports which side of the close each announcement fell on, so the
split is auditable rather than assumed.

Annual vs quarterly, and corrections
-------------------------------------
「四半期決算短信」 is a quarterly flash and a different event population;
「(訂正)」 filings restate an earlier announcement and must never be counted as
fresh news. Both are classified out of the primary sample rather than filtered
silently, so their counts stay visible.

Rule 3: event extraction only — no return, no score, no recommendation.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "TSE_CLOSE",
    "EarningsEvent",
    "EarningsEventError",
    "classify_tanshin",
    "event_date_for",
    "parse_earnings_event",
    "extract_earnings_events",
    "summarize_events",
]

# TSE close since 2024-11-05 (closing auction 15:25-15:30).
TSE_CLOSE = time(15, 30)

_TANSHIN = re.compile(r"決算短信")
_QUARTERLY = re.compile(r"四半期|中間")
_CORRECTION = re.compile(r"訂正|再訂正")
# Notices ABOUT a 短信 (delays, scheduling) are not the announcement itself.
_ABOUT_NOTICE = re.compile(r"に関するお知らせ|の開示|延期|遅延|日程")

SUBTYPE_ANNUAL = "annual"
SUBTYPE_QUARTERLY = "quarterly"
SUBTYPE_CORRECTION = "correction"
SUBTYPE_NOTICE = "notice_about_tanshin"


class EarningsEventError(ValueError):
    """Raised when a disclosure cannot be dated safely."""


def classify_tanshin(title: str) -> str | None:
    """Classify a 決算短信 title, or None when it is not one.

    Order matters: a 「(訂正)四半期決算短信」 is a correction first — counting it
    as a fresh quarterly event would double-count the original announcement.
    """
    if not isinstance(title, str) or not _TANSHIN.search(title):
        return None
    if _CORRECTION.search(title):
        return SUBTYPE_CORRECTION
    # A notice ABOUT the 短信 (e.g. "開示が期末後50日を超えたことに関するお知らせ")
    # is not the announcement; the announcement itself is titled plainly.
    if _ABOUT_NOTICE.search(title) and not re.search(r"決算短信〔", title):
        return SUBTYPE_NOTICE
    if _QUARTERLY.search(title):
        return SUBTYPE_QUARTERLY
    return SUBTYPE_ANNUAL


def event_date_for(
    published_ts: str,
    trading_days: Sequence[str],
) -> tuple[str | None, bool]:
    """(tradable event date, was_after_close).

    An announcement at or after the 15:30 close is dated to the next trading
    day. Returns ``(None, after_close)`` when no trading day at or after the
    target exists in the calendar — an event we cannot date is excluded, never
    approximated.
    """
    try:
        dt = datetime.fromisoformat(published_ts)
    except (TypeError, ValueError) as exc:
        raise EarningsEventError(
            f"published_ts must be ISO 8601, got {published_ts!r}") from exc
    after_close = dt.time() >= TSE_CLOSE
    target = dt.date() + timedelta(days=1) if after_close else dt.date()
    target_s = target.isoformat()
    for d in trading_days:                      # trading_days must be sorted
        if d >= target_s:
            return d, after_close
    return None, after_close


@dataclass(frozen=True)
class EarningsEvent:
    event_id: str
    symbol: str
    subtype: str
    published_ts: str        # PIT: when TDnet published it
    event_date: str          # first tradable session at/after publication
    after_close: bool
    title: str
    disclosure_id: str

    @property
    def is_primary(self) -> bool:
        """Only a plain ANNUAL flash enters the primary T2 sample."""
        return self.subtype == SUBTYPE_ANNUAL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_primary"] = self.is_primary
        return d


def parse_earnings_event(
    disclosure: Mapping[str, Any],
    trading_days: Sequence[str],
) -> EarningsEvent | None:
    """One stored TDnet disclosure → an EarningsEvent, or None if not a 短信."""
    title = disclosure.get("title")
    subtype = classify_tanshin(title if isinstance(title, str) else "")
    if subtype is None:
        return None
    for field in ("ticker", "published_ts", "disclosure_id"):
        v = disclosure.get(field)
        if not isinstance(v, str) or not v.strip():
            raise EarningsEventError(
                f"earnings disclosure missing {field!r}; refusing to date an "
                f"event with a hole in its provenance")
    event_date, after_close = event_date_for(disclosure["published_ts"], trading_days)
    if event_date is None:
        return None                              # undatable ⇒ excluded, not guessed
    return EarningsEvent(
        event_id=f"{disclosure['ticker']}|{disclosure['published_ts']}|{subtype}",
        symbol=disclosure["ticker"],
        subtype=subtype,
        published_ts=disclosure["published_ts"],
        event_date=event_date,
        after_close=after_close,
        title=title,
        disclosure_id=disclosure["disclosure_id"],
    )


def extract_earnings_events(
    disclosures: Iterable[Mapping[str, Any]],
    trading_days: Sequence[str],
) -> tuple[list[EarningsEvent], dict[str, int]]:
    """Extract all datable earnings events; returns (events, skip counts)."""
    events: list[EarningsEvent] = []
    skipped = {"not_tanshin": 0, "undatable": 0, "malformed": 0}
    for d in disclosures:
        title = d.get("title")
        if classify_tanshin(title if isinstance(title, str) else "") is None:
            skipped["not_tanshin"] += 1
            continue
        try:
            ev = parse_earnings_event(d, trading_days)
        except EarningsEventError:
            skipped["malformed"] += 1
            continue
        if ev is None:
            skipped["undatable"] += 1
            continue
        events.append(ev)
    return events, skipped


def summarize_events(events: Sequence[EarningsEvent]) -> dict[str, Any]:
    by_subtype: dict[str, int] = {}
    for e in events:
        by_subtype[e.subtype] = by_subtype.get(e.subtype, 0) + 1
    primary = [e for e in events if e.is_primary]
    after = sum(1 for e in primary if e.after_close)
    return {
        "total_tanshin": len(events),
        "by_subtype": dict(sorted(by_subtype.items())),
        "primary_annual": len(primary),
        "primary_symbols": len({e.symbol for e in primary}),
        "primary_after_close": after,
        "primary_after_close_fraction": (after / len(primary)) if primary else None,
        "note": ("after_close events are dated to the NEXT trading day; counting "
                 "them on the publication day would credit a return nobody "
                 "could have captured"),
    }
