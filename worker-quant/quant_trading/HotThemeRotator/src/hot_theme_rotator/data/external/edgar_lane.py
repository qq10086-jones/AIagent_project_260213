"""P34-09 — SEC EDGAR read-only replication lane.

Purpose, and its strict limit
------------------------------
The US market is NOT a capital destination for this account (see the §1 cost
argument: US via a Japanese retail broker costs ~4.0–9.9x the JP lot hurdle).
This lane exists for one thing only: **independent replication of a signal we
already study in Japan**, using a corpus that is free, keyless, and timestamped.

That is a real use. Japan's effective-sample bottleneck is severe, and a signal
that fails to replicate on a different market with far more history is a signal
worth doubting sooner. It is not a licence to open a US factor zoo — a second
market multiplies search breadth, and search breadth is what the deflation
denominator already struggles to bound.

filing date is NOT acceptance timestamp
----------------------------------------
EDGAR exposes both ``filingDate`` (the reporting date) and
``acceptanceDateTime`` (when EDGAR actually accepted the submission). They differ,
and the difference matters: a filing accepted at 18:30 ET was not tradable at that
day's close. :func:`pit_available_at` uses acceptance and applies a
next-session rule, and :class:`EdgarFiling` refuses to be built from filing date
alone when acceptance is available.

companyfacts is not an event-time source
-----------------------------------------
``companyfacts`` returns every XBRL fact a company ever reported, keyed by
period — it is a panel, not an event stream. Using it to date events silently
substitutes fiscal period for announcement time. :func:`assert_not_event_source`
exists to make that misuse loud.

Access policy
-------------
SEC requires a descriptive User-Agent identifying the requester, and rate-limits
aggressively. :func:`build_headers` refuses a placeholder UA rather than letting
the lane get the whole project blocked. Nothing here executes, sizes, or holds.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Mapping, Sequence

__all__ = [
    "SEC_RATE_LIMIT_PER_SEC",
    "EdgarLaneError",
    "EdgarFiling",
    "build_headers",
    "pit_available_at",
    "assert_not_event_source",
    "replication_scope_guard",
]

SEC_RATE_LIMIT_PER_SEC = 10          # SEC fair-access guidance
# One or more name tokens, then a contact address. Multi-word names are normal
# ("Jane Q Public jane@org.example"), so the name part repeats.
_UA_RE = re.compile(r"^(?:[^@\s]+\s+)+[^@\s]*@[^@\s]+\.[^@\s]+$")
_PLACEHOLDERS = ("your name", "example.com", "test@test", "user@user", "changeme")

# US market close 16:00 ET. A submission accepted after this is not actionable
# until the next session.
_US_CLOSE = time(16, 0)


class EdgarLaneError(ValueError):
    """Raised on a misuse that would corrupt point-in-time semantics."""


def build_headers(user_agent: str) -> dict[str, str]:
    """Build SEC-compliant request headers, refusing placeholder identities.

    SEC's fair-access policy requires a UA that identifies a real requester with
    contact details. A placeholder gets the source IP blocked, which would take
    down every other data pull from this machine — so it fails here, loudly,
    rather than at request time.
    """
    ua = (user_agent or "").strip()
    if not _UA_RE.match(ua):
        raise EdgarLaneError(
            f"user_agent must be 'Name email@domain' per SEC fair-access policy, "
            f"got {ua!r}")
    if any(p in ua.lower() for p in _PLACEHOLDERS):
        raise EdgarLaneError(
            f"user_agent {ua!r} looks like a placeholder; SEC blocks these and "
            f"the block lands on the whole host")
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


@dataclass(frozen=True)
class EdgarFiling:
    accession: str
    cik: str
    form: str
    filing_date: str                     # reporting date
    acceptance_datetime: str | None      # when EDGAR accepted it (ET)

    def __post_init__(self) -> None:
        if not self.accession or not self.cik:
            raise EdgarLaneError("accession and cik are required")
        try:
            datetime.fromisoformat(self.filing_date)
        except ValueError as exc:
            raise EdgarLaneError(
                f"filing_date must be ISO 8601, got {self.filing_date!r}") from exc
        if self.acceptance_datetime:
            try:
                datetime.fromisoformat(self.acceptance_datetime)
            except ValueError as exc:
                raise EdgarLaneError(
                    f"acceptance_datetime must be ISO 8601, got "
                    f"{self.acceptance_datetime!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pit_available_at(filing: EdgarFiling, *, require_acceptance: bool = True
                     ) -> str:
    """First date the filing could inform a trade.

    Uses ``acceptance_datetime``, not ``filing_date``: they differ, and a filing
    accepted after the 16:00 ET close was not actionable that session. With
    ``require_acceptance=True`` (the default) a filing lacking acceptance is
    refused rather than silently dated from ``filing_date``, because that
    substitution is invisible downstream and always biases toward earlier
    availability — the look-ahead direction.
    """
    if not filing.acceptance_datetime:
        if require_acceptance:
            raise EdgarLaneError(
                f"{filing.accession}: no acceptance_datetime. Dating this from "
                f"filing_date would assume availability up to a day early, which "
                f"is the look-ahead direction. Pass require_acceptance=False only "
                f"if you accept that bias explicitly.")
        return filing.filing_date

    accepted = datetime.fromisoformat(filing.acceptance_datetime)
    d = accepted.date()
    if accepted.time() >= _US_CLOSE:
        d = d + timedelta(days=1)
    return d.isoformat()


def assert_not_event_source(dataset: str) -> None:
    """Refuse to treat a period-keyed panel as an event stream."""
    panel_datasets = {"companyfacts", "companyconcept", "frames"}
    if dataset.lower() in panel_datasets:
        raise EdgarLaneError(
            f"{dataset!r} is keyed by fiscal PERIOD, not by announcement time; "
            f"using it as an event source substitutes period end for disclosure "
            f"date. Use the submissions index (with acceptanceDateTime) to date "
            f"events, and companyfacts only for the values."
        )


def replication_scope_guard(
    signals_studied_in_jp: Sequence[str],
    signals_requested: Sequence[str],
) -> None:
    """Allow only replication of signals already under study in the JP lane.

    The lane's justification is independent replication. A signal that exists
    only here is not a replication — it is a new search, on a second market,
    inflating the trial family that the deflation denominator has to cover.
    """
    known = {s.strip().lower() for s in signals_studied_in_jp}
    novel = [s for s in signals_requested if s.strip().lower() not in known]
    if novel:
        raise EdgarLaneError(
            f"signals {sorted(novel)} are not under study in the JP lane. This "
            f"lane replicates; it does not originate. Opening a new signal here "
            f"widens the search without widening the evidence, and every new "
            f"trial must be registered against the deflation denominator."
        )
